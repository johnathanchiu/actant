"""Toolsets: a plain class whose public methods are tools.

A product writes an ordinary class, with no actant import::

    class Notes:
        @classmethod
        async def open(cls, folder: str) -> "Notes": ...   # optional; else Notes(**init)

        async def add(self, title: str, body: str = "") -> str:
            \"\"\"Save a note.\"\"\"   # the docstring is the tool description

Tools are the public functions (``async def`` or plain ``def``) on the class and
its bases, minus ``open`` and ``close``. A plain ``def`` runs in a worker thread;
an ``async def`` runs on the host's event loop, so blocking or CPU-heavy work in
one stalls every concurrent call until it awaits. Each schema comes from the method's signature without
``self`` (:func:`actant.sandbox.host.parameters_model`); arguments are
validated against it on both sides, so methods receive the annotated types.

A method returns ``str``, an object with ``.text`` and ``.images`` (file paths
or bytes), or any JSON-encodable value. Images become the base64 image content
blocks the LLM adapters accept.

The same :func:`tools` run the methods through any :class:`Runner`:
:class:`LocalRunner` in-process, :class:`RemoteRunner` against a host endpoint,
or :class:`SandboxRunner` against the calling thread's sandbox (a name in the
spec's ``toolsets``). Every runner encodes results the same way, so switching between
them changes where the code runs and nothing the model sees.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Mapping
from http import HTTPStatus
from typing import Protocol

from pydantic import ValidationError

from actant.core import JSONObject
import actant.sandbox.host as host
from actant.sandbox.base import Endpoint
from actant.tools.base import BaseToolInvocation, CallContext, ToolResult, ToolSchema

DEFAULT_CALL_TIMEOUT_S = 600.0


class Runner(Protocol):
    """Runs one toolset method and returns what the model sees. ``needs_sandbox`` asks
    the runtime to put the thread's sandbox on the :class:`CallContext`."""

    needs_sandbox: bool

    async def call(
        self, method: str, args: Mapping[str, object], ctx: CallContext
    ) -> ToolResult: ...


def to_tool_result(response: Mapping[str, object]) -> ToolResult:
    """A host response (``{text, images, error}``) as a :class:`ToolResult`."""
    text = str(response.get("text") or "")
    error = response.get("error")
    if error is not None:
        return ToolResult.fail("\n".join(part for part in (str(error), text) if part))
    images = response.get("images") or []
    if not isinstance(images, list) or not images:
        return ToolResult.ok(text)
    # LLM APIs reject empty text blocks.
    blocks: list[dict[str, object]] = [{"type": "text", "text": text}] if text else []
    for image in images:
        blocks.append({"type": "text", "text": f"Image {image['name']}:"})
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image["media_type"],
                    "data": image["data_b64"],
                },
            }
        )
    return ToolResult(output=text, content_blocks=blocks)


class LocalRunner:
    """Runs methods on an instance in this process."""

    needs_sandbox = False

    def __init__(self, instance: object) -> None:
        self.instance = instance

    async def call(self, method: str, args: Mapping[str, object], ctx: CallContext) -> ToolResult:
        del ctx
        return to_tool_result(await host.call_method(self.instance, method, args))


class RemoteRunner:
    """Runs methods of the host's ``toolset``. ``key`` names the host-side instance,
    created from ``init`` on its first call."""

    needs_sandbox = False

    def __init__(
        self,
        endpoint: Endpoint,
        toolset: str,
        key: str,
        init: Mapping[str, object] | None = None,
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        self.endpoint = endpoint
        self.toolset = toolset
        self.key = key
        self.init = dict(init or {})
        self.timeout = timeout

    async def call(self, method: str, args: Mapping[str, object], ctx: CallContext) -> ToolResult:
        del ctx
        return await call_host(
            self.endpoint,
            self.toolset,
            method,
            args,
            key=self.key,
            init=self.init,
            timeout=self.timeout,
        )


class SandboxRunner:
    """Runs methods of ``toolset`` (a name in ``SandboxSpec.toolsets``) on the host in the
    calling thread's sandbox, one instance per thread, created from ``init``. A connect
    credential the host rejects is refreshed once through :meth:`Sandbox.endpoint`."""

    needs_sandbox = True

    def __init__(
        self,
        toolset: str,
        init: Mapping[str, object] | None = None,
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        self.toolset = toolset
        self.init = dict(init or {})
        self.timeout = timeout

    async def call(self, method: str, args: Mapping[str, object], ctx: CallContext) -> ToolResult:
        sandbox = ctx.sandbox
        endpoint = await sandbox.endpoint() if sandbox is not None else None
        if sandbox is None or endpoint is None:
            return ToolResult.fail(
                "this thread's sandbox serves no toolsets (SandboxSpec.toolsets)"
            )
        body = _body(self.toolset, ctx.thread_id, self.init, method, args)
        status, result = await _send(endpoint, body, self.timeout)
        if status == HTTPStatus.UNAUTHORIZED:  # rejected before running: safe to retry
            endpoint = await sandbox.endpoint(refresh=True)
            assert endpoint is not None
            _, result = await _send(endpoint, body, self.timeout)
        return result


async def call_host(
    endpoint: Endpoint,
    toolset: str,
    method: str,
    args: Mapping[str, object],
    *,
    key: str,
    init: Mapping[str, object] | None = None,
    timeout: float = DEFAULT_CALL_TIMEOUT_S,
) -> ToolResult:
    """``POST /v1/call`` to a toolset host. Transport failures become failed results."""
    body = _body(toolset, key, init or {}, method, args)
    return (await _send(endpoint, body, timeout))[1]


def _body(
    toolset: str, key: str, init: Mapping[str, object], method: str, args: Mapping[str, object]
) -> bytes:
    request = {"toolset": toolset, "key": key, "init": dict(init), "method": method}
    return json.dumps({**request, "args": dict(args)}).encode()


async def _send(endpoint: Endpoint, body: bytes, timeout: float) -> tuple[int | None, ToolResult]:
    try:
        status, data = await asyncio.to_thread(host.post, endpoint, host.CALL_PATH, body, timeout)
    except OSError as error:
        return None, ToolResult.fail(f"toolset host unreachable at {endpoint.url}: {error}")
    try:
        response = json.loads(data)
    except ValueError:
        response = None
    if not isinstance(response, dict):
        tail = data[-1000:].decode(errors="replace")
        return status, ToolResult.fail(f"toolset host returned HTTP {status}: {tail}")
    return status, to_tool_result(response)


class ToolsetInvocation(BaseToolInvocation[dict[str, object], ToolResult]):
    def __init__(self, tool: ToolsetTool, params: dict[str, object], ctx: CallContext) -> None:
        super().__init__(params)
        self._tool = tool
        self._ctx = ctx

    def get_description(self) -> str:
        return f"Running {self._tool.name}"

    async def execute(self) -> ToolResult:
        return await self._tool.runner.call(self._tool.name, self.params, self._ctx)


class ToolsetTool:
    """One toolset method as an actant :class:`~actant.tools.base.Tool`."""

    def __init__(self, cls: type, method: str, runner: Runner) -> None:
        self.name = method
        self.runner = runner
        self.needs_sandbox = runner.needs_sandbox
        self._model = host.parameters_model(cls, method)
        description = inspect.getdoc(getattr(cls, method)) or f"Run {method}."
        self._schema: ToolSchema = {
            "type": "function",
            "function": {
                "name": method,
                "description": description,
                "parameters": self._model.model_json_schema(),
            },
        }

    @property
    def schema(self) -> ToolSchema:
        return self._schema

    async def build(self, params: JSONObject, ctx: CallContext) -> ToolsetInvocation:
        try:
            validated = self._model.model_validate(params).model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(f"Invalid arguments for {self.name}: {exc}") from exc
        return ToolsetInvocation(self, validated, ctx)


def tools(cls: type, runner: Runner) -> list[ToolsetTool]:
    """``cls``'s tools, each run through ``runner``."""
    return [ToolsetTool(cls, method, runner) for method in host.public_methods(cls)]


def toolset_schema(cls: type) -> list[ToolSchema]:
    """The tool schemas of ``cls``, without choosing where they run."""
    return [tool.schema for tool in tools(cls, LocalRunner(None))]
