"""Toolsets: a plain class whose public async methods are tools.

A product writes an ordinary class, with no actant import::

    class Notes:
        @classmethod
        async def open(cls, folder: str) -> "Notes": ...   # optional; else Notes(**init)

        async def add(self, title: str, body: str = "") -> str:
            \"\"\"Save a note.\"\"\"   # the docstring is the tool description

Tools are the public coroutine functions on the class and its bases, minus
``open`` and ``close``. Each schema comes from the method's signature without
``self``; annotations are resolved one parameter at a time, so a return
annotation that only exists for type checkers does not matter.

A method returns ``str``, an object with ``.text`` and ``.images`` (file paths
or bytes), or any JSON-encodable value. Images become the base64 image content
blocks the LLM adapters accept.

The same :func:`tools` run the methods through any :class:`Runner`:
:class:`LocalRunner` in-process, :class:`RemoteRunner` against a host endpoint,
or :class:`SandboxRunner` against the calling thread's sandbox (the spec's
``toolset``). Every runner encodes results the same way, so switching between
them changes where the code runs and nothing the model sees.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Mapping
from http.client import HTTPConnection, HTTPSConnection
from typing import Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from actant.core import JSONObject
import actant.sandbox.host as host
from actant.sandbox.base import Endpoint
from actant.tools.base import BaseToolInvocation, CallContext, ToolResult, ToolSchema

DEFAULT_CALL_TIMEOUT_S = 600.0


class Runner(Protocol):
    """Runs one toolset method and returns what the model sees."""

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
    blocks: list[dict[str, object]] = [{"type": "text", "text": text}]
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

    def __init__(self, instance: object) -> None:
        self.instance = instance

    async def call(self, method: str, args: Mapping[str, object], ctx: CallContext) -> ToolResult:
        del ctx
        return to_tool_result(await host.call_method(self.instance, method, args))


class RemoteRunner:
    """Runs methods on a toolset host. ``key`` names the host-side instance, created
    from ``init`` on its first call."""

    def __init__(
        self,
        endpoint: Endpoint,
        key: str,
        init: Mapping[str, object] | None = None,
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        self.endpoint = endpoint
        self.key = key
        self.init = dict(init or {})
        self.timeout = timeout

    async def call(self, method: str, args: Mapping[str, object], ctx: CallContext) -> ToolResult:
        del ctx
        return await call_host(self.endpoint, self.key, self.init, method, args, self.timeout)


class SandboxRunner:
    """Runs methods on the host serving the calling thread's sandbox (``SandboxSpec.toolset``),
    one instance per thread, created from ``init``."""

    needs_sandbox = True

    def __init__(
        self, init: Mapping[str, object] | None = None, *, timeout: float = DEFAULT_CALL_TIMEOUT_S
    ) -> None:
        self.init = dict(init or {})
        self.timeout = timeout

    async def call(self, method: str, args: Mapping[str, object], ctx: CallContext) -> ToolResult:
        endpoint = ctx.sandbox.endpoint if ctx.sandbox is not None else None
        if endpoint is None:
            return ToolResult.fail("this thread's sandbox serves no toolset (SandboxSpec.toolset)")
        return await call_host(endpoint, ctx.thread_id, self.init, method, args, self.timeout)


async def call_host(
    endpoint: Endpoint,
    key: str,
    init: Mapping[str, object],
    method: str,
    args: Mapping[str, object],
    timeout: float = DEFAULT_CALL_TIMEOUT_S,
) -> ToolResult:
    """``POST /v1/call`` to a toolset host. Transport failures become failed results."""
    body = json.dumps({"key": key, "init": dict(init), "method": method, "args": dict(args)})
    try:
        status, data = await asyncio.to_thread(_post, endpoint, body.encode(), timeout)
    except OSError as error:
        return ToolResult.fail(f"toolset host unreachable at {endpoint.url}: {error}")
    try:
        response = json.loads(data)
    except ValueError:
        response = None
    if not isinstance(response, dict):
        tail = data[-1000:].decode(errors="replace")
        return ToolResult.fail(f"toolset host returned HTTP {status}: {tail}")
    return to_tool_result(response)


def _post(endpoint: Endpoint, body: bytes, timeout: float) -> tuple[int, bytes]:
    url = urlsplit(endpoint.url)
    connection = (HTTPSConnection if url.scheme == "https" else HTTPConnection)(
        url.netloc, timeout=timeout
    )
    try:
        headers = {"Content-Type": "application/json", **endpoint.headers}
        connection.request("POST", url.path.rstrip("/") + host.CALL_PATH, body, headers)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def _parameters_model(cls: type, method: str) -> type[BaseModel]:
    function = inspect.getattr_static(cls, method)
    parameters = list(inspect.signature(function).parameters.values())[1:]  # ``self``
    fields: dict[str, tuple[object, object]] = {}
    for parameter in parameters:
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            raise TypeError(f"{cls.__name__}.{method} must use named parameters only")
        annotation = parameter.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"{cls.__name__}.{method}: parameter {parameter.name!r} needs a type")
        if isinstance(annotation, str):
            # ``from __future__ import annotations``: resolve in the method's module.
            annotation = eval(annotation, function.__globals__, {cls.__name__: cls})  # noqa: S307
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (annotation, default)
    return create_model(  # pyright: ignore[reportCallIssue, reportArgumentType]
        f"{cls.__name__}{method.title().replace('_', '')}Params",
        __config__=ConfigDict(extra="forbid"),
        **fields,  # pyright: ignore[reportArgumentType]
    )


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
        self.needs_sandbox = bool(getattr(runner, "needs_sandbox", False))
        self._model = _parameters_model(cls, method)
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
