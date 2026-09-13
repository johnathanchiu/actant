"""The worker half of remote tools: start the in-sandbox host and call tools through it.

The LLM loop stays on the worker; tool bodies and their state run inside the
sandbox (:mod:`actant.sandbox.host`), and only text and image bytes come back.
Every call is one ``sandbox.exec``, so any backend that can run a command can
host remote tools, with no port or tunnel to open.

:class:`RemoteTool` is the product's switch: give it a :class:`RemoteHost` to
run in the sandbox, or ``local=ctx`` to run the same function in-process.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import shlex
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import actant.sandbox.host as host_module
from actant.sandbox.host import (
    NO_HOST,
    PING,
    REQUESTS_DIR,
    SCRUB_ENV_VAR,
    SOCKET_TEMPLATE,
    Command,
    Field,
)
from actant.sandbox.base import Sandbox
from actant.tools.base import CallContext, ToolResult
from actant.tools.function import FunctionTool, ToolArguments, ToolFunction


@dataclass(frozen=True)
class RemoteResult:
    """A tool's response. ``images`` are ``(path as the tool gave it, bytes)``."""

    text: str
    images: list[tuple[str, bytes]] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def parse(cls, response: Mapping[str, object]) -> RemoteResult:
        images = [
            (str(image[Field.PATH]), base64.b64decode(str(image[Field.DATA])))
            for image in response.get(Field.IMAGES) or []  # pyright: ignore[reportGeneralTypeIssues]
        ]
        error = response.get(Field.ERROR)
        return cls(
            str(response.get(Field.TEXT) or ""), images, None if error is None else str(error)
        )


class RemoteHost:
    """One host process per sandbox, reached through ``sandbox.exec``.

    The default socket is keyed by ``sandbox.id``: local sandboxes share one
    machine's ``/tmp``, and a shared name would let one sandbox's host answer
    another's calls. Hashed because a unix socket path must fit in ~104 bytes.
    """

    def __init__(
        self, sandbox: Sandbox, *, socket: str | None = None, python: str = "python"
    ) -> None:
        self.sandbox = sandbox
        if socket is None:
            digest = hashlib.sha256(sandbox.id.encode()).hexdigest()[:16]
            socket = SOCKET_TEMPLATE.format(key=digest)
        self.socket = socket
        self.python = python
        self.log_path = socket.removesuffix(".sock") + ".log"
        self._start_lock = asyncio.Lock()
        # What ``start`` was given, so ``call`` can bring a dead host back.
        self._started: tuple[str, dict[str, object]] | None = None

    async def start(
        self,
        factory: str,
        *,
        cwd: str | None = None,
        after: str | None = None,
        scrub: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Launch the server unless one already answers. Returns once it answers.

        ``scrub`` is added to the sandbox's own ``ACTANT_SCRUB_ENV`` (``spec.scrub_env``)
        for the tools; see :func:`actant.sandbox.host.scrubbed_env`. The host itself
        is launched with ``keep_env`` so it keeps the keys its tools call services with.
        """
        self._started = (factory, {"cwd": cwd, "after": after, "scrub": scrub, "env": env})
        async with self._start_lock:
            if (await self._request({Field.TOOL: PING}, wait=0, timeout=30)).error is None:
                return
            argv = [self.python, "-m", host_module.__name__, Command.SERVE]
            argv += ["--socket", self.socket, "--factory", factory]
            if after:
                argv += ["--after", after]
            command = f"nohup {shlex.join(argv)} >{shlex.quote(self.log_path)} 2>&1 &"
            if scrub:
                extra = shlex.quote(",".join(scrub))
                var = SCRUB_ENV_VAR
                command = f'{var}="${{{var}:+${var},}}"{extra} {command}'
            launched = await self.sandbox.exec(
                ["sh", "-c", command], cwd=cwd, timeout=30, env=env, keep_env=True
            )
            if launched.returncode != 0:
                raise RuntimeError(f"could not launch the actant host: {launched.stderr[-2000:]}")
            pong = await self._request({Field.TOOL: PING}, wait=30, timeout=60)
            if pong.error is not None:
                raise RuntimeError(
                    f"actant host did not come up: {pong.error}\n{await self.log()}"
                )

    async def call(
        self,
        context: str,
        tool: str,
        args: Mapping[str, object],
        *,
        init: Mapping[str, object] | None = None,
        timeout: float = 600,
    ) -> RemoteResult:
        """Run one tool. A host that died (OOM, a crash) is restarted once and the call retried.

        The short wait is safe because ``start`` returns only once the host answers;
        a longer one would only delay noticing a dead host.
        """
        request = {
            Field.CONTEXT: context,
            Field.INIT: init,
            Field.TOOL: tool,
            Field.ARGS: dict(args),
        }
        result = await self._request(request, wait=5, timeout=timeout)
        if self._started is not None and NO_HOST in (result.error or ""):
            factory, options = self._started
            try:
                await self.start(factory, **options)  # pyright: ignore[reportArgumentType]
            except RuntimeError as error:
                return RemoteResult("", error=f"{result.error}\nrestart failed: {error}")
            result = await self._request(request, wait=5, timeout=timeout)
        return result

    async def log(self) -> str:
        result = await self.sandbox.exec(["cat", self.log_path], timeout=30)
        return result.stdout + result.stderr

    async def _request(self, request: object, *, wait: float, timeout: float) -> RemoteResult:
        # A file, not argv: Linux caps one argument at 128 KiB. ``call`` deletes it.
        request_file = f"{REQUESTS_DIR}/{uuid.uuid4().hex}.json"
        argv = [self.python, "-m", host_module.__name__, Command.CALL, "--socket", self.socket]
        argv += ["--request-file", request_file, "--wait", str(wait)]
        try:
            await self.sandbox.write(request_file, json.dumps(request).encode())
            result = await self.sandbox.exec(argv, timeout=timeout)
        except Exception as error:  # noqa: BLE001 -- a tool result, never a crashed turn
            return RemoteResult("", error=f"host call failed: {type(error).__name__}: {error}")
        lines = result.stdout.strip().splitlines()
        if result.returncode == 0 and lines:
            try:
                return RemoteResult.parse(json.loads(lines[-1]))
            except (ValueError, KeyError, TypeError):
                pass
        return RemoteResult(
            result.stdout[-2000:],
            error=f"host call failed (exit {result.returncode}): {result.stderr[-2000:]}",
        )


class RemoteTool(FunctionTool):
    """A ``fn(ctx, **args)`` product tool, run in the sandbox host or in-process.

    The schema comes from the signature minus ``ctx``, exactly as a
    :class:`FunctionTool` derives it. The sandbox resolves the function by
    ``module:qualname``, so it must be importable there under the same name.
    """

    _takes_context = True

    def __init__(
        self,
        fn: ToolFunction,
        *,
        context: str,
        init: Mapping[str, object] | None = None,
        host: RemoteHost | None = None,
        local: object | None = None,
    ) -> None:
        if (host is None) == (local is None):
            raise TypeError(
                "Pass exactly one of `host` (run in the sandbox) or `local` (a context)"
            )
        super().__init__(fn)
        self.context = context
        self.init = init
        self.host = host
        self.local = local
        self.path = f"{fn.__module__}:{fn.__qualname__}"

    async def _execute(self, params: ToolArguments, ctx: CallContext | None) -> ToolResult:
        del ctx
        if self.host is not None:
            result = await self.host.call(self.context, self.path, params, init=self.init)
        else:
            # The same adaptation the host applies, so both modes return alike.
            try:
                output = await host_module.invoke(self.function, self.local, params)
                response = host_module.adapt(output)
            except Exception as error:
                response = host_module.failure(error)
            result = RemoteResult.parse(response)
        return as_tool_result(result)


def as_tool_result(result: RemoteResult) -> ToolResult:
    """Text as output; images as the base64 content blocks the LLM adapters accept."""
    if result.error is not None:
        return ToolResult.fail(f"{result.error}\n{result.text}".rstrip())
    if not result.images:
        return ToolResult.ok(result.text)
    blocks: list[dict[str, object]] = [{"type": "text", "text": result.text}]
    for path, data in result.images:
        blocks.append({"type": "text", "text": f"Image {path}:"})
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mimetypes.guess_type(path)[0] or "image/png",
                    "data": base64.b64encode(data).decode(),
                },
            }
        )
    return ToolResult(output=result.text, content_blocks=blocks)
