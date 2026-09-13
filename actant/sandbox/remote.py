"""The worker half of remote tools: start the in-sandbox host and call tools through it.

The LLM loop stays on the worker; tool bodies and their state run inside the
sandbox (:mod:`actant.sandbox.host`), and only text and image bytes come back.
Every call is one ``sandbox.exec``, so any backend that can run a command can
host remote tools, with no port or tunnel to open.

:class:`RemoteTool` is the product's switch: give it a :class:`RemoteHost` to
run in the sandbox, or ``local=ctx`` to run the same function in-process.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import actant.sandbox.host as host_module
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
            (str(image["path"]), base64.b64decode(str(image["data"])))
            for image in response.get("images") or []  # pyright: ignore[reportGeneralTypeIssues]
        ]
        error = response.get("error")
        return cls(str(response.get("text") or ""), images, None if error is None else str(error))


class RemoteHost:
    """One host process per sandbox, reached through ``sandbox.exec``."""

    def __init__(
        self, sandbox: Sandbox, *, socket: str = "/tmp/actant-host.sock", python: str = "python"
    ) -> None:
        self.sandbox = sandbox
        self.socket = socket
        self.python = python
        self.log_path = socket.removesuffix(".sock") + ".log"

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

        ``scrub`` reaches the tools as ``ACTANT_SCRUB_ENV``; see
        :func:`actant.sandbox.host.scrubbed_env`.
        """
        if (await self._request({"tool": host_module.PING}, wait=0, timeout=30)).error is None:
            return
        argv = [self.python, "-m", "actant.sandbox.host", "serve"]
        argv += ["--socket", self.socket, "--factory", factory]
        if after:
            argv += ["--after", after]
        command = f"nohup {shlex.join(argv)} >{shlex.quote(self.log_path)} 2>&1 &"
        launched = await self.sandbox.exec(
            ["sh", "-c", command],
            cwd=cwd,
            timeout=30,
            env={**(env or {}), "ACTANT_SCRUB_ENV": ",".join(scrub)},
        )
        if launched.returncode != 0:
            raise RuntimeError(f"could not launch the actant host: {launched.stderr[-2000:]}")
        pong = await self._request({"tool": host_module.PING}, wait=30, timeout=60)
        if pong.error is not None:
            raise RuntimeError(f"actant host did not come up: {pong.error}\n{await self.log()}")

    async def call(
        self,
        context: str,
        tool: str,
        args: Mapping[str, object],
        *,
        init: Mapping[str, object] | None = None,
        timeout: float = 600,
    ) -> RemoteResult:
        request = {"context": context, "init": init, "tool": tool, "args": dict(args)}
        return await self._request(request, wait=30, timeout=timeout)

    async def log(self) -> str:
        result = await self.sandbox.exec(["cat", self.log_path], timeout=30)
        return result.stdout + result.stderr

    async def _request(self, request: object, *, wait: float, timeout: float) -> RemoteResult:
        encoded = base64.b64encode(json.dumps(request).encode()).decode()
        argv = [self.python, "-m", "actant.sandbox.host", "call", "--socket", self.socket]
        argv += ["--request", encoded, "--wait", str(wait)]
        result = await self.sandbox.exec(argv, timeout=timeout)
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
