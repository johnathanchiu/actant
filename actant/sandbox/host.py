"""The in-sandbox half of remote tools: a server that keeps tool state next to the files.

Product tools are plain ``async def fn(ctx, **args)`` functions that read and
write files and keep in-memory state on ``ctx``. Running their bodies on the
worker would ship every file across the sandbox boundary; running each call as
a fresh process would lose ``ctx``. So one long-lived process inside the
sandbox owns the contexts and runs the tools, and the worker sends it a JSON
line per call through ``python -m actant.sandbox.host call`` (the only thing a
sandbox's ``exec`` can reach). See :mod:`actant.sandbox.remote` for the worker
side.

Stdlib only: the sandbox image installs actant without the runtime or any
backend extra, and this module must import there.

Protocol, one JSON line each way per connection::

    {"context": str, "init": dict | null, "tool": "pkg.mod:fn", "args": dict}  (Field)
    {"text": str, "images": [{"path": str, "data": base64}], "error": str | null}
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import fcntl
import importlib
import inspect
import json
import mimetypes
import os
import socket as socketlib
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path

PING = "__ping__"
#: Names the variables ``exec`` and :func:`scrubbed_env` remove, comma-separated.
SCRUB_ENV_VAR = "ACTANT_SCRUB_ENV"
#: Per-sandbox state, relative to the sandbox root; kept out of storage sync.
STATE_DIR = ".actant"
#: Request files ``call`` reads and deletes.
REQUESTS_DIR = f"{STATE_DIR}/requests"
#: Default socket; ``{key}`` is a short hash of the sandbox id (paths cap at ~104 bytes).
SOCKET_TEMPLATE = "/tmp/actant-host-{key}.sock"
#: What ``call`` prints when nothing listens; the worker restarts the host on it.
NO_HOST = "no actant host"


class Command(StrEnum):
    SERVE = "serve"
    CALL = "call"


class Field(StrEnum):
    """Keys of the request and response lines."""

    CONTEXT = "context"
    INIT = "init"
    TOOL = "tool"
    ARGS = "args"
    TEXT = "text"
    IMAGES = "images"
    PATH = "path"
    DATA = "data"
    ERROR = "error"


def response(
    text: str, images: Sequence[dict[str, str]] = (), error: str | None = None
) -> dict[str, object]:
    return {Field.TEXT: text, Field.IMAGES: list(images), Field.ERROR: error}


def scrubbed_env() -> dict[str, str]:
    """``os.environ`` minus the names in ``ACTANT_SCRUB_ENV`` (comma-separated).

    The server itself keeps every variable: the factory and the tools call
    services with those keys. Pass this as ``env=`` to any subprocess that runs
    model-written code, so the script never sees them.

    Best effort, not a security boundary: code running as the same user can
    still read ``/proc/<pid>/environ`` of the server or talk to its socket.
    """
    scrub = {name.strip() for name in os.environ.get(SCRUB_ENV_VAR, "").split(",")}
    return {key: value for key, value in os.environ.items() if key not in scrub}


def resolve(path: str) -> Callable[..., object]:
    """``"pkg.mod:qual.name"`` to the object it names."""
    module_name, _, qualname = path.partition(":")
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target  # pyright: ignore[reportReturnType]


def adapt(output: object) -> dict[str, object]:
    """A tool's return value as a response: text, plus image bytes read from disk.

    An object with ``.text`` and ``.images`` (paths) is how a tool hands back
    renders; the bytes are read here because the files only exist in the sandbox.
    Only image files under the working directory are read: a path is model-steerable,
    and anything else (``/proc/self/environ``, a key file) would leave the sandbox.
    """
    if isinstance(output, str):
        return response(output)
    if hasattr(output, "text") and hasattr(output, "images"):
        text = str(output.text)  # pyright: ignore[reportAttributeAccessIssue]
        images = []
        cwd = Path.cwd().resolve()
        for path in output.images:  # pyright: ignore[reportAttributeAccessIssue]
            target = Path(path).resolve()
            is_image = (mimetypes.guess_type(target.name)[0] or "").startswith("image/")
            if not is_image or cwd not in target.parents:
                text += f"\n(dropped {path}: not an image file under the working directory)"
                continue
            data = base64.b64encode(target.read_bytes()).decode()
            images.append({Field.PATH: str(path), Field.DATA: data})
        return response(text, images)
    try:
        text = json.dumps(output, default=str)
    except (TypeError, ValueError):
        text = str(output)
    return response(text)


def failure(error: BaseException) -> dict[str, object]:
    """An exception as a response the model can correct from; the tail keeps it short."""
    tail = "".join(traceback.format_exception(error)[-3:])
    return response(tail, error=f"{type(error).__name__}: {error}")


async def invoke(fn: Callable[..., object], ctx: object, args: dict[str, object]) -> object:
    if inspect.iscoroutinefunction(fn):
        return await fn(ctx, **args)
    return await asyncio.to_thread(fn, ctx, **args)


class Server:
    def __init__(self, factory: str, after: str | None) -> None:
        self.factory = resolve(factory)
        self.after = after
        # Futures, not contexts: two parallel first calls must share one factory run.
        self.contexts: dict[str, asyncio.Future[object]] = {}
        self._after_due = asyncio.Event()

    async def context(self, key: str, init: dict[str, object] | None) -> object:
        if key not in self.contexts:

            async def create() -> object:
                made = self.factory(key, init)
                return await made if inspect.isawaitable(made) else made

            self.contexts[key] = asyncio.ensure_future(create())
        future = self.contexts[key]
        try:
            return await future
        except BaseException:
            # Let the next call retry the factory, unless a retry already replaced it.
            if self.contexts.get(key) is future:
                del self.contexts[key]
            raise

    async def handle(self, request: dict[str, object]) -> dict[str, object]:
        if request.get(Field.TOOL) == PING:
            return response("pong")
        try:
            init = request.get(Field.INIT)
            args = request.get(Field.ARGS) or {}
            if not isinstance(init, dict | None) or not isinstance(args, dict):
                raise TypeError("`init` and `args` must be JSON objects")
            ctx = await self.context(str(request[Field.CONTEXT]), init)
            return adapt(await invoke(resolve(str(request[Field.TOOL])), ctx, args))
        except Exception as error:
            return failure(error)
        finally:
            if self.after:
                self._after_due.set()

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            try:
                reply = await self.handle(json.loads(line))
            except ValueError as error:
                reply = failure(error)
            writer.write(json.dumps(reply).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()

    async def run_after(self) -> None:
        """Run ``after`` once per burst of requests: one running, at most one pending."""
        while True:
            await self._after_due.wait()
            self._after_due.clear()
            process = await asyncio.create_subprocess_shell(
                self.after or "", stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            if process.returncode:
                # Nobody waits on the hook (a failed push is otherwise invisible); stderr is the log.
                tail = stderr.decode(errors="replace")[-2000:]
                print(
                    f"after hook exited {process.returncode}: {tail}", file=sys.stderr, flush=True
                )


async def serve(socket_path: str, factory: str, after: str | None) -> None:
    # Held for the server's life. Two launches racing each other (two RemoteHosts on
    # one sandbox) would both see no answer on the socket before either binds, so a
    # connect check is not enough; the loser must not unlink the winner's socket.
    lock = open(socket_path + ".lock", "w")  # noqa: SIM115 -- released only on exit
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"an actant host is already serving {socket_path}") from None
    server = Server(factory, after)
    Path(socket_path).unlink(missing_ok=True)  # a stale file from a dead server
    # Responses carry base64 images; the default 64 KiB line limit is too small.
    unix = await asyncio.start_unix_server(server.connection, socket_path, limit=2**30)
    after_task = asyncio.create_task(server.run_after()) if after else None
    async with unix:
        await unix.serve_forever()
    del after_task, lock


def call(socket_path: str, request_file: str, wait: float) -> int:
    """Send one request and print the response line. Waits for a booting server.

    The request comes from a file, not argv: one argument is capped at 128 KiB on
    Linux, which a tool's ``write`` content easily exceeds. The file is deleted here.
    """
    request_path = Path(request_file)
    request = request_path.read_bytes()
    request_path.unlink(missing_ok=True)
    deadline = time.monotonic() + wait
    while True:
        sock = socketlib.socket(socketlib.AF_UNIX, socketlib.SOCK_STREAM)
        try:
            sock.connect(socket_path)
            break
        except (FileNotFoundError, ConnectionRefusedError) as error:
            sock.close()
            if time.monotonic() >= deadline:
                print(f"{NO_HOST} at {socket_path}: {error}", file=sys.stderr)
                return 1
            time.sleep(0.05)
    with sock, sock.makefile("rwb") as stream:
        stream.write(request.strip() + b"\n")
        stream.flush()
        line = stream.readline()
    if not line:
        print("actant host closed the connection without a response", file=sys.stderr)
        return 1
    sys.stdout.write(line.decode())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m actant.sandbox.host")
    commands = parser.add_subparsers(dest="command", required=True)
    serve_cmd = commands.add_parser(Command.SERVE)
    serve_cmd.add_argument("--socket", required=True)
    serve_cmd.add_argument("--factory", required=True, help="pkg.mod:fn(key, init) -> ctx")
    serve_cmd.add_argument("--after", help="shell command run (debounced) after requests")
    call_cmd = commands.add_parser(Command.CALL)
    call_cmd.add_argument("--socket", required=True)
    call_cmd.add_argument("--request-file", required=True, help="JSON request; deleted once read")
    call_cmd.add_argument("--wait", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.command == Command.SERVE:
        asyncio.run(serve(args.socket, args.factory, args.after))
        return 0
    return call(args.socket, args.request_file, args.wait)


if __name__ == "__main__":
    sys.exit(main())
