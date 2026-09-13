"""The toolset host: serves one toolset class over HTTP from inside a sandbox.

A toolset is a plain class whose public ``async def`` methods are tools (see
:mod:`actant.tools.toolset`). Running its methods next to the sandbox's files
keeps file-heavy work and in-memory state out of the worker; only text and
image bytes cross the boundary.

::

    python -m actant.sandbox.entry -- --toolset pkg.mod:Class --port 8080

(:mod:`actant.sandbox.entry` is the command line; this module is imported, never
run as ``__main__``, so :func:`script_env` sees the configuration ``main`` set.)

The toolset is fixed at launch. Requests name a method and never a module.

Protocol (JSON bodies)::

    GET  /v1/health                                  -> {"ok": true}
    POST /v1/call {key, init, method, args}          -> {text, images, error}
         images: [{name, media_type, data_b64}]

One instance per ``key``, created on its first call with ``init``
(``await Class.open(**init)`` when the class defines ``open``, else
``Class(**init)``); later calls with that key reuse it and ignore ``init``.
Calls run concurrently. When ``ACTANT_HOST_TOKEN`` is set at launch, every call
needs ``Authorization: Bearer <token>``; the variable is removed from the
process environment once read.

Stdlib only: this module must import in any image that has actant installed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import hmac
import importlib
import inspect
import json
import mimetypes
import os
import signal
import sys
import traceback
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from pathlib import Path

TOKEN_ENV = "ACTANT_HOST_TOKEN"
#: The line the host prints to stdout once it accepts connections: ``<prefix> <port>``.
READY_PREFIX = "actant-host-listening"
CALL_PATH = "/v1/call"
HEALTH_PATH = "/v1/health"
#: The instance lifecycle; never exposed as tools.
LIFECYCLE = frozenset({"open", "close"})
MAX_BODY = 256 * 1024 * 1024
_IMAGE_MAGIC = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
    b"RIFF": "image/webp",
}

_scrub: frozenset[str] = frozenset()


def script_env() -> dict[str, str]:
    """This process's environment minus the names the host was launched with ``--scrub``.

    The host keeps every variable because tools call services with them. Pass this
    as ``env=`` to any subprocess a tool starts for code it did not write. Best
    effort, not a security boundary.
    """
    return {name: value for name, value in os.environ.items() if name not in _scrub}


def public_methods(cls: type) -> list[str]:
    """The toolset's tools: public coroutine functions on the class and its bases,
    in definition order, excluding :data:`LIFECYCLE`."""
    names: list[str] = []
    for klass in reversed(cls.__mro__):
        for name in vars(klass):
            if name not in names:
                names.append(name)
    return [
        name
        for name in names
        if not name.startswith("_")
        and name not in LIFECYCLE
        and inspect.iscoroutinefunction(inspect.getattr_static(cls, name))
    ]


def load(path: str) -> type:
    """``"pkg.mod:Class"`` to the class."""
    module_name, _, qualname = path.partition(":")
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    if not isinstance(target, type):
        raise TypeError(f"{path} is not a class")
    return target


def response(
    text: str = "", images: Sequence[Mapping[str, str]] = (), error: str | None = None
) -> dict[str, object]:
    return {"text": text, "images": list(images), "error": error}


def encode_output(output: object) -> dict[str, object]:
    """A tool's return value as a response.

    ``str`` is the text. An object with ``.text`` and ``.images`` carries images as
    file paths or bytes; a path must name an image file (by extension), because a
    path can be model-steered and anything else would leave the sandbox. Other
    values are JSON-encoded.
    """
    if isinstance(output, str):
        return response(output)
    if hasattr(output, "text") and hasattr(output, "images"):
        text = str(output.text)  # pyright: ignore[reportAttributeAccessIssue]
        images: list[dict[str, str]] = []
        for index, image in enumerate(output.images):  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(image, bytes | bytearray):
                name, data = f"image-{index}", bytes(image)
                media_type = next(
                    (kind for magic, kind in _IMAGE_MAGIC.items() if data.startswith(magic)),
                    "image/png",
                )
            else:
                name = str(image)
                media_type = mimetypes.guess_type(name)[0] or ""
                if not media_type.startswith("image/"):
                    text += f"\n(dropped {name}: not an image file)"
                    continue
                data = Path(name).read_bytes()
            images.append(
                {
                    "name": name,
                    "media_type": media_type,
                    "data_b64": base64.b64encode(data).decode(),
                }
            )
        return response(text, images)
    try:
        return response(json.dumps(output, default=str))
    except (TypeError, ValueError):
        return response(str(output))


def failure(error: BaseException) -> dict[str, object]:
    """An exception as a response the model can correct from; the tail keeps it short."""
    tail = "".join(traceback.format_exception(error)[-3:])
    return response(error=f"{type(error).__name__}: {error}\n{tail}")


async def call_method(
    instance: object, method: str, args: Mapping[str, object]
) -> dict[str, object]:
    """Run one tool method and encode its result or its exception."""
    try:
        return encode_output(await getattr(instance, method)(**args))
    except Exception as error:  # noqa: BLE001 -- a tool failure is a result, not a crash
        return failure(error)


async def open_instance(cls: type, init: Mapping[str, object]) -> object:
    opener = getattr(cls, "open", None)
    return await opener(**init) if opener is not None else cls(**init)


class Host:
    def __init__(
        self, cls: type, *, token: str | None = None, push: Sequence[str] | None = None
    ) -> None:
        self.cls = cls
        self.methods = frozenset(public_methods(cls))
        self.token = token
        self.push = list(push) if push else None
        # Futures, not instances: two parallel first calls share one ``open``.
        self.instances: dict[str, asyncio.Future[object]] = {}
        self._push_due = asyncio.Event()

    async def instance(self, key: str, init: Mapping[str, object]) -> object:
        future = self.instances.get(key)
        if future is None:
            future = self.instances[key] = asyncio.ensure_future(open_instance(self.cls, init))
        try:
            return await future
        except BaseException:
            # The next call retries ``open``, unless a retry already replaced this one.
            if self.instances.get(key) is future:
                del self.instances[key]
            raise

    async def call(self, body: Mapping[str, object]) -> tuple[HTTPStatus, dict[str, object]]:
        method, key = body.get("method"), body.get("key")
        init, args = body.get("init") or {}, body.get("args") or {}
        if not isinstance(key, str) or not isinstance(init, dict) or not isinstance(args, dict):
            return HTTPStatus.BAD_REQUEST, response(
                error="`key` must be a string; `init` and `args` objects"
            )
        if method not in self.methods:
            return HTTPStatus.NOT_FOUND, response(error=f"unknown tool method {method!r}")
        try:
            instance = await self.instance(key, init)
            return HTTPStatus.OK, await call_method(instance, str(method), args)
        except Exception as error:  # noqa: BLE001 -- ``open`` failed; report, keep serving
            return HTTPStatus.OK, failure(error)
        finally:
            if self.push:
                self._push_due.set()

    async def pusher(self) -> None:
        """Push storage after calls: one push running, at most one pending."""
        assert self.push
        while True:
            await self._push_due.wait()
            self._push_due.clear()
            process = await asyncio.create_subprocess_exec(
                *self.push, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            if process.returncode:
                tail = stderr.decode(errors="replace")[-2000:]
                print(
                    f"storage push exited {process.returncode}: {tail}",
                    file=sys.stderr,
                    flush=True,
                )

    async def route(
        self, verb: str, path: str, headers: Mapping[str, str], body: bytes
    ) -> tuple[HTTPStatus, dict[str, object]]:
        if verb == "GET" and path == HEALTH_PATH:
            return HTTPStatus.OK, {"ok": True}
        if path != CALL_PATH:
            return HTTPStatus.NOT_FOUND, response(error=f"no route {verb} {path}")
        if verb != "POST":
            return HTTPStatus.METHOD_NOT_ALLOWED, response(error="use POST")
        if self.token is not None and not hmac.compare_digest(
            headers.get("authorization", ""), f"Bearer {self.token}"
        ):
            return HTTPStatus.UNAUTHORIZED, response(error="bad or missing bearer token")
        try:
            request = json.loads(body)
        except ValueError as error:
            return HTTPStatus.BAD_REQUEST, response(error=f"body is not JSON: {error}")
        if not isinstance(request, dict):
            return HTTPStatus.BAD_REQUEST, response(error="body must be a JSON object")
        return await self.call(request)

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """One HTTP/1.1 request per connection, answered with ``Connection: close``."""
        try:
            try:
                head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1").split("\r\n")
                verb, target, _ = head[0].split(" ", 2)
                headers = {}
                for line in head[1:]:
                    name, sep, value = line.partition(":")
                    if sep:
                        headers[name.strip().lower()] = value.strip()
                length = int(headers.get("content-length") or 0)
                if length > MAX_BODY:
                    status, payload = (
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        response(error="body too large"),
                    )
                else:
                    body = await reader.readexactly(length)
                    status, payload = await self.route(verb, target.split("?")[0], headers, body)
            except (ValueError, asyncio.IncompleteReadError, asyncio.LimitOverrunError) as error:
                status, payload = (
                    HTTPStatus.BAD_REQUEST,
                    response(error=f"malformed request: {error}"),
                )
            data = json.dumps(payload).encode()
            writer.write(
                f"HTTP/1.1 {status.value} {status.phrase}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode()
                + data
            )
            await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()

    async def close_instances(self) -> None:
        for future in self.instances.values():
            if future.done() and not future.cancelled() and future.exception() is None:
                closer = getattr(future.result(), "close", None)
                if closer is not None:
                    with contextlib.suppress(Exception):
                        await closer()


async def serve(host: Host, bind: str, port: int) -> None:
    server = await asyncio.start_server(host.connection, bind, port)
    bound = server.sockets[0].getsockname()[1]
    print(f"{READY_PREFIX} {bound}", flush=True)
    pusher = asyncio.create_task(host.pusher()) if host.push else None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    async with server:
        await stop.wait()
    if pusher is not None:
        pusher.cancel()
    await host.close_instances()


def main(argv: Sequence[str] | None = None) -> int:
    global _scrub
    parser = argparse.ArgumentParser(prog="python -m actant.sandbox.entry --")
    parser.add_argument("--toolset", required=True, help="pkg.mod:Class")
    parser.add_argument("--port", type=int, default=8080, help="0 picks a free port")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--scrub", action="append", default=[], help="name script_env() removes")
    parser.add_argument("--push", help="JSON argv run (debounced) after calls to push storage")
    args = parser.parse_args(argv)
    _scrub = frozenset(args.scrub)
    host = Host(
        load(args.toolset),
        token=os.environ.pop(TOKEN_ENV, None),
        push=json.loads(args.push) if args.push else None,
    )
    asyncio.run(serve(host, args.bind, args.port))
    return 0
