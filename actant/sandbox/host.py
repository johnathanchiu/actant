"""The service host: serves named service classes over HTTP from inside a sandbox.

A service is a plain class whose public methods are callable remotely (see
:mod:`actant.sandbox.service`). Running its methods next to the sandbox's files
keeps file-heavy work and in-memory state out of the worker; only text and
image bytes cross the boundary.

::

    python -m actant.sandbox.entry \
        '{"host": {"services": {"tools": "pkg.mod:Tools", "stages": "pkg.mod:Stages"}}}'

(:mod:`actant.sandbox.entry` is the command line; this module is imported, never
run as ``__main__``, so :func:`script_env` sees the configuration ``main`` set.)

Services are fixed at launch. Requests name a service and a method, never a module.
Several services let a product keep the model's tools on one class and the calls
its own code makes (setup, stages, checks) on another, without filtering.

Protocol (JSON bodies, typed in :mod:`actant.sandbox.protocol`)::

    POST /v1/call CallRequest -> CallResponse
    POST /v1/shutdown         -> close instances, push, exit

Connections are kept alive (HTTP/1.1); :func:`post` is the pooled client.

One instance per ``(service, key)``, created on its first call with ``init``
(``await Class.open(**init)`` when the class defines ``open``, else
``Class(**init)``); later calls with that key reuse it and ignore ``init``.
Services do not share instances: two classes that work on the same state build it
from the same ``init`` (the sandbox's files, or an object their ``open`` looks up).
Arguments are validated against the method's signature before the call, so a
parameter annotated with a pydantic model or a ``Literal`` receives that type.
Calls run concurrently. When ``ACTANT_HOST_TOKEN`` is set at launch, every call
needs ``Authorization: Bearer <token>`` (checked before the body is read); the
variable is removed from the process environment once read.

Storage pushes (``HostConfig.push``) never fail or stall a call. A push runs after
calls and every ``interval_s`` while calls have completed since the last successful
one; never two at once. Each is killed (its process group) after ``timeout_s``. A
host that pushes adds a :class:`~actant.sandbox.protocol.StorageStatus` to every
call response (``CallResponse.storage``).

Image uploads (``HostConfig.images``) never fail a call either. Each returned image
is uploaded under a key named by its content hash and presigned, so the response
carries a URL instead of the bytes; an image whose upload or presign fails goes
inline, and ``StorageStatus.image_error`` says why.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import hashlib
import hmac
import importlib
import inspect
import json
import mimetypes
import os
import select
import signal
import sys
import threading
import time
import traceback
from collections.abc import Mapping
from http import HTTPStatus
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from actant.sandbox.base import Endpoint, ImageBucket, SandboxSpec
from actant.sandbox.protocol import (
    CallRequest,
    CallResponse,
    Header,
    HostConfig,
    Image,
    ImageUploadConfig,
    InlineSource,
    PushConfig,
    Route,
    StorageStatus,
    UrlSource,
)

TOKEN_ENV = "ACTANT_HOST_TOKEN"
#: The line the host prints to stdout once it accepts connections: ``<prefix> <port>``.
READY_PREFIX = "actant-host-listening"
#: A pooled client connection idle longer than this is not reused: a proxy may have
#: dropped it, and a request lost on a half-closed connection cannot be told apart
#: from one that ran.
IDLE_REUSE_S = 20.0
MAX_IDLE_PER_HOST = 32
#: The instance lifecycle; never callable.
LIFECYCLE = frozenset({"open", "close"})
MAX_BODY = 256 * 1024 * 1024
#: Images a host uploads at once for one response.
UPLOAD_CONCURRENCY = 8
_IMAGE_MAGIC = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
}

_scrub: frozenset[str] = frozenset()


def script_env() -> dict[str, str]:
    """This process's environment minus the host's ``HostConfig.scrub`` names.

    The host keeps every variable because methods call external APIs with them. Pass this
    as ``env=`` to any subprocess a tool starts for code it did not write. Best
    effort, not a security boundary.
    """
    return {name: value for name, value in os.environ.items() if name not in _scrub}


def service_methods(cls: type) -> list[str]:
    """The service's callable methods: public functions (``async def`` or plain ``def``) on the class
    and its bases, in definition order, excluding :data:`LIFECYCLE`. Static and class
    methods and properties are not callable."""
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
        and inspect.isfunction(inspect.getattr_static(cls, name))
    ]


@functools.cache
def parameters_model(cls: type, method: str) -> type[BaseModel]:
    """The pydantic model of ``cls.method``'s parameters, without ``self``.

    Annotations are resolved one parameter at a time, so a return annotation that
    only exists for type checkers does not matter.
    """
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


def image_type(data: bytes) -> str | None:
    """The media type of PNG, JPEG, GIF or WebP bytes; ``None`` for anything else."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return next((kind for magic, kind in _IMAGE_MAGIC.items() if data.startswith(magic)), None)


def load(path: str) -> type:
    """``"pkg.mod:Class"`` to the class."""
    module_name, _, qualname = path.partition(":")
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    if not isinstance(target, type):
        raise TypeError(f"{path} is not a class")
    return target


def encode_output(output: object) -> CallResponse:
    """A service method's return value as a response.

    ``str`` is the text. An object with ``.text`` and ``.images`` carries images as
    file paths or bytes; a path must name an image file (by extension) and bytes
    must be one (by magic number), because a path can be model-steered and
    anything else would leave the sandbox or be rejected by the LLM. Other values
    are JSON-encoded.
    """
    if isinstance(output, str):
        return CallResponse(text=output)
    if hasattr(output, "text") and hasattr(output, "images"):
        text = str(output.text)  # pyright: ignore[reportAttributeAccessIssue]
        images: list[Image] = []
        for index, image in enumerate(output.images):  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(image, bytes | bytearray):
                name, data = f"image-{index}", bytes(image)
                media_type = image_type(data) or ""
                if not media_type:
                    text += f"\n(dropped {name}: not PNG, JPEG, GIF or WebP bytes)"
                    continue
            else:
                name = str(image)
                media_type = mimetypes.guess_type(name)[0] or ""
                if not media_type.startswith("image/"):
                    text += f"\n(dropped {name}: not an image file)"
                    continue
                data = Path(name).read_bytes()
            source = InlineSource(data_b64=base64.b64encode(data).decode())
            images.append(Image(name=name, media_type=media_type, source=source))
        return CallResponse(text=text, images=images)
    try:
        return CallResponse(text=json.dumps(output, default=str))
    except (TypeError, ValueError):
        return CallResponse(text=str(output))


def failure(error: BaseException) -> CallResponse:
    """An exception as a response the model can correct from; the tail keeps it short."""
    tail = "".join(traceback.format_exception(error)[-3:])
    return CallResponse(error=f"{type(error).__name__}: {error}\n{tail}")


async def call_method(instance: object, method: str, args: Mapping[str, object]) -> CallResponse:
    """Validate the arguments, run one service method, and encode its result or its exception.

    A plain ``def`` runs in a worker thread, so it never stalls other calls. An
    ``async def`` runs on the event loop: CPU-heavy or blocking work inside one
    stalls every concurrent call until it awaits, so such a method should be a
    plain ``def`` or hand the work to ``asyncio.to_thread`` itself.
    """
    try:
        params = parameters_model(type(instance), method).model_validate(args)
        kwargs = {name: getattr(params, name) for name in type(params).model_fields}
        function = getattr(instance, method)
        if inspect.iscoroutinefunction(function):
            output = await function(**kwargs)
        else:
            output = await asyncio.to_thread(function, **kwargs)
        return encode_output(output)
    except Exception as error:  # noqa: BLE001 -- a method failure is a result, not a crash
        return failure(error)


async def _maybe_await(value: object) -> object:
    return await value if inspect.isawaitable(value) else value


async def open_instance(cls: type, init: Mapping[str, object]) -> object:
    opener = getattr(cls, "open", None)
    return await _maybe_await(opener(**init)) if opener is not None else cls(**init)


async def run_bounded(
    argv: list[str], timeout: float, *, stdin: bytes | None = None, capture: bool = False
) -> tuple[str | None, bytes]:
    """Run ``argv`` (its process group killed after ``timeout``); its short failure reason
    (``None`` when it exited 0) and, with ``capture``, its stdout."""
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL if stdin is None else asyncio.subprocess.PIPE,
            # A push lists every file it uploads: not worth holding.
            stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,  # a timeout kills s5cmd and anything it started
        )
    except OSError as error:
        return f"could not start: {error}"[:500], b""
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(stdin), timeout)
    except (TimeoutError, asyncio.CancelledError) as stopped:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()
        if isinstance(stopped, asyncio.CancelledError):
            raise
        return f"timed out after {timeout:g}s", b""
    if process.returncode:
        tail = stderr.decode(errors="replace").strip()[-500:]
        return f"exited {process.returncode}: {tail}", stdout or b""
    return None, stdout or b""


def image_upload_config(
    bucket: ImageBucket, spec: SandboxSpec, thread_id: str
) -> ImageUploadConfig | None:
    """The host's image uploads for ``thread_id``; ``None`` when the spec sends bytes."""
    if spec.image_url_ttl_s is None:
        return None
    return ImageUploadConfig(
        destination=bucket.destination(thread_id),
        endpoint_url=bucket.endpoint_url,
        public_endpoint_url=bucket.public_endpoint_url,
        expires_s=spec.image_url_ttl_s,
        timeout_s=spec.image_upload_timeout_s,
    )


def _s5cmd(endpoint_url: str | None, *args: str) -> list[str]:
    return ["s5cmd", *(["--endpoint-url", endpoint_url] if endpoint_url else []), *args]


async def upload_image(image: Image, config: ImageUploadConfig) -> tuple[Image, str | None]:
    """``image`` with a presigned URL source, or unchanged with the reason it could not be."""
    if not isinstance(image.source, InlineSource):
        return image, None
    data = base64.b64decode(image.source.data_b64)
    extension = mimetypes.guess_extension(image.media_type) or ""
    key = f"{config.destination}{hashlib.sha256(data).hexdigest()}{extension}"
    # One budget for both commands: an unreachable bucket costs at most ``timeout_s``.
    deadline = time.monotonic() + config.timeout_s
    upload = _s5cmd(config.endpoint_url, "pipe", "--content-type", image.media_type, key)
    error, _ = await run_bounded(upload, config.timeout_s, stdin=data)
    if error is not None:
        return image, f"{image.name}: upload {error}"
    expires_at = time.time() + config.expires_s
    presign = _s5cmd(
        config.public_endpoint_url, "presign", "--expire", f"{config.expires_s}s", key
    )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return (
            image,
            f"{image.name}: presign skipped: upload used the {config.timeout_s:g}s budget",
        )
    error, stdout = await run_bounded(presign, remaining, capture=True)
    url = stdout.decode(errors="replace").strip()
    if error is None and not url.startswith(("https://", "http://")):
        error = f"not a URL: {url[:200]!r}"
    if error is not None:
        return image, f"{image.name}: presign {error}"
    return image.model_copy(update={"source": UrlSource(url=url, expires_at=expires_at)}), None


async def upload_images(
    response: CallResponse, config: ImageUploadConfig
) -> tuple[CallResponse, str | None]:
    """``response`` with each image uploaded and presigned where possible, and the reason
    the first that could not be stayed inline. Never raises.

    At most :data:`UPLOAD_CONCURRENCY` upload at once. After one fails, the images not yet
    started stay inline without trying: a bucket that refused one would cost every other
    image its own timeout."""
    if not response.images:
        return response, None
    gate = asyncio.Semaphore(UPLOAD_CONCURRENCY)
    failed: list[str] = []

    async def one(image: Image) -> Image:
        async with gate:
            if failed:
                return image
            try:
                uploaded, error = await upload_image(image, config)
            except Exception as exc:  # noqa: BLE001 -- an upload failure never fails the call
                uploaded, error = image, f"{image.name}: {type(exc).__name__}: {exc}"[:500]
            if error is not None:
                failed.append(error)
            return uploaded

    images = await asyncio.gather(*(one(image) for image in response.images))
    return response.model_copy(update={"images": images}), failed[0] if failed else None


_idle: dict[tuple[str, str], list[tuple[HTTPConnection, float]]] = {}
_idle_lock = threading.Lock()


def _prune() -> None:
    """Close stale idle connections to every host, so closed sandboxes leave no sockets."""
    now = time.monotonic()
    for slot, pool in list(_idle.items()):
        for connection, since in pool:
            if now - since >= IDLE_REUSE_S:
                connection.close()
        pool[:] = [(c, since) for c, since in pool if now - since < IDLE_REUSE_S]
        if not pool:
            del _idle[slot]


def post(endpoint: Endpoint, path: str, body: bytes, timeout: float) -> tuple[int, bytes]:
    """``POST`` to a host over a pooled keep-alive connection. Blocking: run it in a thread.

    Never sends a call twice. An idle connection that is readable (the server closed
    it, or sent something unasked) is discarded before use. If writing the request
    fails on a reused connection, the server never received a complete request, so
    it is sent again on another connection. Any failure once the request is written
    raises ``OSError``: the call may have run, and repeating it could run it twice.
    """
    url = urlsplit(endpoint.url)
    slot = (url.scheme, url.netloc)
    headers = {Header.CONTENT_TYPE: "application/json", **endpoint.headers}
    target = url.path.rstrip("/") + path
    while True:
        connection = None
        with _idle_lock:
            _prune()
            pool = _idle.get(slot, [])
            while pool and connection is None:
                candidate, _ = pool.pop()
                sock = candidate.sock
                if sock is not None and not select.select([sock], [], [], 0)[0]:
                    connection = candidate
                else:
                    candidate.close()
        reused = connection is not None
        if connection is None:
            factory = HTTPSConnection if url.scheme == "https" else HTTPConnection
            connection = factory(url.netloc, timeout=timeout)
        else:
            assert connection.sock is not None
            connection.sock.settimeout(timeout)
        try:
            connection.request("POST", target, body, headers)
        except OSError:
            connection.close()
            if reused:  # the body is incomplete on the server's side: nothing ran
                continue
            raise
        except BaseException:
            connection.close()
            raise
        try:
            reply = connection.getresponse()
            data = reply.read()
        except BaseException:
            connection.close()
            raise
        if reply.will_close:
            connection.close()
        else:
            with _idle_lock:
                pool = _idle.setdefault(slot, [])
                if len(pool) < MAX_IDLE_PER_HOST:
                    pool.append((connection, time.monotonic()))
                    connection = None
            if connection is not None:
                connection.close()
        return reply.status, data


class Host:
    def __init__(
        self,
        services: Mapping[str, type],
        *,
        token: str | None = None,
        push: PushConfig | None = None,
        images: ImageUploadConfig | None = None,
    ) -> None:
        self.services = dict(services)
        self.methods = {name: frozenset(service_methods(cls)) for name, cls in services.items()}
        self.token = token
        self.push = push
        self.images = images
        # Futures, not instances: two parallel first calls share one ``open``.
        self.instances: dict[tuple[str, str], asyncio.Future[object]] = {}
        self._push_due = asyncio.Event()
        self._push_lock = asyncio.Lock()
        self._pending = False
        self._last_attempt: float | None = None
        self._last_success: float | None = None
        self._last_error: str | None = None
        self._failures = 0
        self._shutdown: asyncio.Future[None] | None = None
        self.stop = asyncio.Event()
        self.writers: set[asyncio.StreamWriter] = set()

    async def instance(self, service: str, key: str, init: Mapping[str, object]) -> object:
        slot = (service, key)
        future = self.instances.get(slot)
        if future is None:
            opening = open_instance(self.services[service], init)
            future = self.instances[slot] = asyncio.ensure_future(opening)
        try:
            return await future
        except BaseException:
            # A failed ``open``: the next call retries it, unless a retry already replaced
            # this one. A cancelled waiter leaves an ``open`` still running in place.
            if future.done() and self.instances.get(slot) is future:
                del self.instances[slot]
            raise

    async def call(self, request: CallRequest) -> tuple[HTTPStatus, CallResponse]:
        if request.service not in self.services:
            return HTTPStatus.NOT_FOUND, CallResponse(
                error=f"unknown service {request.service!r}; served: {sorted(self.services)}"
            )
        if request.method not in self.methods[request.service]:
            return HTTPStatus.NOT_FOUND, CallResponse(
                error=f"unknown service method {request.method!r}"
            )
        try:
            instance = await self.instance(request.service, request.key, request.init)
            payload = await call_method(instance, request.method, request.args)
        except Exception as error:  # noqa: BLE001 -- ``open`` failed; report, keep serving
            payload = failure(error)
        image_error = None
        if self.images:
            payload, image_error = await upload_images(payload, self.images)
        if self.push:
            self._pending = True
            self._push_due.set()
        if self.push or self.images:
            status = self.storage_status(image_error=image_error)
            payload = payload.model_copy(update={"storage": status})
        return HTTPStatus.OK, payload

    def storage_status(self, *, image_error: str | None = None) -> StorageStatus:
        """The push status added to call responses, with this response's image error."""
        return StorageStatus(
            last_attempt_at=self._last_attempt,
            last_success_at=self._last_success,
            last_error=self._last_error,
            consecutive_failures=self._failures,
            pending=self._pending,
            image_error=image_error,
        )

    async def pusher(self) -> None:
        """Push after calls (one running, at most one pending) and every
        ``push.interval_s`` while a completed call is not yet pushed."""
        assert self.push
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._push_due.wait(), self.push.interval_s)
            if self._push_due.is_set() or self._pending:
                self._push_due.clear()
                await self.push_now()

    async def push_now(self) -> None:
        """Run the push command once, bounded by ``push.timeout_s``. Never raises."""
        if not self.push:
            return
        async with self._push_lock:
            self._pending = False
            self._last_attempt = started = time.time()
            error = await self._run_push(self.push)
            if error is None:
                self._last_success, self._last_error, self._failures = started, None, 0
            else:
                # A call during the push set ``_pending`` again; a failure keeps it set.
                self._pending = True
                self._last_error, self._failures = error, self._failures + 1
                print(f"storage push failed: {error}", file=sys.stderr, flush=True)

    @staticmethod
    async def _run_push(push: PushConfig) -> str | None:
        """The push's short failure reason, or ``None`` when it exited 0."""
        return (await run_bounded(push.argv, push.timeout_s))[0]

    def reject(
        self, verb: str, path: str, headers: Mapping[str, str]
    ) -> tuple[HTTPStatus, CallResponse] | None:
        """Why a request is refused before its body is read, or ``None``.
        ``headers`` are keyed by lower-cased name."""
        if path not in set(Route):
            return HTTPStatus.NOT_FOUND, CallResponse(error=f"no route {verb} {path}")
        if verb != "POST":
            return HTTPStatus.METHOD_NOT_ALLOWED, CallResponse(error="use POST")
        if self.token is not None and not hmac.compare_digest(
            headers.get(Header.AUTHORIZATION.lower(), ""), f"Bearer {self.token}"
        ):
            return HTTPStatus.UNAUTHORIZED, CallResponse(error="bad or missing bearer token")
        if int(headers.get(Header.CONTENT_LENGTH.lower()) or 0) > MAX_BODY:
            return HTTPStatus.REQUEST_ENTITY_TOO_LARGE, CallResponse(error="body too large")
        return None

    async def route(self, body: bytes) -> tuple[HTTPStatus, CallResponse]:
        try:
            request = CallRequest.model_validate_json(body)
        except ValidationError as error:
            return HTTPStatus.BAD_REQUEST, CallResponse(error=f"bad call request: {error}")
        return await self.call(request)

    async def connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """HTTP/1.1 with keep-alive: requests on one connection are answered in turn."""
        self.writers.add(writer)
        try:
            close = False
            while not close:
                try:
                    head = await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError as error:
                    if error.partial:  # anything but a clean close between requests
                        await respond(
                            writer, HTTPStatus.BAD_REQUEST, CallResponse(error="truncated")
                        )
                    return
                except asyncio.LimitOverrunError:
                    await respond(
                        writer, HTTPStatus.BAD_REQUEST, CallResponse(error="head too large")
                    )
                    return
                status, payload, close = await self.handle(head, reader)
                await respond(writer, status, payload, close=close)
                if self._shutdown is not None and self._shutdown.done() and close:
                    self.stop.set()  # after the reply, so the caller sees shutdown finish
        except ConnectionError:
            pass
        finally:
            self.writers.discard(writer)
            writer.close()

    async def handle(
        self, head: bytes, reader: asyncio.StreamReader
    ) -> tuple[HTTPStatus, CallResponse, bool]:
        """One request: its status, payload, and whether the connection must close."""
        try:
            lines = head.decode("latin-1").split("\r\n")
            verb, target, version = lines[0].split(" ", 2)
            headers: dict[str, str] = {}
            for line in lines[1:]:
                name, sep, value = line.partition(":")
                if sep:
                    headers[name.strip().lower()] = value.strip()
            connection = headers.get(Header.CONNECTION.lower(), "")
            close = version != "HTTP/1.1" or connection.lower() == "close"
            path = target.split("?")[0]
            refused = self.reject(verb, path, headers)
            if refused is not None:
                return *refused, True  # the body is unread: the connection is spent
            body = await reader.readexactly(int(headers.get(Header.CONTENT_LENGTH.lower()) or 0))
        except (ValueError, asyncio.IncompleteReadError) as error:
            return HTTPStatus.BAD_REQUEST, CallResponse(error=f"malformed request: {error}"), True
        if path == Route.SHUTDOWN:
            await self.shutdown()
            return HTTPStatus.OK, CallResponse(text="stopped"), True
        return *(await self.route(body)), close

    async def close_instances(self) -> None:
        for future in self.instances.values():
            if future.done() and not future.cancelled() and future.exception() is None:
                closer = getattr(future.result(), "close", None)
                if closer is not None:
                    with contextlib.suppress(Exception):
                        await _maybe_await(closer())

    def shutdown(self) -> asyncio.Future[None]:
        """Close instances and push storage once more. Runs once; :func:`serve` then exits.

        The push waits for one already running, so it ends within twice ``push.timeout_s``."""
        if self._shutdown is None:
            self._shutdown = asyncio.ensure_future(self._close_and_push())
        return self._shutdown

    async def _close_and_push(self) -> None:
        await self.close_instances()
        await self.push_now()


async def respond(
    writer: asyncio.StreamWriter,
    status: HTTPStatus,
    payload: CallResponse,
    *,
    close: bool = True,
) -> None:
    data = payload.to_json()
    head = [
        f"HTTP/1.1 {status.value} {status.phrase}",
        f"{Header.CONTENT_TYPE}: application/json",
        f"{Header.CONTENT_LENGTH}: {len(data)}",
        *([f"{Header.CONNECTION}: close"] if close else []),
    ]
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + data)
    await writer.drain()


async def serve(host: Host, bind: str, port: int) -> None:
    """Serve until SIGTERM, SIGINT, SIGHUP or ``POST /v1/shutdown``; on any exit, close
    the instances and push storage once more. Modal's ``terminate`` and timeouts send
    SIGKILL, which nothing can handle: close the sandbox through its handle instead."""
    server = await asyncio.start_server(host.connection, bind, port)
    bound = server.sockets[0].getsockname()[1]
    print(f"{READY_PREFIX} {bound}", flush=True)
    pusher = asyncio.create_task(host.pusher()) if host.push else None
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        loop.add_signal_handler(signum, host.stop.set)
    try:
        await host.stop.wait()
    finally:
        server.close()
        await host.shutdown()
        if pusher is not None:
            pusher.cancel()
        for writer in list(host.writers):
            writer.close()


def load_services(config: HostConfig) -> dict[str, type]:
    """``config``'s service classes, imported."""
    return {name: load(path) for name, path in config.services.items()}


def main(config: HostConfig, services: Mapping[str, type] | None = None) -> int:
    """Serve ``config`` (its ``services`` when already loaded) until stopped
    (:mod:`actant.sandbox.entry` is the command line)."""
    global _scrub
    _scrub = frozenset(config.scrub)
    services = load_services(config) if services is None else services
    token = os.environ.pop(TOKEN_ENV, None)
    host = Host(services, token=token, push=config.push, images=config.images)
    asyncio.run(serve(host, config.bind, config.port))
    return 0
