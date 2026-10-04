"""Services served from worker processes beside the host (:class:`ServiceConfig.processes`).

The host still takes every request: its checks, call ids, cancels, image uploads and pushes.
A placed service's call is run by the worker its key was dealt to on the key's first call; the
worker (:mod:`actant.sandbox.worker`) serves its keys' instances as the host would, so it
validates the arguments and encodes the result, and the host sends that response on.

A worker that exits fails its calls in flight (never sent again: each may have run in part).
Its slot starts one replacement; once that one is gone too, the slot's keys are served in the
host. A worker exits, closing its instances, when its socket closes (the host shut down or
died).

What only the host may hold or write (an index, a file other services write too) a worker's
code reaches with :func:`on_host`: ``await on_host(fn, *args)`` runs a :func:`host_function`
on the host's event loop, and runs it right there when called in the host. A host function's
arguments and result are pickled.

Messages are length-prefixed pickles over a socketpair: both ends are this package, and
nothing else can reach the socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import itertools
import os
import pickle
import socket
import struct
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping
from enum import StrEnum
from typing import ParamSpec, TypeVar, cast, overload

from actant.sandbox.protocol import CallRequest, CallResponse, ServiceConfig, WorkerConfig

#: A slot's starts: its first worker and one replacement.
STARTS = 2
#: How long a closing worker may take to close its instances before it is killed.
CLOSE_GRACE_S = 10.0

_SIZE = struct.Struct("!Q")

P = ParamSpec("P")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., object])


class Kind(StrEnum):
    CALL = "call"
    CANCEL = "cancel"
    DONE = "done"
    #: A worker runs a host function: ``(name, args, kwargs)``.
    ASK = "ask"
    #: Its outcome: ``(True, result)`` or ``(False, exception)``.
    ANSWER = "answer"


def frame(kind: str, number: int, payload: object) -> bytes:
    data = pickle.dumps((kind, number, payload), pickle.HIGHEST_PROTOCOL)
    return _SIZE.pack(len(data)) + data


def answer(number: int, ok: bool, value: object) -> bytes:
    """An outcome's frame; one that does not survive pickling goes as its error's text."""
    try:
        data = frame(Kind.ANSWER, number, (ok, value))
        if not ok:
            pickle.loads(data[_SIZE.size :])  # an exception whose class cannot be rebuilt
        return data
    except Exception as error:  # noqa: BLE001 -- the answer is what could not be sent
        reason = value if not ok else error
        return frame(
            Kind.ANSWER, number, (False, RuntimeError(f"{type(reason).__name__}: {reason}"))
        )


_functions: dict[str, Callable[..., object]] = {}


def _name(function: Callable[..., object]) -> str:
    return f"{function.__module__}:{function.__qualname__}"


def host_function(function: F) -> F:
    """Let workers run ``function`` on the host (:func:`on_host`). A module-level function;
    plain or ``async``. A plain one runs on the host's event loop, so it must be short."""
    _functions[_name(function)] = function
    return function


async def _run(
    function: Callable[..., object], args: tuple[object, ...], kwargs: dict[str, object]
) -> object:
    value = function(*args, **kwargs)
    return await value if inspect.isawaitable(value) else value


@overload
async def on_host(function: Callable[P, Awaitable[R]], *args: P.args, **kwargs: P.kwargs) -> R: ...
@overload
async def on_host(function: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R: ...
async def on_host(function: Callable[P, object], *args: P.args, **kwargs: P.kwargs) -> object:
    """``function(*args, **kwargs)`` on the host: asked of it from a worker, run here in it."""
    name = _name(function)
    if _functions.get(name) is not function:
        raise TypeError(f"{name} is not a @host_function")
    if _link is None:
        return await _run(function, args, kwargs)
    return await _link.ask(name, args, kwargs)


class HostLink:
    """The host as a worker reaches it: host functions in flight, by number."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer = writer
        self._numbers = itertools.count()
        self._asked: dict[int, asyncio.Future[tuple[bool, object]]] = {}

    async def ask(self, name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        number = next(self._numbers)
        future = self._asked[number] = asyncio.get_running_loop().create_future()
        try:
            await send(self.writer, frame(Kind.ASK, number, (name, args, kwargs)))
            ok, value = await future
        finally:
            self._asked.pop(number, None)
        if not ok:
            assert isinstance(value, BaseException)
            raise value
        return value

    def answered(self, number: int, outcome: tuple[bool, object]) -> None:
        future = self._asked.get(number)
        if future is not None and not future.done():
            future.set_result(outcome)


#: Set in a worker process (:mod:`actant.sandbox.worker`).
_link: HostLink | None = None


def link(writer: asyncio.StreamWriter) -> HostLink:
    """This worker's way to the host, for :func:`on_host`."""
    global _link
    _link = HostLink(writer)
    return _link


async def _resolve(name: str) -> Callable[..., object]:
    if name not in _functions:  # its module is imported in the worker, maybe not yet here
        importlib.import_module(name.partition(":")[0])
    if name not in _functions:
        raise TypeError(f"{name} is not a @host_function")
    return _functions[name]


async def send(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(data)  # the whole frame before any await: frames never interleave
    await writer.drain()


async def receive(reader: asyncio.StreamReader) -> tuple[str, int, object]:
    (size,) = _SIZE.unpack(await reader.readexactly(_SIZE.size))
    kind, number, payload = pickle.loads(await reader.readexactly(size))
    return kind, number, payload


class Worker:
    """One worker process and its calls in flight, by number."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.process, self.reader, self.writer = process, reader, writer
        self.alive = True
        self._numbers = itertools.count()
        self._calls: dict[int, asyncio.Future[CallResponse]] = {}
        self._answering: set[asyncio.Task[None]] = set()
        self._listening = asyncio.ensure_future(self._listen())

    @classmethod
    async def spawn(cls, config: WorkerConfig, env: Mapping[str, str]) -> Worker:
        ours, theirs = socket.socketpair()
        with theirs:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "actant.sandbox.worker",
                config.model_dump_json(),
                str(theirs.fileno()),
                stdin=asyncio.subprocess.DEVNULL,
                pass_fds=(theirs.fileno(),),
                env={**os.environ, **env},
            )
        reader, writer = await asyncio.open_connection(sock=ours)
        return cls(process, reader, writer)

    async def call(self, request: CallRequest) -> CallResponse:
        if not self.alive:
            return self._died()
        number = next(self._numbers)
        future = self._calls[number] = asyncio.get_running_loop().create_future()
        try:
            await send(self.writer, frame(Kind.CALL, number, request))
            return await future
        except ConnectionError:
            return self._died()
        except asyncio.CancelledError:
            if self.alive:
                with contextlib.suppress(ConnectionError):
                    await send(self.writer, frame(Kind.CANCEL, number, request.call_id))
            raise
        finally:
            self._calls.pop(number, None)

    async def _answer(self, number: int, asked: object) -> None:
        try:
            name, args, kwargs = cast(tuple[str, tuple[object, ...], dict[str, object]], asked)
            data = answer(number, True, await _run(await _resolve(name), args, kwargs))
        except Exception as error:  # noqa: BLE001 -- raised in the worker's caller
            data = answer(number, False, error)
        with contextlib.suppress(ConnectionError):
            await send(self.writer, data)

    def _died(self) -> CallResponse:
        code = self.process.returncode
        return CallResponse(
            error=f"WorkerDied: the process serving this call exited (code {code}) before it "
            "answered; the call may have run in part"
        )

    async def _listen(self) -> None:
        try:
            while True:
                kind, number, payload = await receive(self.reader)
                if kind == Kind.ASK:
                    # its own task: a host function may await a call this worker answers
                    task = asyncio.create_task(self._answer(number, payload))
                    self._answering.add(task)
                    task.add_done_callback(self._answering.discard)
                    continue
                future = self._calls.get(number)
                if kind == Kind.DONE and future is not None and not future.done():
                    assert isinstance(payload, CallResponse)
                    future.set_result(payload)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as error:  # noqa: BLE001 -- an unreadable message loses the worker
            print(f"worker {self.process.pid} dropped: {error!r}", file=sys.stderr, flush=True)
        finally:
            self.alive = False
            self.writer.close()
            with contextlib.suppress(ProcessLookupError):
                self.process.kill()
            await self.process.wait()
            for future in self._calls.values():
                if not future.done():
                    future.set_result(self._died())

    async def close(self) -> None:
        """End the worker's input (it closes its instances and exits); kill it after
        :data:`CLOSE_GRACE_S`."""
        if self.writer.can_write_eof() and not self.writer.is_closing():
            self.writer.write_eof()
        try:
            await asyncio.wait_for(self.process.wait(), CLOSE_GRACE_S)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                self.process.kill()
        await asyncio.wait([self._listening])


class Slot:
    """One worker's place: started on first use, replaced once after it exits."""

    def __init__(self, start: Callable[[], Awaitable[Worker]]) -> None:
        self._start = start
        self._starts = 0
        self._current: asyncio.Future[Worker] | None = None

    def _spent(self) -> bool:
        current = self._current
        if current is None:
            return True
        if not current.done():
            return False
        return current.cancelled() or current.exception() is not None or not current.result().alive

    async def worker(self) -> Worker | None:
        """The running worker; ``None`` once it could not be started or replaced."""
        if self._spent():
            if self._starts >= STARTS:
                return None
            self._starts += 1
            self._current = asyncio.ensure_future(self._start())
        assert self._current is not None
        try:
            return await asyncio.shield(self._current)
        except OSError as error:
            print(f"could not start a worker: {error!r}", file=sys.stderr, flush=True)
            return None

    async def close(self) -> None:
        if self._current is not None and not self._spent():
            await (await self._current).close()


class Workers:
    """A placed service's slots; each key keeps the slot it was dealt."""

    def __init__(self, service: str, config: ServiceConfig, scrub: Iterable[str]) -> None:
        worker = WorkerConfig(service=service, path=config.path, scrub=list(scrub))
        self.slots = [
            Slot(lambda: Worker.spawn(worker, config.env)) for _ in range(config.processes)
        ]
        self.keys: dict[str, Slot] = {}

    def slot(self, key: str) -> Slot:
        slot = self.keys.get(key)
        if slot is None:
            slot = self.keys[key] = self.slots[len(self.keys) % len(self.slots)]
        return slot

    async def call(self, request: CallRequest) -> CallResponse | None:
        """The call run by its key's worker; ``None`` when the host must run it."""
        worker = await self.slot(request.key).worker()
        return None if worker is None else await worker.call(request)

    async def close(self) -> None:
        await asyncio.gather(*(slot.close() for slot in self.slots))
