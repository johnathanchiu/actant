"""Services served from worker processes beside the host (:class:`ServiceConfig.processes`).

The host still takes every request: its checks, call ids, cancels, image uploads and pushes.
A placed service's call is run by the worker its key was dealt to on the key's first call; the
worker (:mod:`actant.sandbox.worker`) serves its keys' instances as the host would, so it
validates the arguments and encodes the result, and the host sends that response on.

A worker that exits fails its calls in flight (never sent again: each may have run in part).
Its slot starts one replacement; once that one is gone too, the slot's keys are served in the
host. A worker exits, closing its instances, when its socket closes (the host shut down or
died).

Messages are length-prefixed pickles over a socketpair: both ends are this package, and
nothing else can reach the socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import pickle
import socket
import struct
import sys
from collections.abc import Awaitable, Callable, Iterable, Mapping
from enum import StrEnum

from actant.sandbox.protocol import CallRequest, CallResponse, ServiceConfig, WorkerConfig

#: A slot's starts: its first worker and one replacement.
STARTS = 2
#: How long a closing worker may take to close its instances before it is killed.
CLOSE_GRACE_S = 10.0

_SIZE = struct.Struct("!Q")


class Kind(StrEnum):
    CALL = "call"
    CANCEL = "cancel"
    DONE = "done"


def frame(kind: str, number: int, payload: object) -> bytes:
    data = pickle.dumps((kind, number, payload), pickle.HIGHEST_PROTOCOL)
    return _SIZE.pack(len(data)) + data


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
