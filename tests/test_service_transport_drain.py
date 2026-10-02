"""Local and remote caller exits cannot silently leave service writes running."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus

import pytest

from actant.sandbox import Endpoint, host
from actant.sandbox.protocol import CallResponse, CancelRequest
from actant.sandbox.service import LocalRunner, ServiceDrainError, call_host


@asynccontextmanager
async def server_for(serving: host.Host) -> AsyncIterator[tuple[Endpoint, asyncio.Server]]:
    connections: set[asyncio.Task[None]] = set()

    async def connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        connections.add(task)
        try:
            await serving.connection(reader, writer)
        finally:
            connections.discard(task)

    server = await asyncio.start_server(connection, "127.0.0.1", 0)
    assert server.sockets
    endpoint = Endpoint(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    try:
        yield endpoint, server
    finally:
        server.close()
        await server.wait_closed()
        for writer in list(serving.writers):
            writer.close()
        if connections:
            await asyncio.wait_for(asyncio.gather(*connections, return_exceptions=True), 2)
        await serving.close_instances()


async def test_cancelled_local_sync_call_waits_for_thread_write() -> None:
    started, release, finished = (threading.Event() for _ in range(3))
    effects: list[str] = []

    class Author:
        def write(self) -> str:
            started.set()
            assert release.wait(5)
            effects.append("old-write")
            finished.set()
            return "written"

    caller = asyncio.create_task(LocalRunner(Author()).call("write", {}))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        caller.cancel()
        await asyncio.sleep(0)
        caller.cancel()
        await asyncio.sleep(0)
        assert not caller.done() and not finished.is_set()
        release.set()
        result = await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), 2)
        assert isinstance(result[0], asyncio.CancelledError) and finished.is_set()
        effects.append("replacement-start")
        assert effects == ["old-write", "replacement-start"]
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


async def test_http_timeout_waits_for_remote_drain_before_returning_error() -> None:
    started, release, finished, cancellation = (asyncio.Event() for _ in range(4))

    class Author:
        async def write(self) -> str:
            started.set()
            await release.wait()
            finished.set()
            return "written"

    class ObservedHost(host.Host):
        async def cancel(self, request: CancelRequest) -> tuple[HTTPStatus, CallResponse]:
            cancellation.set()
            return await super().cancel(request)

    async def release_on_cancel() -> None:
        await asyncio.wait_for(cancellation.wait(), 2)
        release.set()

    async with server_for(ObservedHost({"author": Author})) as (endpoint, _):
        unblock = asyncio.create_task(release_on_cancel())
        caller = asyncio.create_task(
            call_host(endpoint, "author", "write", {}, key="k", timeout=0.25)
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            reply = await asyncio.wait_for(caller, 2)
            assert cancellation.is_set() and finished.is_set()
            assert reply.error is not None and "effects drained" in reply.error
        finally:
            release.set()
            await asyncio.gather(caller, unblock, return_exceptions=True)


async def test_unreachable_cancel_endpoint_raises_uncertain_drain_error() -> None:
    started, release, finished = (asyncio.Event() for _ in range(3))

    class Author:
        async def write(self) -> str:
            started.set()
            await release.wait()
            finished.set()
            return "written"

    async with server_for(host.Host({"author": Author})) as (endpoint, server):
        caller = asyncio.create_task(
            call_host(endpoint, "author", "write", {}, key="k", timeout=0.25)
        )
        try:
            await asyncio.wait_for(started.wait(), 2)
            # The existing call is still connected; only a new drain connection is refused.
            server.close()
            with pytest.raises(ServiceDrainError, match="drain unresolved"):
                await asyncio.wait_for(caller, 2)
            assert not finished.is_set(), "test needs a genuinely unresolved remote writer"
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)
            await asyncio.wait_for(finished.wait(), 2)
