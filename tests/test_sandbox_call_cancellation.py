"""Caller cancellation must drain service effects before its workspace is reused."""

from __future__ import annotations

import asyncio
import contextlib

from actant.sandbox import Endpoint, host
from actant.sandbox.service import call_host


async def test_cancelled_host_call_drains_before_replacement_starts() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    effects: list[str] = []

    class Author:
        async def write_later(self) -> str:
            started.set()
            try:
                await release.wait()
                effects.append("old-author-write")
                return "written"
            finally:
                finished.set()

    serving = host.Host({"author": Author})
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
    caller = asyncio.create_task(
        call_host(endpoint, "author", "write_later", {}, key="old-author", timeout=5)
    )
    delayed_release: asyncio.TimerHandle | None = None
    try:
        await asyncio.wait_for(started.wait(), 2)
        # Also allow a correct implementation to drain an uncancellable method.
        # Its completion is independent of the caller returning from cancellation.
        delayed_release = asyncio.get_running_loop().call_later(0.1, release.set)
        caller.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(caller, 2)
        host_finished_when_caller_drained = finished.is_set()

        # A scheduler awaiting the cancelled caller can now reuse the same folder.
        effects.append("replacement-start")
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
        assert host_finished_when_caller_drained and "old-author-write" not in effects[
            effects.index("replacement-start") + 1 :
        ], (
            "A drained caller still allows its remote author to write after replacement: "
            f"{effects}"
        )
    finally:
        if delayed_release is not None:
            delayed_release.cancel()
        # Release the unfixed host method even when an assertion or timeout fails.
        release.set()
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        server.close()
        await server.wait_closed()
        for writer in list(serving.writers):
            writer.close()
        if connections:
            await asyncio.wait_for(asyncio.gather(*connections, return_exceptions=True), 2)
        await serving.close_instances()
