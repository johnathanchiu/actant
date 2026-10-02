"""A caller that gives up on a call stops it on the host, without waiting for it."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import pytest

from actant.sandbox import Endpoint, host, service
from actant.sandbox.protocol import CallRequest


class Writer:
    events: list[str] = []
    started: asyncio.Event

    async def write_later(self) -> str:
        Writer.started.set()
        try:
            await asyncio.Event().wait()
            Writer.events.append("late-write")  # never: the call was stopped
            return "written"
        finally:
            Writer.events.append("cleanup")


@pytest.fixture
async def served() -> AsyncIterator[tuple[host.Host, Endpoint]]:
    Writer.events, Writer.started = [], asyncio.Event()
    serving = host.Host({"w": Writer})
    server = await asyncio.start_server(serving.connection, "127.0.0.1", 0)
    yield serving, Endpoint(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    server.close()
    for writer in list(serving.writers):
        writer.close()


async def test_a_cancelled_caller_stops_its_call_on_the_host(
    served: tuple[host.Host, Endpoint],
) -> None:
    serving, endpoint = served
    caller = asyncio.create_task(
        service.call_host(endpoint, "w", "write_later", {}, key="k", call_id="c")
    )
    await asyncio.wait_for(Writer.started.wait(), 2)
    started = time.monotonic()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert time.monotonic() - started < 1.0
    assert Writer.events == ["cleanup"]  # the method's finally ran before the cancel returned
    assert serving.calls == {}


async def test_a_caller_that_times_out_stops_its_call(served: tuple[host.Host, Endpoint]) -> None:
    serving, endpoint = served
    reply = await service.call_host(endpoint, "w", "write_later", {}, key="k", timeout=0.3)
    assert reply.error is not None and "may have run" in reply.error
    assert Writer.events == ["cleanup"]
    assert serving.calls == {}


async def test_a_call_attached_to_a_cancelled_run_answers_cancelled() -> None:
    Writer.events, Writer.started = [], asyncio.Event()
    serving = host.Host({"w": Writer})
    request = CallRequest(service="w", key="k", method="write_later", call_id="c")
    waiting = asyncio.create_task(serving.call(request))
    await asyncio.wait_for(Writer.started.wait(), 2)
    assert (await serving.cancel("c")).text == "cancelled"
    _, reply = await waiting
    assert reply.error is not None and reply.error.startswith("cancelled")
    assert (await serving.cancel("c")).text == "not running"


async def test_a_cancel_waits_only_a_short_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host, "CANCEL_GRACE_S", 0.1)

    class Stubborn:
        async def hold(self) -> str:
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.shield(asyncio.sleep(0.5))  # slow cleanup
            return "never"

    serving = host.Host({"s": Stubborn})
    request = CallRequest(service="s", key="k", method="hold", call_id="c")
    waiting = asyncio.create_task(serving.call(request))
    await asyncio.sleep(0.01)
    started = time.monotonic()
    assert (await serving.cancel("c")).text == "cancelling"
    assert time.monotonic() - started < 0.4
    await waiting
