"""A host runs each call id once: a call sent again attaches to its run."""

from __future__ import annotations

import asyncio
from http import HTTPStatus

import pytest

from actant.sandbox import Endpoint, host, service
from actant.sandbox.protocol import CallRequest


class Counter:
    runs = 0
    gate: asyncio.Event

    async def add(self, n: int) -> int:
        Counter.runs += 1
        await Counter.gate.wait()
        return n + 1


@pytest.fixture(autouse=True)
def _reset() -> None:
    Counter.runs = 0
    Counter.gate = asyncio.Event()


def _request(n: int = 1, call_id: str = "thread:call") -> CallRequest:
    return CallRequest(service="s", key="k", method="add", args={"n": n}, call_id=call_id)


async def test_a_call_sent_again_while_it_runs_attaches_to_the_run() -> None:
    serving = host.Host({"s": Counter})
    first = asyncio.create_task(serving.call(_request()))
    await asyncio.sleep(0)
    again = asyncio.create_task(serving.call(_request()))
    await asyncio.sleep(0.01)
    first.cancel()  # its caller went away: the run goes on for the retry
    Counter.gate.set()
    assert await again == (HTTPStatus.OK, (await serving.calls["thread:call"].task))
    assert (await again)[1].text == "2"
    assert Counter.runs == 1


async def test_a_finished_call_sent_again_gets_its_result_without_running() -> None:
    serving = host.Host({"s": Counter})
    Counter.gate.set()
    first = await serving.call(_request())
    assert await serving.call(_request()) == first
    assert Counter.runs == 1


async def test_a_different_call_under_a_known_id_is_refused() -> None:
    serving = host.Host({"s": Counter})
    Counter.gate.set()
    await serving.call(_request(1))
    status, reply = await serving.call(_request(5))
    assert status is HTTPStatus.CONFLICT
    assert reply.error is not None and "different call" in reply.error
    assert Counter.runs == 1


async def test_finished_results_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host, "MAX_FINISHED", 2)
    serving = host.Host({"s": Counter})
    Counter.gate.set()
    for index in range(5):
        await serving.call(_request(call_id=f"c{index}"))
    assert sorted(serving.calls) == ["c2", "c3", "c4"]  # the oldest dropped first
    monkeypatch.setattr(host, "KEEP_FINISHED_S", 0.0)
    await serving.call(_request(call_id="c5"))
    assert sorted(serving.calls) == ["c5"]


async def test_a_running_call_is_never_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host, "MAX_FINISHED", 0)
    monkeypatch.setattr(host, "KEEP_FINISHED_S", 0.0)
    serving = host.Host({"s": Counter})
    running = asyncio.create_task(serving.call(_request(call_id="slow")))
    await asyncio.sleep(0)
    other = asyncio.create_task(serving.call(_request(call_id="other")))
    await asyncio.sleep(0)
    assert sorted(serving.calls) == ["other", "slow"]
    Counter.gate.set()
    await asyncio.gather(running, other)


async def test_a_call_sent_again_after_its_result_was_delivered_does_not_run_again() -> None:
    serving = host.Host({"s": Counter})
    server = await asyncio.start_server(serving.connection, "127.0.0.1", 0)
    endpoint = Endpoint(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    Counter.gate.set()
    try:
        # The worker dies after the result was written to it, before it stored the result.
        for _ in range(2):
            reply = await service.call_host(endpoint, "s", "add", {"n": 1}, key="k", call_id="x")
            assert reply.text == "2"
        assert Counter.runs == 1
    finally:
        server.close()
        for writer in list(serving.writers):
            writer.close()
        await server.wait_closed()
