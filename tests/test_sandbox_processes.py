"""A service placed in worker processes answers as the host would, each key on one worker."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from actant.sandbox import host
from actant.sandbox.protocol import CallRequest, CallResponse, ServiceConfig
from actant.sandbox.processes import on_host
from service_fixtures import HOST, Placed, where

TESTS = str(Path(__file__).parent)


def placed(processes: int, **env: str) -> ServiceConfig:
    return ServiceConfig(
        path="service_fixtures:Placed", processes=processes, env={"PYTHONPATH": TESTS, **env}
    )


@pytest.fixture
async def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[host.Host]:
    monkeypatch.chdir(tmp_path)
    serving = host.Host({"p": Placed}, placed={"p": placed(2, MARK="worker")})
    yield serving
    await serving.close_instances()


async def call(serving: host.Host, key: str, method: str, **args: object) -> CallResponse:
    _, response = await serving.call(CallRequest(service="p", key=key, method=method, args=args))
    return response


async def test_each_key_stays_on_the_worker_it_was_dealt(served: host.Host) -> None:
    pids = {key: (await call(served, key, "pid")).text for key in "abc"}
    assert pids["a"] != pids["b"] and pids["c"] == pids["a"]
    assert str(os.getpid()) not in pids.values()
    for key in "cba":
        assert (await call(served, key, "pid")).text == pids[key]
    env = await call(served, "a", "env", name="MARK")
    assert env.text == '{"host": "worker", "script": "worker"}'


async def test_a_worker_answers_as_the_host_does(served: host.Host) -> None:
    local = host.Host({"p": Placed})
    calls: list[tuple[str, dict[str, object]]] = [
        ("bump", {"by": 2}),
        ("bump", {}),
        ("area", {"box": {"width": 2, "height": 3}, "unit": "cm"}),
        ("picture", {"size": 16, "as_file": True}),
        ("bump", {"by": "two"}),
        ("fail", {}),
    ]
    for method, args in calls:
        ours = await call(served, "k", method, **args)
        _, theirs = await local.call(CallRequest(service="p", key="k", method=method, args=args))
        if method == "picture":  # random bytes: compare the shape
            assert [i.name for i in ours.images] == [i.name for i in theirs.images]
            assert ours.text == theirs.text
        else:
            assert ours.model_dump(exclude={"error"}) == theirs.model_dump(exclude={"error"})
            assert (ours.error or "").splitlines()[:1] == (theirs.error or "").splitlines()[:1]


async def test_a_dead_worker_fails_its_call_is_replaced_once_then_the_host_serves(
    served: host.Host,
) -> None:
    first = (await call(served, "a", "pid")).text
    crashed = await call(served, "a", "crash")
    assert crashed.error is not None and crashed.error.startswith("WorkerDied")
    second = (await call(served, "a", "pid")).text
    assert second not in {first, str(os.getpid())}
    assert (await call(served, "a", "bump")).text == '{"count": 1}'  # opened afresh
    crashed = await call(served, "a", "crash")
    assert crashed.error is not None and crashed.error.startswith("WorkerDied")
    assert (await call(served, "a", "pid")).text == str(os.getpid())
    # the other worker's keys never noticed
    assert (await call(served, "b", "pid")).text not in {first, second, str(os.getpid())}


async def test_a_cancel_reaches_the_worker(served: host.Host, tmp_path: Path) -> None:
    request = CallRequest(service="p", key="a", method="hold", args={"marker": "m"}, call_id="c")
    calling = asyncio.create_task(served.call(request))
    while not (tmp_path / "m").exists():
        await asyncio.sleep(0.02)
    assert (await served.cancel("c")).text == "cancelled"
    await asyncio.wait_for(calling, 2)
    for _ in range(100):
        if (tmp_path / "m.cancelled").exists():
            break
        await asyncio.sleep(0.02)
    assert (tmp_path / "m.cancelled").exists()


async def test_closing_the_host_closes_the_workers_instances(served: host.Host) -> None:
    await call(served, "a", "bump", by=5)
    await call(served, "b", "bump", by=7)
    await served.close_instances()
    assert sorted(p.name for p in Path().glob("closed-at-*")) == ["closed-at-5", "closed-at-7"]


async def test_a_worker_runs_host_functions_on_the_host(served: host.Host) -> None:
    here = str(os.getpid())
    assert (await call(served, "a", "where", tag="t")).text == f"t {here}"
    bad = await call(served, "a", "where", tag="bad")
    assert bad.error is not None and bad.error.startswith("KeyError: 'bad'")
    refused = await call(served, "a", "unregistered")
    assert refused.error is not None and "is not a @host_function" in refused.error
    # in the host itself the function runs right there
    assert await on_host(where, "h") == f"h {here}"


async def test_a_host_function_may_await_the_worker_that_asked(served: host.Host) -> None:
    HOST[:] = [served]
    worker = (await call(served, "a", "pid")).text
    relayed = await asyncio.wait_for(call(served, "a", "relay", to="a"), 5)
    assert relayed.text == worker


async def test_a_worker_says_when_it_starts_and_how_many_calls_it_served(
    served: host.Host, capfd: pytest.CaptureFixture[str]
) -> None:
    pids = {key: (await call(served, key, "pid")).text for key in "abc"}
    await call(served, "a", "bump")
    await served.close_instances()
    err = capfd.readouterr().err
    a, b = pids["a"], pids["b"]
    assert f"worker {a} started for p" in err and f"worker {b} started for p" in err
    assert f"worker {a} for p gone: served 3 calls" in err
    assert f"worker {b} for p gone: served 1 calls" in err
