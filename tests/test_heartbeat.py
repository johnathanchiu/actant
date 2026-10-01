"""Slow sandbox work beats the caller's Temporal activity while it runs."""

from __future__ import annotations

import asyncio

import pytest
from temporalio.testing import ActivityEnvironment

import actant.heartbeat
import actant.sandbox.service as service
from actant.heartbeat import heartbeating
from actant.sandbox.base import Endpoint
from actant.sandbox.protocol import CallResponse


@pytest.fixture(autouse=True)
def _fast_beats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(actant.heartbeat, "HEARTBEAT_EVERY_S", 0.02)


async def test_outside_an_activity_the_body_just_runs() -> None:
    async with heartbeating():
        await asyncio.sleep(0.05)


async def test_a_service_call_beats_the_activity_it_runs_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_post(*args: object) -> tuple[int | None, CallResponse]:
        await asyncio.sleep(0.2)
        return 200, CallResponse(text="done")

    monkeypatch.setattr(service, "_post", slow_post)
    beats: list[object] = []
    env = ActivityEnvironment()
    env.on_heartbeat = lambda *details: beats.append(details)
    endpoint = Endpoint(url="http://host.test")

    async def call() -> tuple[int | None, CallResponse]:
        return await service._send(endpoint, b"{}", 1.0)

    status, response = await env.run(call)
    assert status == 200 and response.text == "done"
    assert len(beats) >= 3
    count = len(beats)
    await asyncio.sleep(0.1)
    assert len(beats) == count  # the beats stop with the call
