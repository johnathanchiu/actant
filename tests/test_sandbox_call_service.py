"""The service a host call serves, seen by what the call runs and starts."""

from __future__ import annotations

import asyncio

import pytest

from actant.sandbox import host
from actant.sandbox.protocol import CallRequest


class Seen:
    async def here(self) -> str | None:
        return host.current_service()

    async def in_a_task(self) -> str | None:
        return await asyncio.create_task(_service())

    def in_a_thread(self) -> str | None:
        return host.current_service()


async def _service() -> str | None:
    return host.current_service()


@pytest.mark.parametrize("method", ["here", "in_a_task", "in_a_thread"])
async def test_a_call_sees_the_service_it_serves(method: str) -> None:
    serving = host.Host({"planner": Seen, "author": Seen})
    for service in ("planner", "author"):
        status, reply = await serving.call(
            CallRequest(service=service, key="k", method=method, args={})
        )
        assert reply.text == service, (status, reply)
    assert host.current_service() is None
