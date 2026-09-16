"""Default worker event wiring."""

from __future__ import annotations

from typing import cast
from temporalio.client import Client
from unittest.mock import Mock

import pytest

from actant.runtime.events import PublishingStreamListener, PublishingThreadHooks
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime import AgentRuntime
from actant.runtime.types.threads import AgentThread


@pytest.mark.asyncio
async def test_worker_publishes_events_without_custom_factories() -> None:
    stores = InMemoryRuntimeStores()
    worker = AgentRuntime(client=cast(Client, Mock()), stores=stores)
    thread = AgentThread(id="thread-id", agent_id="assistant")

    hooks = worker._context.hooks(thread)
    listener = worker._context.listener(thread)

    assert isinstance(hooks, PublishingThreadHooks)
    assert isinstance(listener, PublishingStreamListener)

    await hooks.on_turn_start(1, "turn-id")
    await listener.on_text_delta("Hello")

    assert stores.publisher.events["thread:thread-id"] == [
        {
            "type": "turn_start",
            "thread_id": "thread-id",
            "data": {
                "agent_id": "assistant",
                "turn": 1,
                "turn_id": "turn-id",
                "turn_uid": "turn-id",
            },
        },
        {
            "type": "text_delta",
            "thread_id": "thread-id",
            "data": {"agent_id": "assistant", "delta": "Hello"},
        },
    ]


async def test_observer_factory_failure_is_observational() -> None:
    from actant.runtime.temporal.activities.context import ActivityContext

    def broken(thread: AgentThread):
        raise RuntimeError("event configuration unavailable")

    context = ActivityContext(
        stores=InMemoryRuntimeStores(), hooks_factory=broken, listener_factory=broken
    )
    thread = AgentThread(id="t", agent_id="a")
    await context.hooks(thread).on_turn_start(1, "turn")
    await context.listener(thread).on_text_delta("hello")
