"""Default worker event wiring."""

from __future__ import annotations

import pytest

from actant.runtime.events import PublishingStreamListener, PublishingThreadHooks
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.worker import TemporalRuntimeWorker
from actant.runtime.types.threads import AgentThread


@pytest.mark.asyncio
async def test_worker_publishes_events_without_custom_factories() -> None:
    stores = InMemoryRuntimeStores()
    worker = TemporalRuntimeWorker(stores=stores, agents={})
    thread = AgentThread(id="thread-id", agent_id="assistant")

    hooks = worker._activities._hooks(thread)
    listener = worker._activities._listener(thread)

    assert isinstance(hooks, PublishingThreadHooks)
    assert isinstance(listener, PublishingStreamListener)

    await hooks.on_turn_start(1, "turn-id")
    await listener.on_text_delta("Hello")

    assert stores.publisher.events["thread:thread-id"] == [
        {
            "type": "turn_start",
            "thread_id": "thread-id",
            "data": {"turn": 1, "turn_id": "turn-id", "turn_uid": "turn-id"},
        },
        {
            "type": "text_delta",
            "thread_id": "thread-id",
            "data": {"delta": "Hello"},
        },
    ]
