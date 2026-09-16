"""Single event path: immutable activity identity, structured results and isolation."""

import asyncio
from typing import cast
from unittest.mock import Mock
import pytest
from temporalio.client import Client
from actant.core import JSONObject
from actant.runtime import AgentRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.types.threads import AgentThread
from actant.runtime.temporal.activities.context import ActivityContext
from actant.tools import ToolResult


async def test_runtime_events_require_explicit_sink() -> None:
    stores = InMemoryRuntimeStores()
    runtime = AgentRuntime(client=cast(Client, Mock()), stores=stores)
    thread = AgentThread(id="t", agent_id="a")
    await runtime._context.events(thread).on_text_delta("ignored")
    assert not stores.publisher.events
    context = ActivityContext(stores=stores, event_sink=stores.publisher)
    first = context.events(thread, run_id="r1", turn_id="turn1", turn_index=1)
    second = context.events(thread, run_id="r2", turn_id="turn2", turn_index=2)
    await second.on_turn_start(2, "turn2")
    await first.on_text_delta("late first turn")
    await first.on_tool_result("call", ToolResult(output={"value": 1}, metadata={"artifacts": []}))
    events = stores.publisher.events["thread:t"]
    assert events[1]["data"] == {
        "agent_id": "a",
        "run_id": "r1",
        "turn_id": "turn1",
        "turn_uid": "turn1",
        "turn_index": 1,
        "delta": "late first turn",
    }
    data = events[2]["data"]
    assert isinstance(data, dict)
    assert data["result"] == {"result": {"value": 1}, "metadata": {"artifacts": []}}


@pytest.mark.parametrize("cancel", [False, True])
async def test_event_failure_is_observational_but_cancellation_propagates(cancel: bool) -> None:
    class Sink:
        async def publish(self, channel: str, event: JSONObject) -> None:
            if cancel:
                raise asyncio.CancelledError()
            raise RuntimeError("offline")

    context = ActivityContext(stores=InMemoryRuntimeStores(), event_sink=Sink())
    event = context.events(AgentThread(id="t", agent_id="a"))
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await event.on_stream_reset()
    else:
        await event.on_stream_reset()
        await event.on_complete(True, "done", "done")


async def test_tool_event_retains_partial_output_with_error() -> None:
    stores = InMemoryRuntimeStores()
    context = ActivityContext(stores=stores, event_sink=stores.publisher)
    events = context.events(AgentThread(id="t", agent_id="a"))
    await events.on_tool_result("call", ToolResult(output={"partial": True}, error="interrupted"))
    data = stores.publisher.events["thread:t"][0]["data"]
    assert isinstance(data, dict)
    assert data["result"] == {"result": {"partial": True}, "error": "interrupted"}
