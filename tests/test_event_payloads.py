from unittest.mock import AsyncMock

import pytest

from actant.runtime.events.payloads import ToolResultPayload
from actant.runtime.events.publisher import ScopedEventSink


async def test_invalid_observation_does_not_escape_to_activity():
    sink = AsyncMock()
    scoped = ScopedEventSink(sink, "agent", "run", "turn")
    await scoped.publish("thread:t", {"type": "tool_result", "data": {"result": {}}})
    sink.publish.assert_not_awaited()


async def test_tool_event_preserves_partial_output_and_absent_fields():
    sink = AsyncMock()
    scoped = ScopedEventSink(sink, "agent", "run", "turn")
    await scoped.publish(
        "thread:t",
        {
            "type": "tool_result",
            "data": {
                "tool_call_id": "call",
                "result": {"result": {"partial": 1}, "error": "failed"},
            },
        },
    )
    data = sink.publish.await_args.args[1]["data"]
    parsed = ToolResultPayload.model_validate(data)
    assert parsed.result.result == {"partial": 1}
    assert parsed.result.error == "failed"
    assert parsed.turn_id == "turn"
    assert "content_blocks" not in data["result"]


async def test_observer_cancellation_propagates():
    import asyncio

    sink = AsyncMock()
    sink.publish.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await ScopedEventSink(sink, "agent", None, None).publish(
            "thread:t", {"type": "stream_reset", "data": {}}
        )
