"""MessageStore round-trips for the parts-based persistence model.

Exercises the multimodal carrier through the ``MessageStore`` Protocol
via the in-memory backend — the SQL-backed backends have parallel
implementations and pick up integration coverage from downstream apps.
"""

from __future__ import annotations

import json

import pytest

from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.runtime.stores.in_memory import InMemoryMessageStore


@pytest.mark.asyncio
async def test_append_user_string_round_trips() -> None:
    store = InMemoryMessageStore()
    await store.append_user("agent_1", "thread_x", "hello world")

    [message] = await store.list_for_thread("agent_1", "thread_x")
    assert message.role == "user"
    assert message.content == "hello world"


@pytest.mark.asyncio
async def test_append_user_multimodal_round_trips() -> None:
    store = InMemoryMessageStore()
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "describe this"},
        {
            "type": "asset",
            "storage_key": "user_uploads/abc/xyz",
            "mime": "image/png",
            "asset_public_id": "asset_xyz",
        },
    ]

    await store.append_user("agent_1", "thread_x", blocks)
    [message] = await store.list_for_thread("agent_1", "thread_x")
    assert message.role == "user"
    assert message.content == blocks


@pytest.mark.asyncio
async def test_append_assistant_with_text_and_tool_calls_round_trips() -> None:
    store = InMemoryMessageStore()
    assistant = Message(
        role="assistant",
        content="let me check that",
        tool_calls=[
            ToolCall(
                id="tc_1",
                function=ToolCallFunction(name="lookup", arguments='{"q": "weather"}'),
            )
        ],
    )

    await store.append_assistant_with_tool_calls("agent_1", "thread_x", "turn_1", assistant, [])
    [reconstructed] = await store.list_for_thread("agent_1", "thread_x")
    assert reconstructed.role == "assistant"
    assert reconstructed.content == "let me check that"
    assert reconstructed.tool_calls is not None
    assert reconstructed.tool_calls[0].id == "tc_1"


@pytest.mark.asyncio
async def test_append_tool_result_string_round_trips_legacy_shape() -> None:
    store = InMemoryMessageStore()
    await store.append_tool_result(
        "agent_1",
        "thread_x",
        "turn_1",
        "tc_1",
        "lookup",
        {"output": "ok", "metadata": {"exit_code": 0}},
    )
    [message] = await store.list_for_thread("agent_1", "thread_x")
    assert message.role == "tool"
    assert message.tool_call_id == "tc_1"
    assert message.content == json.dumps({"output": "ok", "metadata": {"exit_code": 0}})


@pytest.mark.asyncio
async def test_append_tool_result_with_content_blocks_round_trips_multimodal() -> None:
    store = InMemoryMessageStore()
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "rendered the floorplan"},
        {
            "type": "asset",
            "storage_key": "agent_artifacts/render-1.png",
            "mime": "image/png",
        },
    ]
    result = {"content_blocks": blocks, "metadata": {"width": 1024}}

    await store.append_tool_result("agent_1", "thread_x", "turn_1", "tc_1", "render", result)
    [message] = await store.list_for_thread("agent_1", "thread_x")
    assert message.role == "tool"
    assert message.tool_call_id == "tc_1"
    assert message.content == blocks


@pytest.mark.asyncio
async def test_append_tool_result_is_idempotent_per_tool_call_id() -> None:
    """Two tool-result writes for the same tool_call_id collapse into
    one persisted message — same guarantee the SQL backends provide
    via the unique-by-(thread, tool_call_id) lookup."""
    store = InMemoryMessageStore()
    await store.append_tool_result(
        "agent_1", "thread_x", "turn_1", "tc_1", "lookup", {"output": "first"}
    )
    await store.append_tool_result(
        "agent_1", "thread_x", "turn_1", "tc_1", "lookup", {"output": "second"}
    )

    messages = await store.list_for_thread("agent_1", "thread_x")
    tool_messages = [m for m in messages if m.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == json.dumps({"output": "first"})


@pytest.mark.asyncio
async def test_thread_round_trip_preserves_message_order() -> None:
    """Full conversation round-trip — user → assistant + tool_calls →
    tool result → user. The order matters because the LLM context build
    depends on it."""
    store = InMemoryMessageStore()

    await store.append_user("agent_1", "thread_x", "hi")
    await store.append_assistant_with_tool_calls(
        "agent_1",
        "thread_x",
        "turn_1",
        Message(
            role="assistant",
            content="thinking",
            tool_calls=[
                ToolCall(
                    id="tc_1",
                    function=ToolCallFunction(name="echo", arguments='{"x": 1}'),
                )
            ],
        ),
        [],
    )
    await store.append_tool_result(
        "agent_1", "thread_x", "turn_1", "tc_1", "echo", {"output": "x=1"}
    )
    await store.append_user("agent_1", "thread_x", "thanks")

    messages = await store.list_for_thread("agent_1", "thread_x")
    assert [m.role for m in messages] == ["user", "assistant", "tool", "user"]
