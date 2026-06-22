"""``MessageStore`` — append-only message log per (agent, thread).

The transcript invariant the runtime promises: every assistant
message that claims a tool_call_id has a corresponding
ToolCallRecord, AND every tool_call eventually has a matching
tool_result. Without this, the next LLM call 400s.

The atomicity test below pins
``append_assistant_with_tool_calls`` writing both rows together —
in production (postgres) that's literal transaction semantics.

One behavior per test.
"""

from __future__ import annotations

import pytest

from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools.calls import ToolCallRecord, ToolCallStatus

_AGENT = "a"
_THREAD = "t"


# --- append_user ---


@pytest.mark.asyncio
async def test_append_user_adds_user_message_to_log() -> None:
    stores = InMemoryRuntimeStores()
    await stores.messages.append_user(_AGENT, _THREAD, "hello")

    log = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert len(log) == 1
    assert log[0].role == "user"
    assert log[0].content == "hello"


@pytest.mark.asyncio
async def test_append_user_supports_multimodal_blocks() -> None:
    """Multimodal user turns ship content as a list of block dicts."""
    stores = InMemoryRuntimeStores()
    blocks: list[dict[str, object]] = [
        {"type": "text", "text": "look"},
        {"type": "image", "url": "u"},
    ]
    await stores.messages.append_user(_AGENT, _THREAD, blocks)

    log = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert log[0].content == blocks


# --- append_assistant ---


@pytest.mark.asyncio
async def test_append_assistant_without_tool_calls() -> None:
    """Plain text-only assistant turn (no tools used)."""
    stores = InMemoryRuntimeStores()
    msg = Message(role="assistant", content="done")
    await stores.messages.append_assistant(_AGENT, _THREAD, "turn_1", msg)

    log = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert log[0].role == "assistant"
    assert log[0].content == "done"


# --- append_assistant_with_tool_calls — atomicity ---


@pytest.mark.asyncio
async def test_atomic_append_writes_both_message_and_tool_calls() -> None:
    """The transcript invariant: the assistant message claiming a
    tool_call_id MUST land alongside the corresponding ToolCallRecord.
    A non-atomic write that committed only one would leave the next
    LLM call 400-ing."""
    stores = InMemoryRuntimeStores()
    msg = Message(
        role="assistant",
        content=None,
        tool_calls=[ToolCall(id="tc_1", function=ToolCallFunction(name="echo", arguments="{}"))],
    )
    record = ToolCallRecord(
        id="tc_1",
        group_id="g_1",
        run_id="run_1",
        agent_id=_AGENT,
        thread_id=_THREAD,
        turn_id="turn_1",
        turn_index=1,
        name="echo",
        args={},
    )

    await stores.messages.append_assistant_with_tool_calls(
        _AGENT, _THREAD, "turn_1", msg, [record]
    )

    log = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert len(log) == 1
    assert log[0].tool_calls is not None
    assert log[0].tool_calls[0].id == "tc_1"

    stored_call = await stores.tool_calls.get("tc_1")
    assert stored_call.id == "tc_1"
    assert stored_call.status == ToolCallStatus.REQUESTED


@pytest.mark.asyncio
async def test_atomic_append_with_multiple_tool_calls() -> None:
    """Parallel tool calls — all records land in one shot."""
    stores = InMemoryRuntimeStores()
    msg = Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(id=f"tc_{i}", function=ToolCallFunction(name="echo", arguments="{}"))
            for i in range(3)
        ],
    )
    records = [
        ToolCallRecord(
            id=f"tc_{i}",
            group_id="g_1",
            run_id="run_1",
            agent_id=_AGENT,
            thread_id=_THREAD,
            turn_id="turn_1",
            turn_index=1,
            name="echo",
            args={},
        )
        for i in range(3)
    ]

    await stores.messages.append_assistant_with_tool_calls(_AGENT, _THREAD, "turn_1", msg, records)

    for i in range(3):
        rec = await stores.tool_calls.get(f"tc_{i}")
        assert rec.id == f"tc_{i}"


# --- append_tool_result ---


@pytest.mark.asyncio
async def test_append_tool_result_adds_tool_message() -> None:
    stores = InMemoryRuntimeStores()
    await stores.messages.append_tool_result(
        _AGENT, _THREAD, "turn_1", "tc_1", "echo", {"result": "ok"}
    )

    log = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert log[0].role == "tool"
    assert log[0].tool_call_id == "tc_1"
    assert log[0].name == "echo"


@pytest.mark.asyncio
async def test_append_tool_result_is_idempotent_on_tool_call_id() -> None:
    """Continuation can fire twice (e.g. two drivers race on the
    same group). Second call must be a no-op so we don't end up
    with two tool messages for one tool_call. Without this the
    LLM transcript would have duplicate tool_results."""
    stores = InMemoryRuntimeStores()
    await stores.messages.append_tool_result(
        _AGENT, _THREAD, "turn_1", "tc_1", "echo", {"result": "first"}
    )
    await stores.messages.append_tool_result(
        _AGENT, _THREAD, "turn_1", "tc_1", "echo", {"result": "second"}
    )

    log = await stores.messages.list_for_thread(_AGENT, _THREAD)
    tool_messages = [m for m in log if m.role == "tool"]
    assert len(tool_messages) == 1


# --- list_for_thread ---


@pytest.mark.asyncio
async def test_list_for_thread_returns_messages_in_append_order() -> None:
    stores = InMemoryRuntimeStores()
    await stores.messages.append_user(_AGENT, _THREAD, "u1")
    await stores.messages.append_assistant(
        _AGENT, _THREAD, "turn", Message(role="assistant", content="a1")
    )
    await stores.messages.append_user(_AGENT, _THREAD, "u2")

    log = await stores.messages.list_for_thread(_AGENT, _THREAD)

    assert [m.content for m in log] == ["u1", "a1", "u2"]


@pytest.mark.asyncio
async def test_list_for_thread_is_isolated_per_thread() -> None:
    stores = InMemoryRuntimeStores()
    await stores.messages.append_user(_AGENT, "t1", "x")
    await stores.messages.append_user(_AGENT, "t2", "y")

    log_t1 = await stores.messages.list_for_thread(_AGENT, "t1")
    log_t2 = await stores.messages.list_for_thread(_AGENT, "t2")

    assert [m.content for m in log_t1] == ["x"]
    assert [m.content for m in log_t2] == ["y"]


@pytest.mark.asyncio
async def test_list_for_thread_is_isolated_per_agent() -> None:
    stores = InMemoryRuntimeStores()
    await stores.messages.append_user("a1", _THREAD, "x")
    await stores.messages.append_user("a2", _THREAD, "y")

    log_a1 = await stores.messages.list_for_thread("a1", _THREAD)
    assert [m.content for m in log_a1] == ["x"]


@pytest.mark.asyncio
async def test_list_for_thread_returns_empty_for_unknown_thread() -> None:
    """No KeyError — just empty. Used by orchestrator on cold start."""
    stores = InMemoryRuntimeStores()

    log = await stores.messages.list_for_thread(_AGENT, "never_used")

    assert log == []
