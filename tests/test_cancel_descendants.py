"""Cancellation follows stored lineage, not the worker that spawned a child."""

from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode

from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.runtime import AgentRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities.threads import ThreadActivities
from actant.runtime.temporal.types import ApplyThreadCancellationInput
from actant.runtime.types.threads import ThreadStatus
from actant.tools.calls import ToolCallRecord, ToolCallStatus


@pytest.mark.parametrize("cleanup", [False, True])
async def test_cancellation_reaches_descendants_on_a_fresh_runtime(cleanup: bool) -> None:
    stores = InMemoryRuntimeStores()
    for name, parent, status in [
        ("root", None, ThreadStatus.ACTIVE),
        ("child", "root", ThreadStatus.ACTIVE),
        ("finished", "root", ThreadStatus.IDLE),
        ("grandchild", "finished", ThreadStatus.ACTIVE),
        ("unrelated", None, ThreadStatus.ACTIVE),
    ]:
        thread = await stores.threads.get_or_create(name, name)
        thread.parent_thread_id = parent
        thread.status = status
        await stores.threads.update(thread)
    handles = {name: Mock(cancel=AsyncMock()) for name in ["root", "child", "grandchild"]}
    # A closed parent must not hide its still-running children.
    handles["root"].cancel.side_effect = RPCError("closed", RPCStatusCode.NOT_FOUND, b"")
    client = Mock()
    runtime = AgentRuntime(client=cast(Client, client), stores=stores)
    client.get_workflow_handle.side_effect = lambda wid: handles[wid.rsplit("-", 1)[-1]]
    if cleanup:
        await ThreadActivities(runtime._context).apply_thread_cancellation(
            ApplyThreadCancellationInput(agent_id="root", thread_id="root")
        )
    else:
        await runtime.cancel_thread("root", "root")
    handles["child"].cancel.assert_awaited_once()
    handles["grandchild"].cancel.assert_awaited_once()
    assert (await stores.threads.get("finished", "finished")).status == ThreadStatus.IDLE
    assert (await stores.threads.get("unrelated", "unrelated")).status == ThreadStatus.ACTIVE
    children = await stores.threads.list_children("root")
    children[0].parent_thread_id = None
    assert len(await stores.threads.list_children("root")) == 2


async def test_cancelling_a_closed_thread_records_it_cancelled_with_its_calls_closed() -> None:
    stores = InMemoryRuntimeStores()
    thread = await stores.threads.get_or_create("a", "t")
    thread.status = ThreadStatus.ACTIVE
    thread.active_run_id = "r"
    await stores.threads.update(thread)
    call = ToolCallRecord(
        id="c",
        group_id="g",
        run_id="r",
        agent_id="a",
        thread_id="t",
        turn_id="turn",
        turn_index=1,
        name="ask",
        args={},
        status=ToolCallStatus.WAITING,
    )
    assistant = Message(
        role="assistant",
        content="asking",
        tool_calls=[ToolCall(id="c", function=ToolCallFunction(name="ask", arguments="{}"))],
    )
    await stores.messages.append_assistant_with_tool_calls("a", "t", "turn", assistant, [call])
    client = Mock()
    client.get_workflow_handle.return_value = Mock(
        cancel=AsyncMock(side_effect=RPCError("closed", RPCStatusCode.NOT_FOUND, b""))
    )
    runtime = AgentRuntime(client=cast(Client, client), stores=stores)

    await runtime.cancel_thread("a", "t")

    thread = await stores.threads.get("a", "t")
    assert thread.status is ThreadStatus.CANCELLED and thread.active_run_id is None
    assert (await stores.tool_calls.get("c")).status is ToolCallStatus.COMPLETED
    messages = await stores.messages.list_for_thread("a", "t")
    assert [m.tool_call_id for m in messages if m.role == "tool"] == ["c"]
