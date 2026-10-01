"""What a cancelled thread leaves in the stores.

One implementation, called by ``AgentRuntime.cancel_thread`` as the cancel is
sent and again by the workflow's cancellation activity once the run has
stopped. Both are needed: the first makes a cancel visible at once, even for a
thread whose workflow had already closed; the second repairs whatever a turn
still in flight wrote after it. Idempotent, so running it twice is harmless.
"""

from __future__ import annotations

from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.types.threads import ThreadStatus
from actant.tools.calls import ToolCallStatus

#: The result an open tool call is closed with: no tool ran to completion.
CANCELLED_RESULT = {"status": "cancelled", "reason": "session_cancelled"}


async def record_cancelled(stores: RuntimeStores, agent_id: str, thread_id: str) -> None:
    """Close every open tool call, pair every tool call in the transcript with a
    result (providers reject a call without one), and mark the thread cancelled."""
    for record in await stores.tool_calls.get_open_for_thread(agent_id, thread_id):
        await stores.tool_calls.update_status(
            record.id, ToolCallStatus.COMPLETED, result=dict(CANCELLED_RESULT)
        )

    tool_call_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for message in await stores.messages.list_for_thread(agent_id, thread_id):
        if message.role == "assistant" and message.tool_calls:
            tool_call_ids.update(call.id for call in message.tool_calls)
        elif message.role == "tool" and message.tool_call_id is not None:
            tool_result_ids.add(message.tool_call_id)

    for tool_call_id in tool_call_ids - tool_result_ids:
        try:
            record = await stores.tool_calls.get(tool_call_id)
        except KeyError:
            continue
        result = record.result if isinstance(record.result, dict) else dict(CANCELLED_RESULT)
        await stores.messages.append_tool_result(
            record.agent_id, record.thread_id, record.turn_id, record.id, record.name, result
        )

    thread = await stores.threads.get_or_create(agent_id, thread_id)
    thread.active_run_id = None
    thread.status = ThreadStatus.CANCELLED
    await stores.threads.update(thread)
