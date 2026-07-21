"""Activities that repair and finalize agent-thread state."""

from __future__ import annotations

from temporalio import activity

from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import ActivityName, ApplyThreadCancellationInput
from actant.runtime.types.threads import ThreadStatus
from actant.tools.calls import ToolCallStatus


class ThreadActivities(ActivityContext):
    """Thread-level lifecycle activities."""

    @activity.defn(name=ActivityName.APPLY_THREAD_CANCELLATION)
    async def apply_thread_cancellation(self, payload: ApplyThreadCancellationInput) -> None:
        """Idempotently repair projections and transcripts after cancellation."""
        open_records = await self.stores.tool_calls.get_open_for_thread(
            payload.agent_id, payload.thread_id
        )
        for record in open_records:
            await self.stores.tool_calls.update_status(
                record.id,
                ToolCallStatus.COMPLETED,
                result={"status": "cancelled", "reason": "session_cancelled"},
            )

        messages = await self.stores.messages.list_for_thread(payload.agent_id, payload.thread_id)
        tool_call_ids: set[str] = set()
        tool_result_ids: set[str] = set()
        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                tool_call_ids.update(call.id for call in message.tool_calls)
            elif message.role == "tool" and message.tool_call_id is not None:
                tool_result_ids.add(message.tool_call_id)

        for tool_call_id in tool_call_ids - tool_result_ids:
            try:
                record = await self.stores.tool_calls.get(tool_call_id)
            except KeyError:
                continue
            result = (
                record.result
                if isinstance(record.result, dict)
                else {"status": "cancelled", "reason": "session_cancelled"}
            )
            await self.stores.messages.append_tool_result(
                record.agent_id,
                record.thread_id,
                record.turn_id,
                record.id,
                record.name,
                result,
            )

        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        thread.active_run_id = None
        thread.status = ThreadStatus.CANCELLED
        await self.stores.threads.update(thread)
