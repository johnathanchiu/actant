"""Activities that repair and finalize agent-thread state."""

from __future__ import annotations

from temporalio import activity

from actant.runtime.cancellation import record_cancelled
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import ActivityName, ApplyThreadCancellationInput


class ThreadActivities:
    """Thread-level lifecycle activities."""

    def __init__(self, context: ActivityContext) -> None:
        self.context = context

    @activity.defn(name=ActivityName.APPLY_THREAD_CANCELLATION)
    async def apply_thread_cancellation(self, payload: ApplyThreadCancellationInput) -> None:
        """Idempotently repair projections and transcripts after cancellation."""
        await record_cancelled(self.context.stores, payload.agent_id, payload.thread_id)
        if self.context.cancel_children is not None:
            await self.context.cancel_children(payload.thread_id)
        if self.context.sandboxes is not None:
            # Only the thread's own sandbox: one it names belongs to whoever opened it.
            thread = await self.context.stores.threads.get_or_create(
                payload.agent_id, payload.thread_id
            )
            if thread.sandbox_id is None:
                await self.context.sandboxes.close(payload.thread_id)
