"""Activities that open, advance, and finalize agent runs."""

from __future__ import annotations

import json
from typing import cast

from temporalio import activity
from temporalio.exceptions import ApplicationError

from actant.core import JSONObject, new_id
from actant.llm.errors import StreamCancelled
from actant.llm.messages import ToolCall as LLMToolCall
from actant.runtime.completion import RunCompletion
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import (
    ActivityName,
    FinalizeRunInput,
    RunOutcome,
    RunTurnInput,
    StartRunInput,
    ToolCallSpec,
    TurnResult,
)
from actant.runtime.types.context import TurnContext
from actant.runtime.types.threads import RunStatus, ThreadStatus
from actant.tools.calls import ToolCallRecord, ToolCallStatus

FINISH_REMINDER = (
    "<finish_required>\n"
    "This run ends only when you call `finish` with your summary and deliverable paths, "
    "or when a tool result is terminal. Call `finish` now, or keep working with your tools.\n"
    "</finish_required>"
)
STOPPED_WITHOUT_FINISHING = "stopped without finishing"


class RunActivities(ActivityContext):
    """Activities for the lifecycle and LLM turns of an agent run."""

    @activity.defn(name=ActivityName.START_RUN)
    async def start_run(self, payload: StartRunInput) -> int:
        """Create the run projection, and say how many turns this thread has had.

        The count is returned because the workflow cannot keep it any more. A
        thread now ends when its work is done and restarts on the next
        message, so anything held only in workflow memory resets -- and turn
        numbering that restarts at one repeats numbers within a thread. The
        store has the real count, and this activity already runs once before
        every run, so reading it here costs nothing extra.
        """
        try:
            await self.stores.runs.create(
                payload.agent_id,
                payload.thread_id,
                run_id=payload.run_id,
                max_turns=payload.max_turns,
            )
        except Exception:
            try:
                await self.stores.runs.get(payload.run_id)
            except Exception:
                raise
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        thread.active_run_id = payload.run_id
        thread.status = ThreadStatus.ACTIVE
        if payload.parent_thread_id and thread.parent_thread_id is None:
            thread.parent_thread_id = payload.parent_thread_id
        await self.stores.threads.update(thread)
        return thread.turn_count

    @activity.defn(name=ActivityName.RUN_TURN)
    async def run_turn(self, payload: RunTurnInput) -> TurnResult:
        """Invoke the LLM once and atomically persist its assistant turn."""
        agent = self._require_agent(payload.agent_id)
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        run = await self.stores.runs.get(payload.run_id)
        hooks = self._hooks(thread)

        for msg in payload.new_messages:
            await self.stores.messages.append_user(
                payload.agent_id, payload.thread_id, msg.content
            )
            await hooks.on_user_message(msg.content)

        messages = await self.stores.messages.list_for_thread(payload.agent_id, payload.thread_id)
        if self.message_preprocessor is not None:
            messages = await self.message_preprocessor(messages)
        context = TurnContext(
            agent=agent,
            system_prompt=agent.persona,
            messages=messages,
            thread_id=payload.thread_id,
            turn_id=payload.turn_id,
            turn_index=payload.turn_index,
        )

        await hooks.on_turn_start(payload.turn_index, payload.turn_id)
        try:
            assistant = await agent.complete(context.messages, self._listener(thread))
        except StreamCancelled as exc:
            raise ApplicationError("turn cancelled", non_retryable=True) from exc

        group_id = new_id("group")[:12]
        records = [
            ToolCallRecord(
                id=tool_call.id,
                group_id=group_id,
                run_id=payload.run_id,
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                turn_id=payload.turn_id,
                turn_index=payload.turn_index,
                name=tool_call.function.name,
                args=_parse_tool_args(tool_call),
            )
            for tool_call in (assistant.tool_calls or [])
        ]
        await self.stores.messages.append_assistant_with_tool_calls(
            payload.agent_id,
            payload.thread_id,
            payload.turn_id,
            assistant,
            records,
        )

        await hooks.on_assistant_message(assistant)
        for record in records:
            await hooks.on_tool_call(record.id, record.name, record.args)

        thread.turn_count += 1
        run.turn_count += 1
        run.status = RunStatus.ACTIVE
        await self.stores.threads.update(thread)
        await self.stores.runs.update(run)

        if not records and agent.completion == "terminal":
            # A task agent does not end by silence. Once: remind it, persisted
            # so the transcript (and any replay) shows the nudge. Twice: the
            # run ends, and the reason says why.
            if payload.text_only_turns == 0:
                await self.stores.messages.append_user(
                    payload.agent_id, payload.thread_id, FINISH_REMINDER
                )
                await hooks.on_user_message(FINISH_REMINDER)
                return TurnResult(
                    turn_id=payload.turn_id, turn_index=payload.turn_index, reminded=True
                )
            return TurnResult(
                turn_id=payload.turn_id,
                turn_index=payload.turn_index,
                stop_reason=STOPPED_WITHOUT_FINISHING,
            )

        return TurnResult(
            turn_id=payload.turn_id,
            turn_index=payload.turn_index,
            tool_calls=[
                ToolCallSpec(
                    id=record.id,
                    group_id=record.group_id,
                    run_id=record.run_id,
                    turn_id=record.turn_id,
                    turn_index=record.turn_index,
                    name=record.name,
                )
                for record in records
            ],
        )

    @activity.defn(name=ActivityName.FINALIZE_RUN)
    async def finalize_run(self, payload: FinalizeRunInput) -> None:
        """Close a run, repair cancelled calls, and notify observers."""
        if payload.outcome == RunOutcome.CANCELLED.value:
            open_records = await self.stores.tool_calls.get_open_for_thread(
                payload.agent_id, payload.thread_id
            )
            for record in open_records:
                await self.stores.tool_calls.update_status(
                    record.id,
                    ToolCallStatus.COMPLETED,
                    result={"status": "cancelled", "reason": "session_cancelled"},
                )

        await self.stores.runs.finish(
            payload.run_id, _run_status(payload.outcome), stop_reason=payload.stop_reason
        )
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        thread.active_run_id = None
        thread.status = _thread_status(payload.outcome)
        await self.stores.threads.update(thread)

        # Deliverables are read back from the run's tool calls, so a retried
        # finalization reports the same list.
        artifacts: list[dict[str, object]] = []
        for record in await self.stores.tool_calls.get_by_run(payload.run_id):
            raw = record.result if isinstance(record.result, dict) else {}
            metadata = raw.get("metadata")
            refs = metadata.get("artifacts") if isinstance(metadata, dict) else None
            if isinstance(refs, list):
                artifacts.extend(ref for ref in refs if isinstance(ref, dict))

        if self.run_completion_handler is not None:
            await self.run_completion_handler(
                RunCompletion(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                    run_id=payload.run_id,
                    outcome=payload.outcome,
                    stop_reason=payload.stop_reason,
                    artifacts=tuple(artifacts),
                )
            )
        await self._hooks(thread).on_complete(
            success=payload.outcome == RunOutcome.COMPLETED.value,
            reason=payload.stop_reason or payload.outcome,
            message="",
        )


def _parse_tool_args(tool_call: LLMToolCall) -> JSONObject:
    try:
        parsed = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return cast(JSONObject, parsed) if isinstance(parsed, dict) else {}


def _run_status(outcome: str) -> RunStatus:
    return {
        RunOutcome.EXHAUSTED.value: RunStatus.EXHAUSTED,
        RunOutcome.FAILED.value: RunStatus.FAILED,
        RunOutcome.CANCELLED.value: RunStatus.CANCELLED,
    }.get(outcome, RunStatus.IDLE)


def _thread_status(outcome: str) -> ThreadStatus:
    return {
        RunOutcome.FAILED.value: ThreadStatus.FAILED,
        RunOutcome.CANCELLED.value: ThreadStatus.CANCELLED,
    }.get(outcome, ThreadStatus.IDLE)
