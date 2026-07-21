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


class RunActivities(ActivityContext):
    """Activities for the lifecycle and LLM turns of an agent run."""

    @activity.defn(name=ActivityName.START_RUN)
    async def start_run(self, payload: StartRunInput) -> None:
        """Create the run projection, idempotently on Temporal retry."""
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
        await self.stores.threads.update(thread)

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

        await self.stores.runs.finish(payload.run_id, _run_status(payload.outcome))
        thread = await self.stores.threads.get_or_create(payload.agent_id, payload.thread_id)
        thread.active_run_id = None
        thread.status = _thread_status(payload.outcome)
        await self.stores.threads.update(thread)

        if self.run_completion_handler is not None:
            await self.run_completion_handler(
                RunCompletion(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                    run_id=payload.run_id,
                    outcome=payload.outcome,
                )
            )
        await self._hooks(thread).on_complete(
            success=payload.outcome == RunOutcome.COMPLETED.value,
            reason=payload.outcome,
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
