"""Activities that open, advance, and finalize agent runs."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import cast

from temporalio import activity
from temporalio.exceptions import ApplicationError

from actant.agents import AgentDefinition
from actant.blocks import BLOCKS, Block
from actant.assets import AssetContext, prepare_messages
from actant.core import JSONObject, new_id
from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall as LLMToolCall
from actant.runtime.compaction import (
    Compaction,
    CompactionRecord,
    compaction_request,
    context_messages,
    count_images,
    crossed_limits,
    fresh_prefix,
    measure_request,
    pending_start,
    pinned_item,
    usage_start,
)
from actant.runtime.completion import RunCompletion
from actant.runtime.gate import TurnStart
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import (
    ActivityName,
    CompactContextInput,
    CompactionOutcome,
    CompactionTrigger,
    CompactionConfig,
    FinalizeRunInput,
    RunOutcome,
    RunTurnInput,
    StartRunInput,
    StartedRun,
    ToolCallSpec,
    TurnResult,
)
from actant.runtime.types.context import TurnContext
from actant.runtime.types.threads import RunStatus, ThreadStatus
from actant.tools.base import MetadataKey
from actant.tools.calls import ToolCallRecord, ToolCallStatus

FINISH_REMINDER = (
    "<finish_required>\n"
    "This run ends only when you call `finish` with your summary and deliverable paths, "
    "or when a tool result is terminal. Call `finish` now, or keep working with your tools.\n"
    "</finish_required>"
)
STOPPED_WITHOUT_FINISHING = "stopped without finishing"

logger = logging.getLogger(__name__)


class RunActivities:
    """Activities for the lifecycle and LLM turns of an agent run."""

    def __init__(self, context: ActivityContext) -> None:
        self.context = context

    @activity.defn(name=ActivityName.START_RUN)
    async def start_run(self, payload: StartRunInput) -> StartedRun:
        """Create the run projection, and say how many turns this thread has had.

        The count is returned because the workflow cannot keep it any more. A
        thread now ends when its work is done and restarts on the next
        message, so anything held only in workflow memory resets -- and turn
        numbering that restarts at one repeats numbers within a thread. The
        store has the real count, and this activity already runs once before
        every run, so reading it here costs nothing extra.
        """
        resolution_error = None
        try:
            agent = await self.context.agent(payload.agent_id, payload.thread_id)
            max_turns = max(1, agent.max_turns_per_thread)
        except ApplicationError as error:
            if not error.non_retryable:
                raise
            # An invalid definition is a terminal run, not an unprojected failed workflow.
            resolution_error = str(error)
            max_turns = max(1, payload.max_turns or 1)
        if payload.max_turns is not None:
            max_turns = min(max_turns, max(1, payload.max_turns))
        try:
            await self.context.stores.runs.create(
                payload.agent_id,
                payload.thread_id,
                run_id=payload.run_id,
                max_turns=max_turns,
            )
        except Exception:
            try:
                await self.context.stores.runs.get(payload.run_id)
            except Exception:
                raise
        thread = await self.context.stores.threads.get_or_create(
            payload.agent_id, payload.thread_id
        )
        thread.active_run_id = payload.run_id
        thread.status = ThreadStatus.ACTIVE
        if payload.parent_thread_id and thread.parent_thread_id is None:
            thread.parent_thread_id = payload.parent_thread_id
        await self.context.stores.threads.update(thread)
        run = await self.context.stores.runs.get(payload.run_id)
        return StartedRun(thread.turn_count, run.max_turns, resolution_error)

    @activity.defn(name=ActivityName.RUN_TURN)
    async def run_turn(self, payload: RunTurnInput) -> TurnResult:
        """Invoke the LLM once and atomically persist its assistant turn."""
        agent = await self.context.agent(payload.agent_id, payload.thread_id)
        thread = await self.context.stores.threads.get_or_create(
            payload.agent_id, payload.thread_id
        )
        run = await self.context.stores.runs.get(payload.run_id)
        events = self.context.events(
            thread, run_id=payload.run_id, turn_id=payload.turn_id, turn_index=payload.turn_index
        )

        for msg in payload.new_messages:
            content = (
                BLOCKS.validate_python(msg.content)
                if isinstance(msg.content, list)
                else msg.content
            )
            await self.context.stores.messages.append_user(
                payload.agent_id, payload.thread_id, content
            )
            await events.on_user_message(content)

        # A turn run again after compaction was already admitted.
        if self.context.turn_gate is not None and not payload.compacted:
            reason = await self.context.turn_gate(
                TurnStart(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                    run_id=payload.run_id,
                    turn_id=payload.turn_id,
                    turn_index=payload.turn_index,
                )
            )
            if reason is not None:
                return TurnResult(
                    turn_id=payload.turn_id,
                    turn_index=payload.turn_index,
                    stop_reason=reason,
                )

        stored = await self.context.stores.messages.list_for_thread(
            payload.agent_id, payload.thread_id
        )
        # Only a thread that compacts reads its boundary; any other builds
        # its request from the whole transcript, exactly as before.
        compaction = payload.context_compaction
        record = (
            await self.context.stores.compactions.latest(payload.agent_id, payload.thread_id)
            if compaction is not None
            else None
        )
        view = context_messages(stored, record)
        messages = await self._prepare(
            view, payload.agent_id, payload.thread_id, payload.run_id, payload.turn_id
        )
        if compaction is not None and not payload.compacted:
            trigger = _compaction_trigger(agent, compaction, record, view, messages)
            if trigger is not None:
                # Nothing is sent: the workflow compacts, then runs this turn again.
                return TurnResult(
                    turn_id=payload.turn_id,
                    turn_index=payload.turn_index,
                    compaction=trigger,
                )
        context = TurnContext(
            agent=agent,
            system_prompt=agent.persona,
            messages=messages,
            thread_id=payload.thread_id,
            turn_id=payload.turn_id,
            turn_index=payload.turn_index,
        )

        await events.on_turn_start(payload.turn_index, payload.turn_id)
        try:
            assistant = await agent.complete(
                context.messages,
                events,
                final_turn=run.turn_count + 1 >= run.max_turns,
            )
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
        await self.context.stores.messages.append_assistant_with_tool_calls(
            payload.agent_id,
            payload.thread_id,
            payload.turn_id,
            assistant,
            records,
        )

        await events.on_assistant_message(assistant)
        for record in records:
            await events.on_tool_call(record.id, record.name, record.args)

        thread.turn_count += 1
        run.turn_count += 1
        run.status = RunStatus.ACTIVE
        await self.context.stores.threads.update(thread)
        await self.context.stores.runs.update(run)

        if not records and agent.completion == "terminal":
            # A task agent does not end by silence. Once: remind it, persisted
            # so the transcript (and any replay) shows the nudge. Twice: the
            # run ends, and the reason says why.
            if payload.text_only_turns == 0:
                await self.context.stores.messages.append_user(
                    payload.agent_id, payload.thread_id, FINISH_REMINDER
                )
                await events.on_user_message(FINISH_REMINDER)
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

    @activity.defn(name=ActivityName.COMPACT_CONTEXT)
    async def compact_context(self, payload: CompactContextInput) -> CompactionOutcome:
        """Summarize the context before a boundary on the agent's own model, with no tools.

        Records the boundary, the summary and the app's pinned blocks; no
        stored message changes. From then on the model's request is the system
        prompt, the summary, the pinned blocks, and the messages from the
        boundary on.
        """
        trigger = payload.trigger
        agent = await self.context.agent(payload.agent_id, payload.thread_id)
        previous = await self.context.stores.compactions.latest(
            payload.agent_id, payload.thread_id
        )
        if (
            previous is not None
            and previous.turn_id == payload.turn_id
            and previous.boundary == trigger.boundary
        ):
            # Already recorded by an earlier attempt of this activity.
            return CompactionOutcome(previous.id, previous.tokens_after, previous.images_after)

        stored = await self.context.stores.messages.list_for_thread(
            payload.agent_id, payload.thread_id
        )
        replaced = context_messages(stored, previous, end=trigger.boundary)
        request = await self._prepare(
            compaction_request(replaced, self.context.compaction_instructions),
            payload.agent_id,
            payload.thread_id,
            payload.run_id,
            payload.turn_id,
        )
        try:
            response = await agent.llm.complete(agent.persona, request, [], None)
        except StreamCancelled as exc:
            raise ApplicationError("compaction cancelled", non_retryable=True) from exc
        summary = response.content.strip() if isinstance(response.content, str) else ""
        if not summary:
            # Carrying on without a summary would drop the context silently.
            raise ApplicationError("the compaction turn returned no summary", non_retryable=True)

        pinned = await self._pinned(
            payload,
            Compaction(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=payload.run_id,
                reason=trigger.reason,
                summary=summary,
            ),
        )
        record = CompactionRecord(
            id=new_id("compaction"),
            agent_id=payload.agent_id,
            thread_id=payload.thread_id,
            run_id=payload.run_id,
            turn_id=payload.turn_id,
            boundary=trigger.boundary,
            reason=trigger.reason,
            summary=summary,
            pinned=pinned,
            tokens_before=trigger.tokens,
            images_before=trigger.images,
            tokens_after=0,
            images_after=0,
            context_window_tokens=trigger.context_window_tokens,
            max_images_per_request=trigger.max_images_per_request,
        )
        fresh = context_messages(stored, record)
        prepared = await self._prepare(
            fresh, payload.agent_id, payload.thread_id, payload.run_id, payload.turn_id
        )
        after = measure_request(agent.persona, fresh, usage_from=usage_start(record))
        record = replace(record, tokens_after=after.tokens, images_after=count_images(prepared))
        await self.context.stores.compactions.append(record)
        thread = await self.context.stores.threads.get_or_create(
            payload.agent_id, payload.thread_id
        )
        events = self.context.events(
            thread, run_id=payload.run_id, turn_id=payload.turn_id, turn_index=payload.turn_index
        )
        await events.on_context_compacted(record)
        return CompactionOutcome(record.id, record.tokens_after, record.images_after)

    async def _pinned(self, payload: CompactContextInput, compaction: Compaction) -> list[Block]:
        """What follows the summary, verbatim: ``pin`` in order, every other
        pinned note by key, then the ``on_compact`` hook's blocks."""
        notes = await self.context.stores.pinned_notes.list_for_thread(
            payload.agent_id, payload.thread_id
        )
        pinned: list[Block] = []
        for name in payload.config.pin:
            provider = self.context.pin_providers.get(name)
            if provider is not None:
                pinned.extend(
                    pinned_item(name, BLOCKS.validate_python(await provider(compaction)))
                )
            elif name in notes:
                pinned.extend(pinned_item(name, notes.pop(name)))
            else:
                logger.warning(
                    "actant.compaction.pin_missing agent=%s thread=%s name=%s: no provider "
                    "is registered and no note is pinned under this name",
                    payload.agent_id,
                    payload.thread_id,
                    name,
                )
        for key, blocks in notes.items():
            pinned.extend(pinned_item(key, blocks))
        if self.context.on_compact is not None:
            pinned.extend(BLOCKS.validate_python(await self.context.on_compact(compaction)))
        else:
            # State that must survive compaction cannot rest on the summary.
            logger.warning(
                "actant.compaction.no_hook agent=%s thread=%s pinned_blocks=%s: no "
                "on_compact hook, so %s",
                payload.agent_id,
                payload.thread_id,
                len(pinned),
                "only pin providers and pinned notes follow the summary"
                if pinned
                else "nothing is pinned after the summary",
            )
        return pinned

    async def _prepare(
        self,
        messages: list[Message],
        agent_id: str,
        thread_id: str,
        run_id: str,
        turn_id: str,
    ) -> list[Message]:
        """The app's preprocessor, then assets resolved for one model call."""
        if self.context.message_preprocessor is not None:
            messages = await self.context.message_preprocessor(messages)
        return await prepare_messages(
            messages,
            self.context.assets,
            AssetContext(agent_id, thread_id, run_id, turn_id),
        )

    @activity.defn(name=ActivityName.FINALIZE_RUN)
    async def finalize_run(self, payload: FinalizeRunInput) -> None:
        """Close a run, repair cancelled calls, and notify observers."""
        if payload.outcome in {RunOutcome.CANCELLED.value, RunOutcome.FAILED.value}:
            records = await self.context.stores.tool_calls.get_by_run(payload.run_id)
            terminal = {ToolCallStatus.COMPLETED, ToolCallStatus.BLOCKED, ToolCallStatus.FAILED}
            for record in sorted(records, key=lambda item: (item.turn_index, item.id)):
                result = record.result
                if record.status not in terminal:
                    cancelled = payload.outcome == RunOutcome.CANCELLED.value
                    result = (
                        {"status": "cancelled", "reason": "session_cancelled"}
                        if cancelled
                        else {
                            "error": f"run failed; tool outcome uncertain: {payload.stop_reason or 'unknown'}"
                        }
                    )
                    await self.context.stores.tool_calls.update_status(
                        record.id,
                        ToolCallStatus.COMPLETED if cancelled else ToolCallStatus.FAILED,
                        result=result,
                    )
                # A surviving timed-out activity may have won the terminal transition.
                record = await self.context.stores.tool_calls.get(record.id)
                result = record.result
                # Idempotent store operation also repairs a turn persisted before its
                # activity died, and a tool group whose finalization failed.
                await self.context.stores.messages.append_tool_result(
                    record.agent_id,
                    record.thread_id,
                    record.turn_id,
                    record.id,
                    record.name,
                    result if result is not None else {"error": "No result"},
                )

        await self.context.stores.runs.finish(
            payload.run_id, _run_status(payload.outcome), stop_reason=payload.stop_reason
        )
        thread = await self.context.stores.threads.get_or_create(
            payload.agent_id, payload.thread_id
        )
        thread.active_run_id = None
        thread.status = _thread_status(payload.outcome)
        await self.context.stores.threads.update(thread)

        # Deliverables are read back from the run's tool calls, so a retried
        # finalization reports the same list.
        artifacts: list[dict[str, object]] = []
        for record in await self.context.stores.tool_calls.get_by_run(payload.run_id):
            raw = record.result if isinstance(record.result, dict) else {}
            metadata = raw.get("metadata")
            refs = metadata.get(MetadataKey.ARTIFACTS) if isinstance(metadata, dict) else None
            if isinstance(refs, list):
                artifacts.extend(ref for ref in refs if isinstance(ref, dict))

        if self.context.run_completion_handler is not None:
            await self.context.run_completion_handler(
                RunCompletion(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                    run_id=payload.run_id,
                    outcome=payload.outcome,
                    stop_reason=payload.stop_reason,
                    artifacts=tuple(artifacts),
                )
            )
        await self.context.events(thread, run_id=payload.run_id).on_complete(
            success=payload.outcome == RunOutcome.COMPLETED.value,
            reason=payload.stop_reason or payload.outcome,
            message="",
        )


def _provider_limit(agent: AgentDefinition, name: str) -> int | None:
    """A limit the model client declares, when it does (``LLMClient`` does not require it)."""
    value = getattr(agent.llm, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _compaction_trigger(
    agent: AgentDefinition,
    config: CompactionConfig,
    record: CompactionRecord | None,
    view: list[Message],
    prepared: list[Message],
) -> CompactionTrigger | None:
    """Whether the request about to be sent crosses a limit, and where to cut if so."""
    window = config.context_window_tokens or _provider_limit(agent, "context_window_tokens")
    max_images = config.max_images_per_request or _provider_limit(agent, "max_images_per_request")
    if window is None and max_images is None:
        return None
    measure = measure_request(agent.persona, view, usage_from=usage_start(record))
    measure = replace(measure, images=count_images(prepared))
    reason = crossed_limits(
        measure,
        context_window_tokens=window,
        max_images_per_request=max_images,
        threshold=config.threshold,
    )
    if reason is None:
        return None
    first = len(fresh_prefix(record)) if record is not None else 0
    start = pending_start(view, floor=first)
    if start <= first:
        # Everything in the request is still pending (one huge message, or
        # results of the call right after the last compaction). A summary
        # could replace nothing, and dropping content is not allowed.
        logger.warning(
            "actant.compaction.nothing_to_summarize agent=%s reason=%s tokens=%s images=%s",
            agent.id,
            reason,
            measure.tokens,
            measure.images,
        )
        return None
    return CompactionTrigger(
        reason=reason,
        tokens=measure.tokens,
        images=measure.images,
        boundary=(record.boundary if record is not None else 0) + start - first,
        context_window_tokens=window,
        max_images_per_request=max_images,
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
