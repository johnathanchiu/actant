"""Activities that open, advance, and finalize agent runs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from typing import cast

from temporalio import activity
from temporalio.exceptions import ApplicationError

from actant.agents import AgentDefinition
from actant.blocks import BLOCKS, Block, CompactionBlock
from actant.assets import AssetContext, prepare_messages
from actant.core import JSONObject, new_id
from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall as LLMToolCall
from actant.runtime.compaction import (
    SUMMARY_MAX_OUTPUT_TOKENS,
    ModelView,
    build_view,
    compaction_request,
    count_images,
    crossed_limits,
    kept_ids,
    measure_request,
    pending_start,
)
from actant.runtime.events.runtime import RuntimeEvents
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


class _Phases:
    """Where a model turn's time before its call went: one line per turn, so a slow start
    (the thread's stores, the turn gate, its pictures, its listeners) names its cause."""

    def __init__(self) -> None:
        self.started = self.last = time.perf_counter()
        self.seconds: dict[str, float] = {}

    def end(self, phase: str) -> None:
        now = time.perf_counter()
        self.seconds[phase] = self.seconds.get(phase, 0.0) + now - self.last
        self.last = now

    def log(self, agent_id: str, turn_index: int) -> None:
        logger.info(
            "turn prepared agent=%s turn=%d seconds=%.3f %s",
            agent_id,
            turn_index,
            self.last - self.started,
            " ".join(f"{phase}={s:.3f}" for phase, s in self.seconds.items()),
        )


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
        # Idempotent: a retried start finds the run its first attempt created.
        await self.context.stores.runs.create(
            payload.agent_id,
            payload.thread_id,
            run_id=payload.run_id,
            max_turns=max_turns,
        )
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
        phases = _Phases()
        agent = await self.context.agent(payload.agent_id, payload.thread_id)
        phases.end("resolve")
        thread = await self.context.stores.threads.get_or_create(
            payload.agent_id, payload.thread_id
        )
        run = await self.context.stores.runs.get(payload.run_id)
        phases.end("load")
        events = self.context.events(
            thread, run_id=payload.run_id, turn_id=payload.turn_id, turn_index=payload.turn_index
        )

        compaction = payload.context_compaction
        # A compacting thread measures before storing new messages, so that when
        # it compacts they land after the compaction row.
        measuring = compaction is not None and not payload.compacted
        inbound = [
            Message(
                role="user",
                content=list(BLOCKS.validate_python(msg.content))
                if isinstance(msg.content, list)
                else msg.content,
                tag=msg.tag,
            )
            for msg in payload.new_messages
        ]
        if not measuring:
            await self._append_inbound(payload, inbound, events)
            phases.end("inbound")

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
            phases.end("gate")

        model_view: ModelView | None = None
        if compaction is None:
            # Exactly as before compaction existed; a compaction row, left by an
            # earlier setting, is never sent.
            history = await self.context.stores.messages.list_for_thread(
                payload.agent_id, payload.thread_id
            )
            view = [m for m in history if m.kind == "message"]
        else:
            model_view = build_view(
                await self.context.stores.messages.list_for_model(
                    payload.agent_id, payload.thread_id
                )
            )
            view = model_view.messages + (inbound if measuring else [])
        phases.end("history")
        messages = await self._prepare(
            view, payload.agent_id, payload.thread_id, payload.run_id, payload.turn_id
        )
        phases.end("assets")
        if compaction is not None and model_view is not None and measuring:
            trigger = _compaction_trigger(agent, compaction, model_view, view, messages)
            if trigger is not None:
                # Nothing is sent or stored: the workflow compacts, then runs this
                # turn again with the same new messages.
                return TurnResult(
                    turn_id=payload.turn_id,
                    turn_index=payload.turn_index,
                    compaction=trigger,
                )
            await self._append_inbound(payload, inbound, events)
            phases.end("inbound")
        context = TurnContext(
            agent=agent,
            system_prompt=agent.persona,
            messages=messages,
            thread_id=payload.thread_id,
            turn_id=payload.turn_id,
            turn_index=payload.turn_index,
        )

        await events.on_turn_start(payload.turn_index, payload.turn_id)
        phases.end("events")
        phases.log(payload.agent_id, payload.turn_index)
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
        """Summarize the model's view before the carried turn, on the agent's own model
        with no tools, and append the compaction row. No stored message changes.

        The summary call carries at most what the last request carried, so it is
        under the image limit, and its output is capped to the window's margin. If
        it is rejected anyway the activity fails, non-retryable, and the thread
        keeps its full context.
        """
        trigger = payload.trigger
        agent = await self.context.agent(payload.agent_id, payload.thread_id)
        view = build_view(
            await self.context.stores.messages.list_for_model(payload.agent_id, payload.thread_id)
        )
        cut = view.ids.index(trigger.carried[0]) if trigger.carried else len(view.messages)
        request = await self._prepare(
            compaction_request(view.messages[:cut], self.context.compaction_instructions),
            payload.agent_id,
            payload.thread_id,
            payload.run_id,
            payload.turn_id,
        )
        window, _ = _limits(agent)
        max_output = None
        if window is not None:
            used = measure_request(agent.persona, request, usage_from=view.usage_from).tokens
            max_output = min(SUMMARY_MAX_OUTPUT_TOKENS, window - used)
            if max_output < 1:
                raise ApplicationError(
                    f"no room in the {window}-token window for a summary of {used} tokens",
                    non_retryable=True,
                )
        try:
            response = await agent.llm.complete(
                agent.persona, request, [], None, max_output_tokens=max_output
            )
        except Exception as exc:
            # Never fall back to dropping context: the run fails and says why.
            raise ApplicationError(f"the summary call failed: {exc}", non_retryable=True) from exc
        summary = response.content.strip() if isinstance(response.content, str) else ""
        if not summary:
            raise ApplicationError("the summary call returned no summary", non_retryable=True)

        # The one full-history read: the latest message of each kept tag may be
        # older than any compaction since.
        history = await self.context.stores.messages.list_for_thread(
            payload.agent_id, payload.thread_id
        )
        ids = [m.id for m in history]
        before = ids.index(trigger.carried[0]) if trigger.carried else len(history)
        chosen = {*kept_ids(history, before, payload.config.keep), *trigger.carried}
        kept = sorted(chosen, key=ids.index)
        block = CompactionBlock(
            summary=summary,
            kept=kept,
            reason=trigger.reason,
            tokens_before=trigger.tokens,
            images_before=trigger.images,
            tokens_after=0,
            images_after=0,
        )
        row = Message(role="user", content=[block], kind="compaction")
        fresh = build_view([*(m for m in history if m.id in chosen), row])
        prepared = await self._prepare(
            fresh.messages, payload.agent_id, payload.thread_id, payload.run_id, payload.turn_id
        )
        after = measure_request(agent.persona, fresh.messages, usage_from=fresh.usage_from)
        block = block.model_copy(
            update={"tokens_after": after.tokens, "images_after": count_images(prepared)}
        )
        record = await self.context.stores.messages.append_compaction(
            payload.agent_id, payload.thread_id, block
        )
        thread = await self.context.stores.threads.get_or_create(
            payload.agent_id, payload.thread_id
        )
        events = self.context.events(
            thread, run_id=payload.run_id, turn_id=payload.turn_id, turn_index=payload.turn_index
        )
        await events.on_context_compacted(record.id, block)
        return CompactionOutcome(record.id, block.tokens_after, block.images_after)

    async def _append_inbound(
        self, payload: RunTurnInput, inbound: list[Message], events: RuntimeEvents
    ) -> None:
        for message in inbound:
            content = cast(str | list[Block], message.content)
            await self.context.stores.messages.append_user(
                payload.agent_id, payload.thread_id, content, tag=message.tag
            )
            await events.on_user_message(content)

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


def _limits(agent: AgentDefinition) -> tuple[int | None, int | None]:
    return (
        _provider_limit(agent, "context_window_tokens"),
        _provider_limit(agent, "max_images_per_request"),
    )


def _compaction_trigger(
    agent: AgentDefinition,
    config: CompactionConfig,
    model_view: ModelView,
    request: list[Message],
    prepared: list[Message],
) -> CompactionTrigger | None:
    """Whether the request about to be sent crosses a limit, and what to carry if so."""
    window, max_images = _limits(agent)
    if window is None and max_images is None:
        return None
    measure = measure_request(agent.persona, request, usage_from=model_view.usage_from)
    measure = replace(measure, images=count_images(prepared))
    reason = crossed_limits(
        measure,
        context_window_tokens=window,
        max_images_per_request=max_images,
        threshold=config.threshold,
    )
    if reason is None:
        return None
    floor = 1 if model_view.compaction is not None else 0
    start = pending_start(model_view.messages, floor=floor)
    if start <= floor:
        # Nothing stored before the pending messages but the last summary: a new
        # one could replace nothing, and dropping content is not allowed.
        logger.warning(
            "actant.compaction.nothing_to_summarize agent=%s reason=%s tokens=%s images=%s",
            agent.id,
            reason,
            measure.tokens,
            measure.images,
        )
        return None
    carried = [i for i in model_view.ids[start:] if i is not None]
    return CompactionTrigger(
        reason=reason, tokens=measure.tokens, images=measure.images, carried=carried
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
