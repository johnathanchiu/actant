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
from actant.llm.base import LLMClient
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
    CompactionConfig,
    CompactionOutcome,
    CompactionTrigger,
    ContextSummary,
    FinalizeRunInput,
    RunOutcome,
    RunTurnInput,
    StartRunInput,
    StartedRun,
    StoreSummaryInput,
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
        if payload.sandbox_id and thread.sandbox_id is None:
            thread.sandbox_id = payload.sandbox_id
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
        if payload.summary is not None:
            # A background summary is ready: it is stored before this turn's
            # messages, so they land after its row.
            await self._store_summary(
                payload.agent_id,
                payload.thread_id,
                payload.run_id,
                payload.summary,
                compaction.keep if compaction is not None else [],
                turn_id=payload.turn_id,
                turn_index=payload.turn_index,
            )
            phases.end("summary")
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
        if self.context.turn_gate is not None and not (payload.compacted or payload.admitted):
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
        ahead: CompactionTrigger | None = None
        if compaction is not None and model_view is not None and measuring:
            trigger = _compaction_trigger(agent, compaction.threshold, model_view, view, messages)
            if trigger is not None:
                # Nothing is sent or stored: the workflow compacts, then runs this
                # turn again with the same new messages.
                return TurnResult(
                    turn_id=payload.turn_id,
                    turn_index=payload.turn_index,
                    compaction=trigger,
                )
            if compaction.background is not None:
                ahead = _compaction_trigger(
                    agent, compaction.background, model_view, view, messages, ahead=True
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
                    turn_id=payload.turn_id,
                    turn_index=payload.turn_index,
                    reminded=True,
                    summarize=ahead,
                )
            return TurnResult(
                turn_id=payload.turn_id,
                turn_index=payload.turn_index,
                stop_reason=STOPPED_WITHOUT_FINISHING,
                summarize=ahead,
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
            summarize=ahead,
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
        summary = await self._summarize(payload)
        outcome = await self._store_summary(
            payload.agent_id,
            payload.thread_id,
            payload.run_id,
            summary,
            payload.config.keep,
            turn_id=payload.turn_id,
            turn_index=payload.turn_index,
        )
        if outcome is None:
            raise ApplicationError("the context was compacted meanwhile", non_retryable=True)
        return outcome

    @activity.defn(name=ActivityName.SUMMARIZE_CONTEXT)
    async def summarize_context(self, payload: CompactContextInput) -> ContextSummary:
        """Write a summary of everything through the trigger's boundary, beside the
        turns, and store nothing: the workflow stores it at a turn boundary."""
        return await self._summarize(payload)

    @activity.defn(name=ActivityName.STORE_SUMMARY)
    async def store_summary(self, payload: StoreSummaryInput) -> None:
        """Store a ready background summary while no run is open."""
        await self._store_summary(
            payload.agent_id,
            payload.thread_id,
            payload.run_id,
            payload.summary,
            payload.config.keep,
        )

    async def _summarize(self, payload: CompactContextInput) -> ContextSummary:
        """One call with no tools, on the configured summarizer or the agent's own model,
        summarizing the view through the trigger's boundary (``through``, else before
        ``carried``)."""
        trigger = payload.trigger
        agent = await self.context.agent(payload.agent_id, payload.thread_id)
        view = build_view(
            await self.context.stores.messages.list_for_model(payload.agent_id, payload.thread_id)
        )
        if trigger.through is not None:
            if trigger.through not in view.ids:
                raise ApplicationError(
                    "the summary's boundary left the context", non_retryable=True
                )
            cut = view.ids.index(trigger.through) + 1
        else:
            cut = view.ids.index(trigger.carried[0]) if trigger.carried else len(view.messages)
        through = view.ids[cut - 1]
        if through is None:
            raise ApplicationError("nothing to summarize", non_retryable=True)
        config = payload.config
        instructions = self.context.compaction_instructions
        if config.summary_tokens is not None:
            length = (
                f"Fit the summary in about {config.summary_tokens * 3 // 4} words: lists over "
                "prose, every open item and key fact kept, nothing narrated."
            )
            instructions = f"{instructions}\n\n{length}" if instructions else length
        request = await self._prepare(
            compaction_request(view.messages[:cut], instructions),
            payload.agent_id,
            payload.thread_id,
            payload.run_id,
            payload.turn_id,
        )
        measure = measure_request(agent.persona, request, usage_from=view.usage_from)
        llm = self._summarizer(agent, config, measure.tokens, count_images(request))
        fast = llm is not agent.llm or config.summary_tokens is not None
        reason = "it ran into the cap"
        try:
            summary = await self._summary_call(
                agent, llm, request, measure.tokens, config.summary_tokens
            )
        except ApplicationError as error:
            if not fast:
                raise
            # A provider may refuse an answer that reaches the cap rather than return it.
            summary, reason = None, str(error)
        if summary is None:
            # Cut short at the cap, or the summarizer failed: written again, whole, on the
            # agent's own model with the default cap.
            logger.warning(
                "actant.compaction.summary_retried agent=%s thread=%s model=%s cap=%s why=%s",
                agent.id,
                payload.thread_id,
                llm.model_id,
                config.summary_tokens,
                reason,
            )
            request = await self._prepare(
                compaction_request(view.messages[:cut], self.context.compaction_instructions),
                payload.agent_id,
                payload.thread_id,
                payload.run_id,
                payload.turn_id,
            )
            summary = await self._summary_call(agent, agent.llm, request, measure.tokens, None)
        if not summary:
            raise ApplicationError("the summary call returned no summary", non_retryable=True)
        return ContextSummary(
            summary=summary, through=through, base=view.compaction_id, trigger=trigger
        )

    def _summarizer(
        self, agent: AgentDefinition, config: CompactionConfig, tokens: int, images: int
    ) -> LLMClient:
        """The client the summary is written on: the configured summarizer when it can take
        the request (its declared window and image limit), else the agent's own model."""

        if config.summarizer is None:
            return agent.llm
        llm = self.context.summarizers.get(config.summarizer)
        if llm is None:
            raise ApplicationError(
                f"no summarizer named {config.summarizer!r} is registered", non_retryable=True
            )
        window = _client_limit(llm, "context_window_tokens")
        max_images = _client_limit(llm, "max_images_per_request")
        output = config.summary_tokens or SUMMARY_MAX_OUTPUT_TOKENS
        if (window is not None and tokens + output > window) or (
            max_images is not None and images > max_images
        ):
            logger.info(
                "actant.compaction.summarizer_too_small agent=%s model=%s tokens=%s images=%s",
                agent.id,
                llm.model_id,
                tokens,
                images,
            )
            return agent.llm
        return llm

    async def _summary_call(
        self,
        agent: AgentDefinition,
        llm: LLMClient,
        request: list[Message],
        tokens: int,
        cap: int | None,
    ) -> str | None:
        """The summary, or ``None`` when it used its whole ``cap`` (cut short)."""

        window = _client_limit(llm, "context_window_tokens")
        max_output = min(cap or SUMMARY_MAX_OUTPUT_TOKENS, SUMMARY_MAX_OUTPUT_TOKENS)
        if window is not None:
            max_output = min(max_output, window - tokens)
            if max_output < 1:
                raise ApplicationError(
                    f"no room in the {window}-token window for a summary of {tokens} tokens",
                    non_retryable=True,
                )
        started = time.perf_counter()
        try:
            response = await llm.complete(
                agent.persona, request, [], None, max_output_tokens=max_output
            )
        except Exception as exc:
            # Never fall back to dropping context: the run fails and says why.
            raise ApplicationError(f"the summary call failed: {exc}", non_retryable=True) from exc
        logger.info(
            "actant.compaction.summary agent=%s model=%s seconds=%.1f input=%s output=%s",
            agent.id,
            llm.model_id,
            time.perf_counter() - started,
            response.input_tokens,
            response.output_tokens,
        )
        if cap is not None and (response.output_tokens or 0) >= max_output:
            return None
        return response.content.strip() if isinstance(response.content, str) else ""

    async def _store_summary(
        self,
        agent_id: str,
        thread_id: str,
        run_id: str,
        summary: ContextSummary,
        keep: list[str],
        *,
        turn_id: str | None = None,
        turn_index: int | None = None,
    ) -> CompactionOutcome | None:
        """Append the compaction row for ``summary``. It replaces the messages through
        its boundary; every message the model sees after the boundary is kept, with
        the latest one of each ``keep`` tag before it. ``None``, storing nothing, when
        the context was compacted since the summary started."""
        rows = await self.context.stores.messages.list_for_model(agent_id, thread_id)
        latest = next((m.id for m in reversed(rows) if m.kind == "compaction"), None)
        if latest != summary.base:
            logger.info(
                "actant.compaction.stale agent=%s thread=%s base=%s latest=%s",
                agent_id,
                thread_id,
                summary.base,
                latest,
            )
            return None
        # The one full-history read: the latest message of each kept tag may be
        # older than any compaction since.
        history = await self.context.stores.messages.list_for_thread(agent_id, thread_id)
        ids = [m.id for m in history]
        before = ids.index(summary.through) + 1
        visible = {m.id for m in rows if m.kind == "message"}
        chosen = {*kept_ids(history, before, keep), *(i for i in ids[before:] if i in visible)}
        kept = sorted(chosen, key=ids.index)
        trigger = summary.trigger
        block = CompactionBlock(
            summary=summary.summary,
            kept=[i for i in kept if i is not None],
            reason=trigger.reason,
            tokens_before=trigger.tokens,
            images_before=trigger.images,
            tokens_after=0,
            images_after=0,
        )
        row = Message(role="user", content=[block], kind="compaction")
        fresh = build_view([*(m for m in history if m.id in chosen), row])
        prepared = await self._prepare(fresh.messages, agent_id, thread_id, run_id, turn_id or "")
        agent = await self.context.agent(agent_id, thread_id)
        after = measure_request(agent.persona, fresh.messages, usage_from=fresh.usage_from)
        block = block.model_copy(
            update={"tokens_after": after.tokens, "images_after": count_images(prepared)}
        )
        record = await self.context.stores.messages.append_compaction(agent_id, thread_id, block)
        thread = await self.context.stores.threads.get_or_create(agent_id, thread_id)
        events = self.context.events(thread, run_id=run_id, turn_id=turn_id, turn_index=turn_index)
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


def _client_limit(llm: LLMClient, name: str) -> int | None:
    value = getattr(llm, name, None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _limits(agent: AgentDefinition) -> tuple[int | None, int | None]:
    return (
        _provider_limit(agent, "context_window_tokens"),
        _provider_limit(agent, "max_images_per_request"),
    )


def _compaction_trigger(
    agent: AgentDefinition,
    fraction: float,
    model_view: ModelView,
    request: list[Message],
    prepared: list[Message],
    *,
    ahead: bool = False,
) -> CompactionTrigger | None:
    """Whether the request about to be sent crosses a limit, and what to carry if so.

    ``ahead`` measures against ``fraction`` of the image limit too, for a background
    summary, and names its boundary (``through``)."""
    window, max_images = _limits(agent)
    if window is None and max_images is None:
        return None
    measure = measure_request(agent.persona, request, usage_from=model_view.usage_from)
    measure = replace(measure, images=count_images(prepared))
    reason = crossed_limits(
        measure,
        context_window_tokens=window,
        max_images_per_request=max_images,
        threshold=fraction,
        image_threshold=fraction if ahead else 1.0,
    )
    if reason is None:
        return None
    floor = 1 if model_view.compaction is not None else 0
    start = pending_start(model_view.messages, floor=floor)
    if ahead:
        through = model_view.ids[start - 1] if start > floor else None
        if through is None:
            return None
        return CompactionTrigger(
            reason=reason, tokens=measure.tokens, images=measure.images, through=through
        )
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
