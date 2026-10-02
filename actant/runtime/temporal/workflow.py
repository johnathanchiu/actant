"""Workflow definitions for the Actant Temporal runtime.

One ``AgentThreadWorkflow`` execution per ``(agent_id, thread_id)``.
The workflow id encodes the thread. Executions close when the inbox is empty;
a later message starts another execution with the same logical id. Commands
(send a message, cancel) flow through Temporal.

The workflow is a thin orchestrator. It:

1. Receives ``inbound`` signals (user messages) into an in-memory inbox.
2. For each agent run: drains the inbox and advances through turns until the model
   stops emitting tool_calls or the turn budget is exhausted. With
   ``interleave_inbox`` it also drains before each later turn, so a message
   sent mid-run reaches the model on the next turn.
3. For each turn's tool_calls: admits every tool, then executes EXECUTE
   tools and durably suspends AWAIT_HUMAN tools until a person answers.
4. Finalizes each tool group via ``finalize_tool_group`` (writes the
   tool_result messages — the transcript invariant lives there).
5. With ``context_compaction``, compacts the model's context when a turn
   reports that its request would cross a limit, then runs that turn again.
   With ``CompactionConfig.background``, a turn past that lower mark starts a
   summary beside the turns (``summarize_context``); it is stored at the next
   turn boundary after it is ready, or once the thread goes idle.
   This is unrelated to rotating Temporal's event history
   (``history_size_threshold``), which never changes what the model sees.

Activities report outcomes. Signals report external events. Only this workflow
advances the agent run. Deferred waits use ``workflow.wait_condition``: no
activity, worker thread, or polling loop remains active while a human decides.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from dataclasses import replace
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from actant.runtime.temporal.activities import (
        RunActivities,
        ThreadActivities,
        ToolActivities,
    )

from actant.runtime.temporal.types import (
    AdmitDecision,
    AdmitInput,
    AdmitOutcome,
    ApplyThreadCancellationInput,
    CompactContextInput,
    CompactionConfig,
    ContextSummary,
    DeferredToolResolution,
    ExecuteInput,
    ExecuteOutcome,
    ExecuteStatus,
    FinalizeRunInput,
    InboundMessage,
    ResolveToolInput,
    RunOutcome,
    RunTurnInput,
    StartRunInput,
    StoreSummaryInput,
    ThreadInput,
    ThreadOutcome,
    ThreadStateView,
    TurnResult,
)

_ADMIT_TIMEOUT = timedelta(minutes=10)
_FINALIZE_TIMEOUT = timedelta(seconds=60)
_PROJECTION_TIMEOUT = timedelta(seconds=30)
#: A blocking compaction whose summary call failed tries again, backing off, within the
#: same run: a run that fails at once only has its caller start the next into the same
#: failure. A refusal that would end the same way again is non-retryable and fails at once.
#: A tool call whose worker was lost (it stopped heartbeating, or outlived its timeout)
#: is attempted again: ``execute_tool`` returns a stored result as is, runs a
#: ``retry_safe`` tool again, and closes any other as interrupted. Every failure inside
#: the activity is already a result, so only a lost attempt reaches this policy.
_LOST_TOOL_RETRY = RetryPolicy(maximum_attempts=3)
#: A model turn whose worker was lost is attempted again: ``run_turn`` picks up from what
#: the lost attempt stored, and fails every other error non-retryably.
_LOST_TURN_RETRY = RetryPolicy(maximum_attempts=3)
_COMPACT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2), backoff_coefficient=2.0, maximum_attempts=3
)


@workflow.defn
class AgentThreadWorkflow:
    """The thread.

    The workflow owns the durable lifetime of one agent thread. Each inbox
    activation starts an agent run that continues until ``COMPLETED``,
    ``EXHAUSTED``, ``FAILED``, or ``CANCELLED``. After finalization, the thread
    workflow closes if its inbox is empty. A later message starts a new
    execution with the same logical thread id and persisted transcript.

    Exhaustion ends only the current agent run. The agent thread remains alive
    and the next inbound message starts a fresh run with a fresh budget.
    """

    def __init__(self) -> None:
        self._inbox: list[InboundMessage] = []
        self._cancelled = False
        self._turn_count_total = 0
        self._current_run_id: str | None = None
        self._tool_resolutions: dict[str, DeferredToolResolution] = {}
        self._resolving_tool_ids: set[str] = set()
        self._resolved_tool_ids: set[str] = set()
        # Populated by ``run()`` so ``get_state`` can echo the workflow's
        # logical identity without parsing workflow_id strings.
        self._agent_id: str = ""
        self._thread_id: str = ""
        self._stop_reason: str | None = None
        # A background summary in flight (``CompactionConfig.background``), and
        # the run that started it.
        self._summarizing: workflow.ActivityHandle[ContextSummary] | None = None
        self._summary_run_id = ""

    # === Signals ===

    @workflow.signal
    def inbound(self, msg: InboundMessage) -> None:
        self._inbox.append(msg)

    @workflow.signal
    def cancel(self) -> None:
        self._cancelled = True

    @workflow.signal
    def resolve_tool(self, resolution: DeferredToolResolution) -> None:
        """Record the first resolution for a tool; duplicates are harmless."""
        tool_call_id = resolution.tool_call_id
        if tool_call_id in self._resolving_tool_ids or tool_call_id in self._resolved_tool_ids:
            return
        self._tool_resolutions.setdefault(tool_call_id, resolution)

    # === Queries ===

    @workflow.query
    def get_state(self) -> ThreadStateView:
        return ThreadStateView(
            agent_id=self._agent_id,
            thread_id=self._thread_id,
            inbox_size=len(self._inbox),
            turn_count_total=self._turn_count_total,
            current_run_id=self._current_run_id,
            cancelled=self._cancelled,
        )

    # === Run ===

    @workflow.run
    async def run(self, payload: ThreadInput) -> str:
        self._agent_id = payload.agent_id
        self._thread_id = payload.thread_id
        self._turn_count_total = payload.turn_count_total
        # Carry-forward inbox lands here on continue_as_new.
        if payload.carry_inbox:
            self._inbox.extend(payload.carry_inbox)

        try:
            while True:
                await self._wait_for_message_or_cancellation()
                if self._cancelled:
                    break
                await self._run_next_agent_run(payload)
                if not self._inbox and self._summarizing is not None:
                    await self._store_summary_when_idle(payload)
                if not self._inbox:
                    # Nothing left to do, so stop rather than sit open. The
                    # conversation is not over: history lives in the stores,
                    # and the next message starts this workflow again with
                    # the same id. Parking would only mean an idle execution
                    # per thread, held for as long as the thread exists.
                    return ThreadOutcome.STOPPED.value
                # Only reached when messages arrived while the last run was
                # working, so the thread keeps going without ever going idle.
                # That is the one case where history still accumulates, and
                # the only reason rotation survives threads that end.
                self._rotate_history_if_needed(payload)
        except asyncio.CancelledError:
            await self._record_cancellation(payload)
            raise
        return ThreadOutcome.CANCELLED.value

    async def _wait_for_message_or_cancellation(self) -> None:
        """Suspend the thread until a message arrives or it is cancelled."""
        await workflow.wait_condition(lambda: bool(self._inbox) or self._cancelled)

    async def _run_next_agent_run(self, payload: ThreadInput) -> None:
        """Open, execute, and finalize one agent run for the queued inbox."""
        run_id = workflow.uuid4().hex
        self._current_run_id = run_id
        new_messages = self._drain_inbox()

        # Seeded from the store, not carried: a thread ends when it is done
        # and restarts on the next message, so a count held only here would
        # reset and turn numbers would repeat within one thread.
        started = await workflow.execute_activity_method(
            RunActivities.start_run,
            StartRunInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                max_turns=payload.max_turns_per_run,
                parent_thread_id=payload.parent_thread_id,
                sandbox_id=payload.sandbox_id,
                summarizer=(payload.context_compaction or CompactionConfig()).summarizer,
            ),
            start_to_close_timeout=_PROJECTION_TIMEOUT,
        )
        self._turn_count_total = started.turn_count
        payload = replace(payload, max_turns_per_run=started.max_turns)
        self._stop_reason = started.error
        outcome = (
            RunOutcome.FAILED
            if started.error is not None
            else await self._run_agent(payload, run_id, new_messages)
        )
        await workflow.execute_activity_method(
            RunActivities.finalize_run,
            FinalizeRunInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                outcome=outcome.value,
                turn_count=self._turn_count_total,
                stop_reason=self._stop_reason,
            ),
            start_to_close_timeout=_PROJECTION_TIMEOUT,
        )
        self._current_run_id = None

    async def _run_agent(
        self,
        payload: ThreadInput,
        run_id: str,
        new_messages: list[InboundMessage],
    ) -> RunOutcome:
        """Run agent turns until a stop condition or the turn budget."""
        assert payload.max_turns_per_run is not None
        turns_remaining = payload.max_turns_per_run
        text_only_turns = 0

        while turns_remaining > 0 and not self._cancelled:
            if payload.interleave_inbox and self._inbox:
                # Messages that arrived while the last turn's tools ran. The
                # previous group is finalized by now, so run_turn appends
                # them after its tool results and ahead of this model call.
                # Without the flag this branch is never taken, and a history
                # recorded before the flag existed replays unchanged.
                new_messages = [*new_messages, *self._drain_inbox()]
            turn_id = workflow.uuid4().hex
            turn_index = self._turn_count_total + 1

            turn_input = RunTurnInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                turn_id=turn_id,
                turn_index=turn_index,
                new_messages=new_messages,
                text_only_turns=text_only_turns,
                context_compaction=payload.context_compaction,
                summary=await self._ready_summary(),
            )
            try:
                turn = await workflow.execute_activity_method(
                    RunActivities.run_turn,
                    turn_input,
                    start_to_close_timeout=timedelta(seconds=payload.activity_timeouts.turn_s),
                    heartbeat_timeout=timedelta(
                        seconds=payload.activity_timeouts.turn_heartbeat_s
                    ),
                    retry_policy=_LOST_TURN_RETRY,
                )
                if turn.compaction is not None and self._summarizing is not None:
                    # Past the hard limit with a summary already being written:
                    # wait for it rather than start another, store it, and
                    # measure the same turn again.
                    summary = await self._take_summary()
                    if summary is not None:
                        if payload.interleave_inbox and self._inbox:
                            new_messages = [*new_messages, *self._drain_inbox()]
                        turn_input = replace(
                            turn_input, new_messages=new_messages, summary=summary, admitted=True
                        )
                        turn = await workflow.execute_activity_method(
                            RunActivities.run_turn,
                            turn_input,
                            start_to_close_timeout=timedelta(
                                seconds=payload.activity_timeouts.turn_s
                            ),
                            heartbeat_timeout=timedelta(
                                seconds=payload.activity_timeouts.turn_heartbeat_s
                            ),
                            retry_policy=_LOST_TURN_RETRY,
                        )
                if turn.compaction is not None:
                    # The request would have crossed a context limit and
                    # nothing was sent. Decided by the activity alone: every
                    # history recorded without compaction has no such result
                    # and never reaches this branch.
                    await workflow.execute_activity_method(
                        RunActivities.compact_context,
                        CompactContextInput(
                            agent_id=payload.agent_id,
                            thread_id=payload.thread_id,
                            run_id=run_id,
                            turn_id=turn_id,
                            turn_index=turn_index,
                            trigger=turn.compaction,
                            config=payload.context_compaction or CompactionConfig(),
                        ),
                        start_to_close_timeout=timedelta(
                            seconds=payload.activity_timeouts.compact_s
                        ),
                        retry_policy=_COMPACT_RETRY,
                    )
                    # The same turn, from the fresh context; it may not compact
                    # again. Its new messages were not stored, so they land after
                    # the compaction row, with anything that arrived meanwhile.
                    if payload.interleave_inbox and self._inbox:
                        new_messages = [*new_messages, *self._drain_inbox()]
                    turn = await workflow.execute_activity_method(
                        RunActivities.run_turn,
                        replace(
                            turn_input, new_messages=new_messages, compacted=True, summary=None
                        ),
                        start_to_close_timeout=timedelta(seconds=payload.activity_timeouts.turn_s),
                        heartbeat_timeout=timedelta(
                            seconds=payload.activity_timeouts.turn_heartbeat_s
                        ),
                        retry_policy=_LOST_TURN_RETRY,
                    )
            except Exception as error:
                self._stop_reason = str(error.__cause__ or error)
                # RUN_TURN failed (LLM error, cancellation, etc.).
                # Surface as FAILED and return to the thread lifecycle —
                # next user message starts a fresh run. The thread
                # stays alive.
                return RunOutcome.FAILED

            new_messages = []
            self._turn_count_total += 1
            turns_remaining -= 1
            if turn.summarize is not None and self._summarizing is None:
                # Only a thread with ``CompactionConfig.background`` reports this.
                self._summarizing = workflow.start_activity_method(
                    RunActivities.summarize_context,
                    CompactContextInput(
                        agent_id=payload.agent_id,
                        thread_id=payload.thread_id,
                        run_id=run_id,
                        turn_id=turn_id,
                        turn_index=turn_index,
                        trigger=turn.summarize,
                        config=payload.context_compaction or CompactionConfig(),
                    ),
                    start_to_close_timeout=timedelta(seconds=payload.activity_timeouts.compact_s),
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
                self._summary_run_id = run_id

            if not turn.tool_calls:
                if turn.reminded:
                    # A task agent answered in prose; the activity appended the
                    # reminder, so the next turn sees it.
                    text_only_turns += 1
                    continue
                if turn.stop_reason:
                    self._stop_reason = turn.stop_reason
                    return RunOutcome.EXHAUSTED
                return RunOutcome.COMPLETED

            try:
                should_stop = await self._run_tool_group(payload, turn)
            except ActivityError as error:
                self._stop_reason = str(error.__cause__ or error)
                return RunOutcome.FAILED
            if self._cancelled:
                return RunOutcome.CANCELLED
            if should_stop:
                return RunOutcome.COMPLETED

        if self._cancelled:
            return RunOutcome.CANCELLED
        return RunOutcome.EXHAUSTED

    async def _run_tool_group(
        self,
        payload: ThreadInput,
        turn: TurnResult,
    ) -> bool:
        """Admit, execute or resolve in parallel, then finalize once.

        Tool exceptions become structured results. A tool call lost with its
        worker is attempted again (``_LOST_TOOL_RETRY``) and ends as a result too.
        Other Temporal-level failures fail the run after siblings drain.
        Every tool_call ends with a terminal status and a persisted
        result by the time ``finalize_tool_group`` runs — which appends
        the tool_result messages and closes the transcript invariant.
        """
        run_id = turn.tool_calls[0].run_id
        group_id = turn.tool_calls[0].group_id

        # 1. Classify all tools in parallel.
        admit_handles = [
            workflow.start_activity_method(
                ToolActivities.admit_tool,
                AdmitInput(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                    run_id=run_id,
                    tool_call_id=spec.id,
                ),
                start_to_close_timeout=_ADMIT_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            for spec in turn.tool_calls
        ]
        admits: dict[str, AdmitOutcome] = {}
        admission_error: ActivityError | None = None
        for fut in workflow.as_completed(admit_handles):
            try:
                outcome = await fut
                admits[outcome.tool_call_id] = outcome
            except ActivityError as error:
                admission_error = error
        if admission_error is not None:
            raise admission_error

        # 2. Each tool produces one outcome. EXECUTE runs it. AWAIT_HUMAN
        #    suspends inside the workflow until a person answers. DENY is
        #    already terminal -- admission wrote the refusal as its result.
        exec_handles = []
        for spec in turn.tool_calls:
            decision = admits[spec.id].decision
            if decision == AdmitDecision.EXECUTE.value:
                exec_handles.append(
                    workflow.start_activity_method(
                        ToolActivities.execute_tool,
                        ExecuteInput(
                            agent_id=payload.agent_id,
                            thread_id=payload.thread_id,
                            run_id=run_id,
                            tool_call_id=spec.id,
                        ),
                        start_to_close_timeout=timedelta(seconds=payload.activity_timeouts.tool_s),
                        # The activity heartbeats while a tool runs, so a
                        # worker that dies mid-tool is noticed in minutes
                        # rather than at the tool's ceiling.
                        heartbeat_timeout=timedelta(
                            seconds=payload.activity_timeouts.tool_heartbeat_s
                        ),
                        retry_policy=_LOST_TOOL_RETRY,
                    )
                )
            elif decision == AdmitDecision.AWAIT_HUMAN.value:
                exec_handles.append(
                    asyncio.create_task(
                        self._resolve_tool(
                            payload,
                            run_id=run_id,
                            tool_call_id=spec.id,
                        )
                    )
                )
            # else DENY -- admission already produced the result, so there
            # is nothing to wait on and nothing to run.

        # 3. This is the durable tool-group barrier. Temporal wakes the
        #    workflow only for activity completions, signals, timers, or cancel.
        terminal_tool = False
        execution_error: ActivityError | None = None
        for fut in workflow.as_completed(exec_handles):
            try:
                outcome = await fut  # result already persisted by activity body
                terminal_tool = terminal_tool or outcome.terminal
            except ActivityError as error:
                # Drain siblings before finalization; do not race late tool writes.
                # Reached only once its retries are spent.
                execution_error = error
        if execution_error is not None:
            raise execution_error

        if self._cancelled:
            return terminal_tool

        # 4. Finalize the group — appends tool_result messages in
        #    sorted-by-id order, closing the transcript invariant.
        await workflow.execute_activity_method(
            ToolActivities.finalize_tool_group,
            group_id,
            start_to_close_timeout=_FINALIZE_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=2),
        )
        return terminal_tool

    async def _resolve_tool(
        self,
        payload: ThreadInput,
        *,
        run_id: str,
        tool_call_id: str,
    ) -> ExecuteOutcome:
        """Suspend until one external resolution arrives, then persist it."""
        try:
            await workflow.wait_condition(
                lambda: tool_call_id in self._tool_resolutions or self._cancelled,
                timeout=timedelta(seconds=payload.external_resolution_timeout_seconds),
                timeout_summary=f"resolve-tool-{tool_call_id}",
            )
        except asyncio.TimeoutError:
            resolution = None
        else:
            if self._cancelled:
                return ExecuteOutcome(
                    tool_call_id=tool_call_id,
                    status=ExecuteStatus.FAILED.value,
                )
            resolution = self._tool_resolutions.pop(tool_call_id)
            self._resolving_tool_ids.add(tool_call_id)

        outcome = await workflow.execute_activity_method(
            ToolActivities.resolve_tool,
            ResolveToolInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                tool_call_id=tool_call_id,
                resolution=resolution,
            ),
            start_to_close_timeout=timedelta(seconds=payload.activity_timeouts.tool_s),
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        self._resolving_tool_ids.discard(tool_call_id)
        self._resolved_tool_ids.add(tool_call_id)
        return outcome

    async def _record_cancellation(self, payload: ThreadInput) -> None:
        """Persist cancellation without leaving an open run or tool call."""
        if self._current_run_id is not None:
            await asyncio.shield(
                workflow.execute_activity_method(
                    RunActivities.finalize_run,
                    FinalizeRunInput(
                        agent_id=payload.agent_id,
                        thread_id=payload.thread_id,
                        run_id=self._current_run_id,
                        outcome=RunOutcome.CANCELLED.value,
                        turn_count=self._turn_count_total,
                    ),
                    start_to_close_timeout=_PROJECTION_TIMEOUT,
                )
            )
        await asyncio.shield(
            workflow.execute_activity_method(
                ThreadActivities.apply_thread_cancellation,
                ApplyThreadCancellationInput(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                ),
                start_to_close_timeout=_PROJECTION_TIMEOUT,
            )
        )

    async def _ready_summary(self) -> ContextSummary | None:
        """The background summary, when it is ready; never waits for one."""
        if self._summarizing is None or not self._summarizing.done():
            return None
        return await self._take_summary()

    async def _take_summary(self) -> ContextSummary | None:
        """Wait for the background summary. A failed one changed nothing: the
        hard limit still compacts when it is reached."""
        handle = self._summarizing
        self._summarizing = None
        if handle is None:
            return None
        try:
            return await handle
        except ActivityError as error:
            cause = error.__cause__ or error
            workflow.logger.warning(
                "actant.compaction.background_failed thread=%s error=%s: %s",
                self._thread_id,
                type(cause).__name__,
                cause,
            )
            return None

    async def _store_summary_when_idle(self, payload: ThreadInput) -> None:
        """The inbox is empty and a summary is still being written: store it before
        the thread closes, unless a message (which runs first) or a cancel comes."""
        handle = self._summarizing
        assert handle is not None
        await workflow.wait_condition(
            lambda: bool(self._inbox) or self._cancelled or handle.done()
        )
        if self._inbox or self._cancelled:
            return
        summary = await self._take_summary()
        if summary is None:
            return
        try:
            await workflow.execute_activity_method(
                RunActivities.store_summary,
                StoreSummaryInput(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                    run_id=self._summary_run_id,
                    summary=summary,
                    config=payload.context_compaction or CompactionConfig(),
                ),
                start_to_close_timeout=_PROJECTION_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
        except ActivityError as error:
            workflow.logger.warning("storing a summary failed: %s", error.__cause__ or error)

    def _rotate_history_if_needed(self, payload: ThreadInput) -> None:
        """Rotate Temporal's event history between agent runs, preserving thread state.

        Not model-context compaction: the model's request is built from the
        stores, so rotating never changes what the model sees.
        """
        if workflow.info().get_current_history_length() <= _history_rotation_threshold(payload):
            return
        if self._summarizing is not None:
            return  # soft: rotate after the summary in flight is stored
        workflow.continue_as_new(
            ThreadInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                max_turns_per_run=payload.max_turns_per_run,
                external_resolution_timeout_seconds=(payload.external_resolution_timeout_seconds),
                carry_inbox=list(self._inbox),
                history_size_threshold=payload.history_size_threshold,
                turn_count_total=self._turn_count_total,
                interleave_inbox=payload.interleave_inbox,
                context_compaction=payload.context_compaction,
                activity_timeouts=payload.activity_timeouts,
            )
        )

    def _drain_inbox(self) -> list[InboundMessage]:
        msgs = list(self._inbox)
        self._inbox.clear()
        return msgs


def _history_rotation_threshold(payload: ThreadInput) -> int:
    return max(1, payload.history_size_threshold)


# Convenience name so callers don't have to know the class location for
# Worker registration.
WORKFLOWS: list[type] = [AgentThreadWorkflow]
