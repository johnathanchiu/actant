"""Workflow definitions for the Actant Temporal runtime.

One ``AgentThreadWorkflow`` execution per ``(agent_id, thread_id)``.
The workflow id encodes the thread, the workflow's lifetime is the
thread's lifetime, and all interaction with the thread (send a
message, cancel) flows through this workflow.

The workflow is a thin orchestrator. It:

1. Receives ``inbound`` signals (user messages) into an in-memory inbox.
2. For each agent run: drains the inbox and advances through turns until the model
   stops emitting tool_calls or the turn budget is exhausted.
3. For each turn's tool_calls: admits every tool, then executes ALLOW tools
   and durably suspends WAIT tools until a resolution signal arrives.
4. Finalizes each tool group via ``finalize_tool_group`` (writes the
   tool_result messages — the transcript invariant lives there).

Activities report outcomes. Signals report external events. Only this workflow
advances the agent run. Deferred waits use ``workflow.wait_condition``: no
activity, worker thread, or polling loop remains active while a human decides.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError, ChildWorkflowError

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
    DeferredToolResolution,
    ExecuteInput,
    ExecuteOutcome,
    ExecuteStatus,
    FinalizeRunInput,
    InboundMessage,
    ReadThreadAnswerInput,
    ResolveToolInput,
    RunOutcome,
    RunTurnInput,
    SpawnRequest,
    StartRunInput,
    ThreadInput,
    ThreadOutcome,
    ThreadStateView,
    TurnResult,
)

_RUN_TURN_TIMEOUT = timedelta(minutes=10)
_TOOL_TIMEOUT = timedelta(minutes=10)
_FINALIZE_TIMEOUT = timedelta(seconds=60)
_PROJECTION_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class AgentThreadWorkflow:
    """The thread.

    The workflow owns the durable lifetime of one agent thread. Each inbox
    activation starts an agent run that continues until ``COMPLETED``,
    ``EXHAUSTED``, ``FAILED``, or ``CANCELLED``. After finalization, the thread
    workflow parks on ``wait_condition`` until another ``inbound`` message
    arrives or the workflow is cancelled.

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
                if payload.exit_when_idle and not self._inbox:
                    return ThreadOutcome.STOPPED.value
                self._compact_history_if_needed(payload)
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

        await workflow.execute_activity_method(
            RunActivities.start_run,
            StartRunInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                max_turns=payload.max_turns_per_run,
            ),
            start_to_close_timeout=_PROJECTION_TIMEOUT,
        )
        outcome = await self._run_agent(payload, run_id, new_messages)
        await workflow.execute_activity_method(
            RunActivities.finalize_run,
            FinalizeRunInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                outcome=outcome.value,
                turn_count=self._turn_count_total,
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
        turns_remaining = payload.max_turns_per_run

        while turns_remaining > 0 and not self._cancelled:
            turn_id = workflow.uuid4().hex
            turn_index = self._turn_count_total + 1

            try:
                turn = await workflow.execute_activity_method(
                    RunActivities.run_turn,
                    RunTurnInput(
                        agent_id=payload.agent_id,
                        thread_id=payload.thread_id,
                        run_id=run_id,
                        turn_id=turn_id,
                        turn_index=turn_index,
                        new_messages=new_messages,
                    ),
                    start_to_close_timeout=_RUN_TURN_TIMEOUT,
                    retry_policy=RetryPolicy(maximum_attempts=1),
                )
            except Exception:
                # RUN_TURN failed (LLM error, cancellation, etc.).
                # Surface as FAILED and return to the thread lifecycle —
                # next user message starts a fresh run. The thread
                # stays alive.
                return RunOutcome.FAILED

            new_messages = []
            self._turn_count_total += 1
            turns_remaining -= 1

            if not turn.tool_calls:
                return RunOutcome.COMPLETED

            should_stop = await self._run_tool_group(payload, turn)
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

        Tool-level failures are absorbed inside activities (each is
        infallible at its boundary), so this method has no try/except.
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
                start_to_close_timeout=_TOOL_TIMEOUT,
                retry_policy=RetryPolicy(maximum_attempts=1),
            )
            for spec in turn.tool_calls
        ]
        admits: dict[str, AdmitOutcome] = {}
        for fut in workflow.as_completed(admit_handles):
            outcome = await fut
            admits[outcome.tool_call_id] = outcome

        # 2. Each non-blocked tool produces one outcome. ALLOW tools execute
        #    normally. WAIT tools suspend inside the workflow until their
        #    resolution signal arrives. BLOCK tools are already terminal.
        exec_handles = []
        for spec in turn.tool_calls:
            decision = admits[spec.id].decision
            if decision == AdmitDecision.ALLOW.value:
                exec_handles.append(
                    workflow.start_activity_method(
                        ToolActivities.execute_tool,
                        ExecuteInput(
                            agent_id=payload.agent_id,
                            thread_id=payload.thread_id,
                            run_id=run_id,
                            tool_call_id=spec.id,
                        ),
                        start_to_close_timeout=_TOOL_TIMEOUT,
                        retry_policy=RetryPolicy(maximum_attempts=1),
                    )
                )
            elif decision == AdmitDecision.SPAWN.value:
                request = admits[spec.id].spawn_request
                if request is None:
                    raise ApplicationError("admit returned SPAWN without a request")
                exec_handles.append(
                    asyncio.create_task(
                        self._spawn_tool(
                            payload,
                            run_id=run_id,
                            tool_call_id=spec.id,
                            request=request,
                        )
                    )
                )
            elif decision == AdmitDecision.WAIT.value:
                exec_handles.append(
                    asyncio.create_task(
                        self._resolve_tool(
                            payload,
                            run_id=run_id,
                            tool_call_id=spec.id,
                        )
                    )
                )
            # else BLOCK — nothing to await

        # 3. This is the durable tool-group barrier. Temporal wakes the
        #    workflow only for activity completions, signals, timers, or cancel.
        terminal_tool = False
        for fut in workflow.as_completed(exec_handles):
            outcome = await fut  # result already persisted by activity body
            terminal_tool = terminal_tool or outcome.terminal

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

    async def _spawn_tool(
        self,
        payload: ThreadInput,
        *,
        run_id: str,
        tool_call_id: str,
        request: SpawnRequest,
    ) -> ExecuteOutcome:
        """Delegate to another agent and wait for what it says.

        A child workflow rather than a separately started one, so the
        parent simply awaits its result. Started with the id the admitting
        side derived from this tool call: a replayed or retried parent
        reattaches to the run already in flight instead of paying for a
        second one.

        The child exits when idle -- a conversation parks waiting for the
        next message, and a parent awaiting that would wait forever.

        Cancelling the parent cancels the child. The cancel arrives as a
        signal, which only sets a flag, so the await has to watch for it:
        without that the parent would sit here until the child finished
        work nobody is waiting for any more, and only notice the cancel
        afterwards. Every other tool path already breaks on the flag.
        """
        child = await workflow.start_child_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(
                agent_id=request.agent_id,
                thread_id=request.thread_id,
                max_turns_per_run=payload.max_turns_per_run,
                external_resolution_timeout_seconds=(payload.external_resolution_timeout_seconds),
                history_size_threshold=payload.history_size_threshold,
                carry_inbox=[InboundMessage(content=request.message)],
                exit_when_idle=True,
            ),
            id=request.thread_id,
        )
        watch = asyncio.ensure_future(workflow.wait_condition(lambda: self._cancelled))
        done, _ = await workflow.wait([child, watch], return_when=asyncio.FIRST_COMPLETED)

        if child not in done:
            # The parent was cancelled while the child was still working.
            child.cancel()
            answer = "The delegated agent was cancelled."
            approved = False
        else:
            watch.cancel()
            try:
                await child
            except ChildWorkflowError as error:
                answer = f"The delegated agent failed: {error}"
                approved = False
            else:
                answer = await workflow.execute_activity_method(
                    ToolActivities.read_thread_answer,
                    ReadThreadAnswerInput(agent_id=request.agent_id, thread_id=request.thread_id),
                    start_to_close_timeout=_PROJECTION_TIMEOUT,
                )
                approved = True

        # Persisted through the same path a deferred resolution takes, so a
        # delegated result and an externally answered one are the same kind
        # of thing to everything downstream.
        self._resolving_tool_ids.add(tool_call_id)
        outcome = await workflow.execute_activity_method(
            ToolActivities.resolve_tool,
            ResolveToolInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                tool_call_id=tool_call_id,
                resolution=DeferredToolResolution(
                    tool_call_id=tool_call_id,
                    approved=approved,
                    answer=answer,
                ),
            ),
            start_to_close_timeout=_TOOL_TIMEOUT,
            retry_policy=RetryPolicy(maximum_attempts=1),
        )
        self._resolving_tool_ids.discard(tool_call_id)
        self._resolved_tool_ids.add(tool_call_id)
        return outcome

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
            start_to_close_timeout=_TOOL_TIMEOUT,
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

    def _compact_history_if_needed(self, payload: ThreadInput) -> None:
        """Rotate Temporal history between agent runs, preserving thread state."""
        if workflow.info().get_current_history_length() <= _history_rotation_threshold(payload):
            return
        workflow.continue_as_new(
            ThreadInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                max_turns_per_run=payload.max_turns_per_run,
                external_resolution_timeout_seconds=(payload.external_resolution_timeout_seconds),
                carry_inbox=list(self._inbox),
                history_size_threshold=payload.history_size_threshold,
                turn_count_total=self._turn_count_total,
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
