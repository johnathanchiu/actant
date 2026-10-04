"""Run a thread in this process, over the same activities the workflow runs.

`AgentRuntime` hosts a Temporal workflow that drives `RunActivities` and `ToolActivities`
through `execute_activity`. Durability is what that buys, and it costs a server, a worker and
a task queue. A laptop that wants one agent to finish one job needs none of that, and before
this the only way to have it was a second agent loop, which then drifts from the first.

So this is a second *caller*, not a second loop: the same `start_run`, `run_turn` and
`finalize_run`, branching on the same `TurnResult` fields, with `await` where the workflow
has `execute_activity`. What it gives up is exactly what Temporal was providing: nothing
survives the process, a tool awaiting a human has only this call's timeout, and a crash is a
traceback rather than a replayable history.

    stores = InMemoryRuntimeStores()
    runtime = LocalThreadRuntime(ActivityContext(stores=stores, resolve_agent=...))
    outcome = await runtime.run(ThreadInput(agent_id=..., thread_id=...), messages)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import replace

from actant.runtime.temporal.activities.runs import RunActivities
from actant.runtime.temporal.activities.tools import ToolActivities
from actant.runtime.temporal.types import (
    AdmitDecision,
    AdmitInput,
    CompactContextInput,
    CompactionConfig,
    ContextSummary,
    ExecuteInput,
    FinalizeRunInput,
    InboundMessage,
    RunOutcome,
    RunTurnInput,
    StartRunInput,
    StoreSummaryInput,
    ThreadInput,
    TurnResult,
)

__all__ = ["LocalRun", "LocalThreadRuntime"]

logger = logging.getLogger(__name__)


class LocalRun:
    """One run's outcome and why it ended."""

    def __init__(self, outcome: RunOutcome, turn_count: int, stop_reason: str | None) -> None:
        self.outcome = outcome
        self.turn_count = turn_count
        self.stop_reason = stop_reason

    def __repr__(self) -> str:
        return (
            f"LocalRun({self.outcome.value}, turns={self.turn_count}, "
            f"stop_reason={self.stop_reason!r})"
        )


class LocalThreadRuntime:
    """The workflow's turn loop, in this process."""

    def __init__(self, context) -> None:
        self._runs = RunActivities(context)
        self._tools = ToolActivities(context)
        # A background summary in flight per (agent, thread), as the workflow keeps one,
        # and the threads with a run open (a summary written while none is stores itself).
        self._summarizing: dict[tuple[str, str], asyncio.Task[ContextSummary | None]] = {}
        self._running: set[tuple[str, str]] = set()

    async def run(
        self,
        payload: ThreadInput,
        messages: list[InboundMessage] | None = None,
    ) -> LocalRun:
        """Turns until the agent stops, its budget runs out, or a terminal tool ends it."""

        self._running.add(self._key(payload))
        try:
            return await self._run(payload, messages)
        finally:
            self._running.discard(self._key(payload))

    async def _run(self, payload: ThreadInput, messages: list[InboundMessage] | None) -> LocalRun:
        run_id = uuid.uuid4().hex
        started = await self._runs.start_run(
            StartRunInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                max_turns=payload.max_turns_per_run,
                parent_thread_id=payload.parent_thread_id,
                sandbox_id=payload.sandbox_id,
                summarizer=(payload.context_compaction or CompactionConfig()).summarizer,
            )
        )
        if started.error is not None:  # an invalid definition: a terminal run, not a loop
            await self._runs.finalize_run(
                FinalizeRunInput(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                    run_id=run_id,
                    outcome=RunOutcome.FAILED.value,
                    turn_count=started.turn_count,
                    stop_reason=started.error,
                )
            )
            return LocalRun(RunOutcome.FAILED, started.turn_count, started.error)
        turns_remaining = started.max_turns
        turn_count = started.turn_count
        new_messages = list(messages or [])
        text_only_turns = 0
        stop_reason: str | None = None
        outcome = RunOutcome.EXHAUSTED

        while turns_remaining > 0:
            turn_count += 1
            turn_input = RunTurnInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                turn_id=uuid.uuid4().hex,
                turn_index=turn_count,
                new_messages=new_messages,
                text_only_turns=text_only_turns,
                context_compaction=payload.context_compaction,
                summary=await self._ready_summary(payload),
            )
            try:
                turn = await self._runs.run_turn(turn_input)
                if turn.compaction is not None and self._key(payload) in self._summarizing:
                    # wait for the summary being written, store it, measure again
                    summary = await self._take_summary(payload)
                    if summary is not None:
                        turn_input = replace(turn_input, summary=summary, admitted=True)
                        turn = await self._runs.run_turn(turn_input)
                if turn.compaction is not None:  # compact, then the same turn again
                    await self._runs.compact_context(
                        CompactContextInput(
                            agent_id=payload.agent_id,
                            thread_id=payload.thread_id,
                            run_id=run_id,
                            turn_id=turn_input.turn_id,
                            turn_index=turn_count,
                            trigger=turn.compaction,
                            config=payload.context_compaction or CompactionConfig(),
                        )
                    )
                    turn = await self._runs.run_turn(
                        replace(turn_input, compacted=True, summary=None)
                    )
            except Exception as error:  # the workflow fails the run here too
                stop_reason = str(error.__cause__ or error)
                outcome = RunOutcome.FAILED
                break

            new_messages = []
            turns_remaining -= 1
            if turn.summarize is not None and self._key(payload) not in self._summarizing:
                self._summarizing[self._key(payload)] = asyncio.create_task(
                    self._summarize(
                        payload,
                        CompactContextInput(
                            agent_id=payload.agent_id,
                            thread_id=payload.thread_id,
                            run_id=run_id,
                            turn_id=turn_input.turn_id,
                            turn_index=turn_count,
                            trigger=turn.summarize,
                            config=payload.context_compaction or CompactionConfig(),
                        ),
                    )
                )

            if not turn.tool_calls:
                if turn.reminded:  # answered in prose; the reminder is already appended
                    text_only_turns += 1
                    continue
                if turn.stop_reason:
                    stop_reason, outcome = turn.stop_reason, RunOutcome.EXHAUSTED
                else:
                    outcome = RunOutcome.COMPLETED
                break

            try:
                terminal = await self._run_tool_group(payload, turn)
            except Exception as error:
                stop_reason = str(error.__cause__ or error)
                outcome = RunOutcome.FAILED
                break
            if terminal:
                outcome = RunOutcome.COMPLETED
                break

        await self._runs.finalize_run(
            FinalizeRunInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                outcome=outcome.value,
                turn_count=turn_count,
                stop_reason=stop_reason,
            )
        )
        # A summary ready by now is stored; one still being written stores itself.
        if (summary := await self._ready_summary(payload)) is not None:
            await self._store(payload, run_id, summary)
        return LocalRun(outcome, turn_count, stop_reason)

    @staticmethod
    def _key(payload: ThreadInput) -> tuple[str, str]:
        return payload.agent_id, payload.thread_id

    async def _ready_summary(self, payload: ThreadInput) -> ContextSummary | None:
        task = self._summarizing.get(self._key(payload))
        if task is None or not task.done():
            return None
        return await self._take_summary(payload)

    async def _take_summary(self, payload: ThreadInput) -> ContextSummary | None:
        task = self._summarizing.pop(self._key(payload), None)
        return None if task is None else await task

    async def _summarize(
        self, payload: ThreadInput, compact: CompactContextInput
    ) -> ContextSummary | None:
        """The background summary, beside the turns. Taken at the next turn boundary;
        stored here when the thread has no run open by then, so no run waits on it."""
        try:
            summary = await self._runs.summarize_context(compact)
        except Exception as error:  # a failed summary changed nothing; the hard limit compacts
            logger.warning(
                "actant.compaction.background_failed thread=%s error=%s: %s",
                payload.thread_id,
                type(error).__name__,
                error,
            )
            return None
        key = self._key(payload)
        if key not in self._running and self._summarizing.get(key) is asyncio.current_task():
            del self._summarizing[key]
            await self._store(payload, compact.run_id, summary)
        return summary

    async def _store(self, payload: ThreadInput, run_id: str, summary: ContextSummary) -> None:
        await self._runs.store_summary(
            StoreSummaryInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                summary=summary,
                config=payload.context_compaction or CompactionConfig(),
            )
        )

    async def _run_tool_group(self, payload: ThreadInput, turn: TurnResult) -> bool:
        """Admit every call, run what may run, then finalize the group once.

        The workflow's version suspends durably on a tool awaiting a human. Here there is no
        one to wake, so an `AWAIT_HUMAN` tool is left to its admission result and the group
        finalizes without it: a local run is not the place to ask for approval.
        """

        run_id = turn.tool_calls[0].run_id
        group_id = turn.tool_calls[0].group_id

        admits = await asyncio.gather(
            *(
                self._tools.admit_tool(
                    AdmitInput(
                        agent_id=payload.agent_id,
                        thread_id=payload.thread_id,
                        run_id=run_id,
                        tool_call_id=spec.id,
                    )
                )
                for spec in turn.tool_calls
            )
        )
        decisions = {a.tool_call_id: a.decision for a in admits}

        running = [
            self._tools.execute_tool(
                ExecuteInput(
                    agent_id=payload.agent_id,
                    thread_id=payload.thread_id,
                    run_id=run_id,
                    tool_call_id=spec.id,
                    context=payload.context,
                )
            )
            for spec in turn.tool_calls
            if decisions.get(spec.id) == AdmitDecision.EXECUTE.value
        ]
        # gather, not as_completed: every sibling drains before finalize, as the workflow
        # does, so a failure never races a late tool's write.
        outcomes = await asyncio.gather(*running, return_exceptions=True)
        failure = next((o for o in outcomes if isinstance(o, BaseException)), None)
        terminal = any(getattr(o, "terminal", False) for o in outcomes)
        if failure is not None:
            raise failure

        await self._tools.finalize_tool_group(group_id)
        return terminal
