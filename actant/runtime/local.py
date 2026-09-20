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
import uuid

from actant.runtime.temporal.activities.runs import RunActivities
from actant.runtime.temporal.activities.tools import ToolActivities
from actant.runtime.temporal.types import (
    AdmitDecision,
    AdmitInput,
    ExecuteInput,
    FinalizeRunInput,
    InboundMessage,
    RunOutcome,
    RunTurnInput,
    StartRunInput,
    ThreadInput,
    TurnResult,
)

__all__ = ["LocalRun", "LocalThreadRuntime"]


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

    async def run(
        self,
        payload: ThreadInput,
        messages: list[InboundMessage] | None = None,
    ) -> LocalRun:
        """Turns until the agent stops, its budget runs out, or a terminal tool ends it."""

        run_id = uuid.uuid4().hex
        started = await self._runs.start_run(
            StartRunInput(
                agent_id=payload.agent_id,
                thread_id=payload.thread_id,
                run_id=run_id,
                max_turns=payload.max_turns_per_run,
                parent_thread_id=payload.parent_thread_id,
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
            try:
                turn = await self._runs.run_turn(
                    RunTurnInput(
                        agent_id=payload.agent_id,
                        thread_id=payload.thread_id,
                        run_id=run_id,
                        turn_id=uuid.uuid4().hex,
                        turn_index=turn_count,
                        new_messages=new_messages,
                        text_only_turns=text_only_turns,
                    )
                )
            except Exception as error:  # the workflow fails the run here too
                stop_reason = str(error.__cause__ or error)
                outcome = RunOutcome.FAILED
                break

            new_messages = []
            turns_remaining -= 1

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
        return LocalRun(outcome, turn_count, stop_reason)

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
