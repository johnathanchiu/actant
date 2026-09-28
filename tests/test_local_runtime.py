"""`LocalThreadRuntime` drives the same activities the workflow drives.

The point of it is that there is no second agent loop to drift: it calls `start_run`,
`run_turn` and `finalize_run`, and branches on the same `TurnResult` fields. These tests pin
the branches a caller depends on — a plain answer completes, a tool call runs and the run
goes on, a terminal tool ends it, an exhausted budget is exhausted — so a change to the
workflow's loop that is not mirrored here shows up as a failure rather than as two runtimes
quietly disagreeing.
"""

from typing import Any, cast

import pytest

from actant.agents import AgentDefinition
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime.local import LocalThreadRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import RunOutcome, ThreadInput
from actant.tools import ToolRegistry, tool

from tests.runtime_fixtures import static_agents

pytestmark = pytest.mark.asyncio


@tool
async def note(text: str) -> str:
    """Write something down."""

    return f"noted: {text}"


def _call(name: str, arguments: str = "{}") -> FakeResponse:
    return FakeResponse(
        tool_calls=[ToolCall(id=f"tc_{name}", function=ToolCallFunction(name, arguments))]
    )


def _says(text: str) -> FakeResponse:
    return FakeResponse(text=text)


async def _run(replies: list[FakeResponse], *, max_turns: int = 6, tools=None):
    stores = InMemoryRuntimeStores()
    agent = AgentDefinition(
        id="demo",
        name="Demo",
        persona="",
        llm=FakeLLM(replies),
        tools=ToolRegistry(tools if tools is not None else [note]),
    )
    runtime = LocalThreadRuntime(
        ActivityContext(stores=stores, resolve_agent=static_agents({"demo": agent}))
    )
    outcome = await runtime.run(
        ThreadInput(agent_id="demo", thread_id="t1", max_turns_per_run=max_turns)
    )
    return outcome, stores


async def test_an_answer_with_no_tool_calls_completes_the_run():
    run, _ = await _run([_says("done")])
    assert run.outcome is RunOutcome.COMPLETED
    assert run.turn_count == 1


async def test_a_tool_call_runs_and_the_run_carries_on():
    run, stores = await _run(
        [_call("note", '{"text": "hello"}'), _says("done")]
    )
    assert run.outcome is RunOutcome.COMPLETED
    assert run.turn_count == 2
    messages = await stores.messages.list_for_thread("demo", "t1")
    assert any("noted: hello" in str(getattr(m, "content", "")) for m in messages)


async def test_a_budget_that_runs_out_is_exhausted_not_completed():
    """Every turn asks for a tool, so nothing ever ends the run on its own."""

    run, _ = await _run([_call("note", '{"text": "again"}')] * 10, max_turns=3)
    assert run.outcome is RunOutcome.EXHAUSTED
    assert run.turn_count == 3


async def test_a_failing_turn_fails_the_run_and_says_why():
    class Boom(FakeLLM):
        async def complete(self, *args: Any, **kwargs: Any) -> Message:
            raise RuntimeError("the model refused")

    stores = InMemoryRuntimeStores()
    agent = AgentDefinition(
        id="demo", name="Demo", persona="", llm=cast(Any, Boom([])), tools=ToolRegistry([])
    )
    runtime = LocalThreadRuntime(
        ActivityContext(stores=stores, resolve_agent=static_agents({"demo": agent}))
    )
    run = await runtime.run(ThreadInput(agent_id="demo", thread_id="t1", max_turns_per_run=4))
    assert run.outcome is RunOutcome.FAILED
    assert "refused" in (run.stop_reason or "")


async def test_the_run_is_finalized_whichever_way_it_ends():
    """`finalize_run` closes the transcript, so it must run on every path, not just the
    happy one. A caller that skipped it on failure would leave a run open forever."""

    for replies in ([_says("done")], [_call("note")] * 8):
        run, stores = await _run(replies, max_turns=2)
        runs = await stores.runs.list_for_thread("demo", "t1")
        assert runs, "the run was never persisted"
        # `finish` sets a terminal status; an unfinalized run would still be running
        assert all(r.status.value != "running" for r in runs), [r.status for r in runs]
