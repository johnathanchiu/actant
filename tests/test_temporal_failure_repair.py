"""Failures outside activity bodies must not strand projections or orphan tool calls."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from temporalio import activity
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from actant import AgentDefinition, tool
from actant.llm.messages import ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime, TemporalRuntimeConfig
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import ActivityContext, TemporalRuntimeActivities
from actant.runtime.temporal.types import AdmitInput, AdmitOutcome, ExecuteInput, ExecuteOutcome
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.types.threads import RunStatus, ThreadStatus
from actant.tools import ToolRegistry
from actant.tools.calls import ToolCallStatus
from runtime_fixtures import static_agents


@pytest.mark.parametrize("boundary", ["admit", "execute"])
async def test_temporal_activity_failure_drains_siblings_and_repairs_transcript(
    boundary: str,
) -> None:
    calls: list[str] = []

    @tool
    async def echo(value: str) -> str:
        calls.append(value)
        await asyncio.sleep(0.02)
        return value

    fake = FakeLLM(
        [
            FakeResponse(
                tool_calls=[
                    ToolCall(
                        id="broken",
                        function=ToolCallFunction(name="echo", arguments='{"value":"broken"}'),
                    ),
                    ToolCall(
                        id="ok", function=ToolCallFunction(name="echo", arguments='{"value":"ok"}')
                    ),
                ]
            ),
            FakeResponse(text="recovered"),
        ]
    )
    agent = AgentDefinition(id="a", name="a", persona="", llm=fake, tools=ToolRegistry([echo]))
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({"a": agent}))
    )
    failures: list[str] = []

    @activity.defn(name="admit_tool")
    async def admission(payload: AdmitInput) -> AdmitOutcome:
        if boundary == "admit" and payload.tool_call_id == "broken":
            failures.append("admit")
            raise ApplicationError("admission worker unavailable", non_retryable=True)
        return await activities.tools.admit_tool(payload)

    @activity.defn(name="execute_tool")
    async def execution(payload: ExecuteInput) -> ExecuteOutcome:
        if boundary == "execute" and payload.tool_call_id == "broken":
            failures.append("execute")
            raise ApplicationError("tool transport outcome uncertain", non_retryable=True)
        return await activities.tools.execute_tool(payload)

    registered = [fn for fn in activities.all if fn.__name__ not in {"admit_tool", "execute_tool"}]
    async with await WorkflowEnvironment.start_local() as env:
        config = TemporalRuntimeConfig(task_queue=uuid4().hex)
        async with Worker(
            env.client,
            task_queue=config.task_queue,
            workflows=[AgentThreadWorkflow],
            activities=[*registered, admission, execution],
        ):
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message("a", "t", "go")
            await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 30)
            [run] = await stores.runs.list_for_thread("a", "t")
            assert run.status is RunStatus.FAILED and run.stop_reason
            assert (await stores.threads.get("a", "t")).status is ThreadStatus.FAILED
            assert not await stores.tool_calls.get_open_for_thread("a", "t")
            assert failures == [boundary]
            assert calls == (["ok"] if boundary == "execute" else [])
            assert (await stores.tool_calls.get("broken")).status is ToolCallStatus.FAILED
            messages = await runtime.thread("a", "t").messages()
            assert [m.role for m in messages] == ["user", "assistant", "tool", "tool"]
            assert len(fake.calls) == 1
            # A subsequent run can use the repaired transcript without orphan calls.
            await runtime.send_message("a", "t", "continue")
            await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 30)
            assert (await runtime.thread("a", "t").messages())[-1].content == "recovered"


async def test_late_tool_completion_preserves_repaired_result() -> None:
    """A timed-out worker can survive and finish after another worker repairs its run."""
    from actant.llm.messages import Message
    from actant.core import JSONObject
    from actant.runtime.temporal.types import FinalizeRunInput
    from actant.tools.calls import ToolCallRecord

    started, release = asyncio.Event(), asyncio.Event()
    effects: list[str] = []
    events: list[JSONObject] = []

    class Sink:
        async def publish(self, channel: str, event: JSONObject) -> None:
            if event["type"] == "tool_result":
                events.append(event)

    @tool
    async def slow() -> str:
        effects.append("once")
        started.set()
        await release.wait()
        return "late success"

    agent = AgentDefinition(
        id="a", name="a", persona="", llm=FakeLLM([]), tools=ToolRegistry([slow])
    )
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(
            stores=stores,
            resolve_agent=static_agents({"a": agent}),
            event_sink=Sink(),
        )
    )
    await stores.threads.get_or_create("a", "t")
    await stores.runs.create("a", "t", run_id="r", max_turns=1)
    record = ToolCallRecord(
        id="c",
        group_id="g",
        run_id="r",
        agent_id="a",
        thread_id="t",
        turn_id="turn",
        turn_index=1,
        name="slow",
        args={},
        status=ToolCallStatus.RUNNING,
    )
    await stores.messages.append_assistant_with_tool_calls(
        "a",
        "t",
        "turn",
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c", function=ToolCallFunction(name="slow", arguments="{}"))],
        ),
        [record],
    )
    task = asyncio.create_task(
        activities.tools.execute_tool(
            ExecuteInput(agent_id="a", thread_id="t", run_id="r", tool_call_id="c")
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        await activities.runs.finalize_run(
            FinalizeRunInput(
                agent_id="a",
                thread_id="t",
                run_id="r",
                outcome="failed",
                turn_count=1,
                stop_reason="heartbeat timeout",
            )
        )
        repaired = (await stores.tool_calls.get("c")).result
        release.set()
        outcome = await asyncio.wait_for(task, 2)
        assert outcome.status == "failed"
        assert (await stores.tool_calls.get("c")).status is ToolCallStatus.FAILED
        assert (await stores.tool_calls.get("c")).result == repaired
        results = [m for m in await stores.messages.list_for_thread("a", "t") if m.role == "tool"]
        assert len(results) == 1 and "heartbeat timeout" in str(results[0].content)
        assert events == []
        assert effects == ["once"]
    finally:
        release.set()
        await task
