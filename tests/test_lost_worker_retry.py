"""A worker lost mid-activity: the run picks up where it left off.

A lost worker is simulated in-process: the first attempt does part of the work and then
hangs without heartbeating, which is all Temporal can see of a worker that died. The
workflow, retry policies and the real activities run the second attempt.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from actant import AgentDefinition, tool
from actant.llm.messages import ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import ActivityTimeouts, AgentRuntime, TemporalRuntimeConfig
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import ActivityContext, TemporalRuntimeActivities
from actant.runtime.temporal.activities.tools import TOOL_INTERRUPTED
from actant.runtime.temporal.types import ExecuteInput, ExecuteOutcome
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.types.threads import RunStatus
from actant.tools import ToolRegistry
from actant.tools.calls import ToolCallStatus
from runtime_fixtures import static_agents

#: Failure detection shortened so a lost attempt is noticed in a second.
_TIMEOUTS = ActivityTimeouts(tool_s=30, tool_heartbeat_s=1)


@pytest.mark.parametrize("retry_safe", [False, True])
async def test_a_tool_call_lost_with_its_worker_is_closed_or_rerun_and_the_run_goes_on(
    retry_safe: bool,
) -> None:
    effects: list[str] = []

    @tool(retry_safe=retry_safe)
    async def place() -> str:
        """Place the thing."""
        effects.append("placed")
        return "placed"

    fake = FakeLLM(
        [
            FakeResponse(
                tool_calls=[
                    ToolCall(id="call", function=ToolCallFunction(name="place", arguments="{}"))
                ]
            ),
            FakeResponse(text="done"),
        ]
    )
    agent = AgentDefinition(id="a", name="a", persona="", llm=fake, tools=ToolRegistry([place]))
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({"a": agent}))
    )

    @activity.defn(name="execute_tool")
    async def execution(payload: ExecuteInput) -> ExecuteOutcome:
        if activity.info().attempt == 1:
            effects.append("placed")  # the effect lands, then the worker dies
            await asyncio.Future()
        return await activities.tools.execute_tool(payload)

    registered = [fn for fn in activities.all if fn.__name__ != "execute_tool"]
    async with await WorkflowEnvironment.start_local() as env:
        config = TemporalRuntimeConfig(task_queue=uuid4().hex, activity_timeouts=_TIMEOUTS)
        async with Worker(
            env.client,
            task_queue=config.task_queue,
            workflows=[AgentThreadWorkflow],
            activities=[*registered, execution],
        ):
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message("a", "t", "go")
            await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 30)

    [run] = await stores.runs.list_for_thread("a", "t")
    assert run.status is RunStatus.IDLE
    record = await stores.tool_calls.get("call")
    messages = await stores.messages.list_for_thread("a", "t")
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert messages[-1].content == "done"
    if retry_safe:
        assert record.status is ToolCallStatus.COMPLETED
        assert effects == ["placed", "placed"]
    else:
        assert record.status is ToolCallStatus.FAILED
        assert effects == ["placed"]
        assert TOOL_INTERRUPTED in str(messages[2].content)
