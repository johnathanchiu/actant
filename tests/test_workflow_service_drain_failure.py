"""An uncertain remote drain must reach the parent as infrastructure failure."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest
from runtime_fixtures import static_agents
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from actant import AgentDefinition, tool
from actant.heartbeat import heartbeating
from actant.llm.messages import ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import ActivityContext, TemporalRuntimeActivities
from actant.runtime.temporal.types import ExecuteInput, ExecuteOutcome, InboundMessage, ThreadInput
from actant.runtime.temporal.workflow import (
    AgentThreadWorkflow,
    _drain_cancelled,
    _service_drain_failed,
)
from actant.tools import ToolRegistry
from actant.tools.calls import ToolCallStatus


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("phase", ["admit_tool", "execute_tool"])
async def test_service_drain_failure_escapes_thread_result_and_cancellation(cancel, phase):
    started, cleaning, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @tool
    async def work() -> str:
        return "unused"

    agent = AgentDefinition(
        id="author",
        name="author",
        persona="",
        llm=FakeLLM(
            [
                FakeResponse(
                    tool_calls=[
                        ToolCall(id="call", function=ToolCallFunction(name="work", arguments="{}"))
                    ]
                )
            ]
        ),
        tools=ToolRegistry([work]),
    )
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({agent.id: agent}))
    )

    @activity.defn(name=phase)
    async def execute(payload: ExecuteInput) -> ExecuteOutcome:
        started.set()
        try:
            async with heartbeating(every_s=0.01):
                await release.wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
        raise ApplicationError(
            "remote drain unconfirmed", type="ServiceDrainError", non_retryable=True
        )

    registered = [method for method in activities.all if method.__name__ != phase]
    async with await WorkflowEnvironment.start_local() as environment:
        queue = uuid4().hex
        async with Worker(
            environment.client,
            task_queue=queue,
            workflows=[AgentThreadWorkflow],
            activities=[*registered, execute],
            max_heartbeat_throttle_interval=timedelta(milliseconds=10),
            default_heartbeat_throttle_interval=timedelta(milliseconds=10),
        ):
            handle = await environment.client.start_workflow(
                AgentThreadWorkflow.run,
                ThreadInput(agent.id, "thread", max_turns_per_run=1),
                id=uuid4().hex,
                task_queue=queue,
                start_signal="inbound",
                start_signal_args=[InboundMessage(content="build")],
            )
            result = asyncio.create_task(handle.result())
            try:
                await asyncio.wait_for(started.wait(), 10)
                if cancel:
                    await handle.cancel()
                    await asyncio.wait_for(cleaning.wait(), 10)
                    await handle.cancel()
                    with pytest.raises(TimeoutError):
                        await asyncio.wait_for(asyncio.shield(result), 0.1)
                release.set()
                with pytest.raises(WorkflowFailureError) as failed:
                    await asyncio.wait_for(result, 10)
                assert _service_drain_failed(failed.value)
                record = await stores.tool_calls.get("call")
                expected = (
                    ToolCallStatus.RUNNING if phase == "execute_tool" else ToolCallStatus.REQUESTED
                )
                assert record.status == expected and record.result is None
                await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(
                    await handle.fetch_history()
                )
            finally:
                release.set()
                await handle.cancel()
                await asyncio.gather(result, return_exceptions=True)


async def test_repeated_cancellation_does_not_drop_drain_failure():
    cleaning, release = asyncio.Event(), asyncio.Event()

    async def effect():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            await release.wait()
            raise ApplicationError("drain failed", type="ServiceDrainError", non_retryable=True)

    worker = asyncio.create_task(effect())
    await asyncio.sleep(0)
    drain = asyncio.create_task(_drain_cancelled([worker]))
    await asyncio.wait_for(cleaning.wait(), 1)
    drain.cancel()
    await asyncio.sleep(0)
    drain.cancel()
    await asyncio.sleep(0)
    assert not drain.done()
    release.set()
    with pytest.raises(ApplicationError, match="drain failed"):
        await drain
