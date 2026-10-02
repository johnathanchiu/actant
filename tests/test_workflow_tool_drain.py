"""Cancelled threads wait for every executing tool's cleanup before finalizing."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from uuid import uuid4

import pytest
from runtime_fixtures import static_agents
from temporalio import activity, workflow
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from actant import AgentDefinition, tool
from actant.heartbeat import heartbeating
from actant.llm.messages import ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import ActivityContext, TemporalRuntimeActivities
from actant.runtime.temporal.types import ExecuteInput, ExecuteOutcome, InboundMessage, ThreadInput
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.tools import ToolRegistry


async def test_workflow_cancellation_drains_every_executing_tool_before_finalization() -> None:
    started: set[str] = set()
    cleaning: set[str] = set()
    cleaned: set[str] = set()
    both_started = asyncio.Event()
    both_cleaning = asyncio.Event()
    release_cleanup = asyncio.Event()

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
                        ToolCall(
                            id=identifier, function=ToolCallFunction(name="work", arguments="{}")
                        )
                        for identifier in ("first", "second")
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

    @activity.defn(name="execute_tool")
    async def execute(payload: ExecuteInput) -> ExecuteOutcome:
        started.add(payload.tool_call_id)
        if len(started) == 2:
            both_started.set()
        try:
            async with heartbeating(every_s=0.01):
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.add(payload.tool_call_id)
            if len(cleaning) == 2:
                both_cleaning.set()
            await release_cleanup.wait()
            cleaned.add(payload.tool_call_id)
            raise
        raise AssertionError("tool must be cancelled")

    registered = [method for method in activities.all if method.__name__ != "execute_tool"]
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
            result = None
            try:
                await asyncio.wait_for(both_started.wait(), 10)
                await handle.cancel()
                result = asyncio.create_task(handle.result())
                await asyncio.wait_for(both_cleaning.wait(), 10)
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(asyncio.shield(result), 0.1)
                assert not cleaned
                release_cleanup.set()
                with pytest.raises(WorkflowFailureError):
                    await asyncio.wait_for(result, 10)
                assert cleaned == {"first", "second"}
                assert not await stores.tool_calls.get_open_for_thread(agent.id, "thread")
                await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(
                    await handle.fetch_history()
                )
            finally:
                release_cleanup.set()
                await handle.cancel()
                if result is not None:
                    await asyncio.gather(result, return_exceptions=True)


async def test_pre_drain_patch_history_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    """New cancellation commands are absent while replaying older tool histories."""
    from test_workflow_thread import _agent, _EchoTool, _tool_call

    agent = _agent(
        FakeLLM([FakeResponse(tool_calls=[_tool_call("echo")]), FakeResponse(text="done")]),
        tools=[_EchoTool()],
    )
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({agent.id: agent}))
    )
    original = workflow.patched

    def before_patch(identifier: str) -> bool:
        return False if identifier == "tool-cancellation-drain-v1" else original(identifier)

    async with await WorkflowEnvironment.start_local() as environment:
        queue = uuid4().hex
        with monkeypatch.context() as prior:
            prior.setattr(workflow, "patched", before_patch)
            async with Worker(
                environment.client,
                task_queue=queue,
                workflows=[AgentThreadWorkflow],
                activities=activities.all,
            ):
                handle = await environment.client.start_workflow(
                    AgentThreadWorkflow.run,
                    ThreadInput(agent.id, "thread", max_turns_per_run=2),
                    id=uuid4().hex,
                    task_queue=queue,
                    start_signal="inbound",
                    start_signal_args=[InboundMessage(content="echo")],
                )
                await asyncio.wait_for(handle.result(), 10)
                history = await handle.fetch_history()
        await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)
