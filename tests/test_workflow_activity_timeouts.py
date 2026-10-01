"""``TemporalRuntimeConfig.activity_timeouts`` sets each activity's time limits."""

from __future__ import annotations

import asyncio
import uuid

import pytest
from temporalio.api.enums.v1 import EventType
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import TemporalRuntimeActivities
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import ActivityTimeouts, TemporalRuntimeConfig
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from runtime_fixtures import static_agents
from test_workflow_thread import _AGENT, _THREAD, _agent, _EchoTool, _tool_call


async def test_each_activity_is_scheduled_with_the_configured_timeouts() -> None:
    timeouts = ActivityTimeouts(turn_s=301, compact_s=302, tool_s=1900, tool_heartbeat_s=95)
    agent = _agent(
        FakeLLM([FakeResponse(tool_calls=[_tool_call("echo")]), FakeResponse(text="done")]),
        tools=[_EchoTool()],
    )
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({agent.id: agent}))
    )
    task_queue = f"test-actant-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            config = TemporalRuntimeConfig(
                task_queue=task_queue,
                address=env.client.service_client.config.target_host,
                namespace=env.client.namespace,
                activity_timeouts=timeouts,
            )
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message(_AGENT, _THREAD, "run echo")
            handle = env.client.get_workflow_handle(workflow_id)
            await asyncio.wait_for(handle.result(), timeout=20.0)
            history = await handle.fetch_history()

    scheduled: dict[str, tuple[int, int]] = {}
    for event in history.events:
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED:
            attributes = event.activity_task_scheduled_event_attributes
            scheduled[attributes.activity_type.name] = (
                attributes.start_to_close_timeout.seconds,
                attributes.heartbeat_timeout.seconds,
            )
    turns = [limit for name, limit in scheduled.items() if "turn" in name]
    assert turns and all(start == 301 for start, _ in turns)
    tools = [limit for name, limit in scheduled.items() if "execute" in name]
    assert tools == [(1900, 95)]


def test_activity_timeouts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        ActivityTimeouts(tool_s=0)
