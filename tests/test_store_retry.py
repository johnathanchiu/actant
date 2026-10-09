"""A run whose store write keeps failing ends the thread instead of retrying forever."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy.exc import ProgrammingError
from temporalio import activity
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from actant import AgentDefinition
from actant.llm.providers.fake import FakeLLM
from actant.runtime import AgentRuntime, TemporalRuntimeConfig
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import ActivityContext, TemporalRuntimeActivities
from actant.runtime.temporal.types import ActivityName, StartedRun, StartRunInput
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.tools import ToolRegistry
from runtime_fixtures import static_agents


@pytest.mark.parametrize(
    ("error", "attempts"),
    [
        # a schema behind its migrations fails the same way every time
        (ProgrammingError("SELECT", {}, Exception('column "x" does not exist')), 1),
        (ConnectionError("database unavailable"), 5),
    ],
)
async def test_start_run_failures_stop_at_the_cap(error: Exception, attempts: int) -> None:
    agent = AgentDefinition(id="a", name="a", persona="", llm=FakeLLM([]), tools=ToolRegistry([]))
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({"a": agent}))
    )
    seen: list[int] = []

    @activity.defn(name=ActivityName.START_RUN)
    async def start_run(payload: StartRunInput) -> StartedRun:
        seen.append(activity.info().attempt)
        raise error

    registered = [fn for fn in activities.all if fn.__name__ != "start_run"]
    async with await WorkflowEnvironment.start_local() as env:
        config = TemporalRuntimeConfig(task_queue=uuid4().hex)
        async with Worker(
            env.client,
            task_queue=config.task_queue,
            workflows=[AgentThreadWorkflow],
            activities=[*registered, start_run],
        ):
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message("a", "t", "go")
            with pytest.raises(WorkflowFailureError):
                await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 60)

    assert seen == list(range(1, attempts + 1))
