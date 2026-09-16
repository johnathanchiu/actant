"""Disposable worker process for the Postgres crash-recovery drill."""

from __future__ import annotations

import asyncio
from datetime import timedelta
import os
from pathlib import Path
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from temporalio.client import Client
from temporalio.worker import UnsandboxedWorkflowRunner, Worker

from actant import AgentDefinition, tool
from actant.llm.messages import ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime.stores.postgres import SQLAlchemyRuntimeStores
from actant.runtime.temporal.activities import ActivityContext, TemporalRuntimeActivities
from actant.runtime.temporal import workflow as workflow_module
from actant.tools import ToolRegistry
from runtime_fixtures import static_agents


async def main(address: str, queue: str, schema: str, effects: str) -> None:
    engine = create_async_engine(
        os.environ["ACTANT_TEST_POSTGRES_URL"],
        execution_options={"schema_translate_map": {None: schema}},
    )
    stores = SQLAlchemyRuntimeStores(async_sessionmaker(engine, expire_on_commit=False))

    @tool
    async def perform() -> str:
        with Path(effects).open("a") as stream:
            stream.write("effect\n")
        await asyncio.Future()  # Parent kills this process before a result can be saved.
        return "unreachable"

    agent = AgentDefinition(
        id="a",
        name="a",
        persona="",
        tools=ToolRegistry([perform]),
        llm=FakeLLM(
            [
                FakeResponse(
                    tool_calls=[
                        ToolCall(
                            id="effect", function=ToolCallFunction(name="perform", arguments="{}")
                        )
                    ]
                )
            ]
        ),
    )
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({"a": agent}))
    )
    # Only shorten failure detection. Execute the production workflow/activity bodies
    # and retry policy. Unsandboxed runner preserves these process-local test constants.
    workflow_module._TOOL_HEARTBEAT_TIMEOUT = timedelta(seconds=3)
    workflow_module._TOOL_TIMEOUT = timedelta(seconds=10)
    try:
        async with Worker(
            await Client.connect(address),
            task_queue=queue,
            workflows=[workflow_module.AgentThreadWorkflow],
            activities=activities.all,
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            await asyncio.Future()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:]))
