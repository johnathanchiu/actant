"""Public runtime execution, recovery, and observer isolation against real Temporal."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment

from actant import AgentDefinition, tool
from actant.blocks import Base64Source, InlineImageBlock
from actant.core import JSONObject
from actant.llm.messages import ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime, TemporalRuntimeConfig
from actant.runtime.completion import RunCompletion
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.types.threads import RunStatus
from actant.tools import ToolRegistry, ToolResult


async def stop(task: asyncio.Task[None]) -> None:
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_client_only_runtime_rejects_polling_without_resolver() -> None:
    runtime = AgentRuntime(client=cast(Client, Mock()), stores=InMemoryRuntimeStores())
    with pytest.raises(ValueError, match="resolve_agent"):
        await runtime.run_worker()


@pytest.mark.asyncio
async def test_same_runtime_submits_runs_and_uses_worker_resolved_limits() -> None:
    stores = InMemoryRuntimeStores()
    fake = FakeLLM([FakeResponse(text="hello"), FakeResponse(text="again")])
    agent = AgentDefinition(
        id="a", name="a", persona="", llm=fake, tools=ToolRegistry([]), max_turns_per_thread=3
    )
    calls: list[tuple[str, str]] = []

    async def resolve(agent_id: str, thread_id: str) -> AgentDefinition:
        calls.append((agent_id, thread_id))
        return agent

    async with await WorkflowEnvironment.start_local() as env:
        runtime = AgentRuntime(
            client=env.client,
            stores=stores,
            event_sink=stores.publisher,
            resolve_agent=resolve,
            config=TemporalRuntimeConfig(task_queue=uuid4().hex),
        )
        polling = asyncio.create_task(runtime.run_worker())
        try:
            first = await runtime.send_message("a", "t", "hi")
            await asyncio.wait_for(env.client.get_workflow_handle(first).result(), 30)
            second = await runtime.send_message("a", "t", "hi again")
            await asyncio.wait_for(env.client.get_workflow_handle(second).result(), 30)
            assert first == second
            assert len(await runtime.thread("a", "t").messages()) == 4
            runs = await stores.runs.list_for_thread("a", "t")
            assert len(runs) == 2 and all(run.max_turns == 3 for run in runs)
            assert calls == [("a", "t")] * 4
            events = stores.publisher.events["thread:t"]
            data = [cast(JSONObject, event["data"]) for event in events]
            assert all(item.get("run_id") for item in data)
            turn_data = [
                cast(JSONObject, e["data"]) for e in events if e["type"] == "assistant_message"
            ]
            assert len({str(item["turn_id"]) for item in turn_data}) == 2
        finally:
            await stop(polling)


@pytest.mark.asyncio
async def test_approval_resumes_with_a_new_runtime_and_resolver() -> None:
    stores = InMemoryRuntimeStores()
    executed: list[int] = []

    @tool(approval="Approve execution?")
    async def perform(value: int) -> str:
        executed.append(value)
        return "done"

    fake = FakeLLM(
        [
            FakeResponse(
                tool_calls=[
                    ToolCall(
                        id="call",
                        function=ToolCallFunction(name="perform", arguments='{"value":7}'),
                    )
                ]
            ),
            FakeResponse(text="finished"),
        ]
    )
    agent = AgentDefinition(id="a", name="a", persona="", llm=fake, tools=ToolRegistry([perform]))
    first_resolutions: list[str] = []
    second_resolutions: list[str] = []

    async def resolve_first(agent_id: str, thread_id: str) -> AgentDefinition:
        first_resolutions.append(thread_id)
        return agent

    async def resolve_second(agent_id: str, thread_id: str) -> AgentDefinition:
        second_resolutions.append(thread_id)
        return agent

    async with await WorkflowEnvironment.start_local() as env:
        config = TemporalRuntimeConfig(task_queue=uuid4().hex)
        api = AgentRuntime(client=env.client, stores=stores, config=config)
        first = AgentRuntime(
            client=env.client, stores=stores, config=config, resolve_agent=resolve_first
        )
        polling = asyncio.create_task(first.run_worker())
        try:
            workflow_id = await api.send_message("a", "child", "go", parent_thread_id="parent")
            async with asyncio.timeout(30):
                while not await api.thread("a", "child").waiting_tools():
                    await asyncio.sleep(0.02)
        finally:
            await stop(polling)
        assert executed == []
        second = AgentRuntime(
            client=env.client, stores=stores, config=config, resolve_agent=resolve_second
        )
        polling = asyncio.create_task(second.run_worker())
        try:
            await api.resolve_tool_call("a", "child", "call", approved=True)
            await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 30)
            await api.resolve_tool_call("a", "child", "call", approved=True)
            assert executed == [7]
            assert first_resolutions and second_resolutions
            assert (await stores.threads.get("a", "child")).parent_thread_id == "parent"
            messages = await api.thread("a", "child").messages()
            assert sum(m.role == "tool" for m in messages) == 1
        finally:
            await stop(polling)


@pytest.mark.asyncio
async def test_observer_failure_does_not_fail_or_repeat_model_work() -> None:
    class BadSink:
        async def publish(self, channel: str, event: JSONObject) -> None:
            raise RuntimeError("event transport unavailable")

    stores = InMemoryRuntimeStores()
    fake = FakeLLM([FakeResponse(text="answer", text_chunks=["answer"])])
    agent = AgentDefinition(id="a", name="a", persona="", llm=fake, tools=ToolRegistry([]))

    async def resolve(agent_id: str, thread_id: str) -> AgentDefinition:
        return agent

    completions: list[str] = []

    async def complete(completion: RunCompletion) -> None:
        completions.append(completion.run_id)
        if len(completions) == 1:
            raise RuntimeError("retry durable callback")

    async with await WorkflowEnvironment.start_local() as env:
        runtime = AgentRuntime(
            client=env.client,
            stores=stores,
            config=TemporalRuntimeConfig(task_queue=uuid4().hex),
            resolve_agent=resolve,
            event_sink=BadSink(),
            run_completion_handler=complete,
        )
        polling = asyncio.create_task(runtime.run_worker())
        try:
            workflow_id = await runtime.send_message("a", "t", "hello")
            await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 30)
            assert len(await runtime.thread("a", "t").messages()) == 2
            [run] = await stores.runs.list_for_thread("a", "t")
            assert run.status is RunStatus.IDLE and run.turn_count == 1
            assert len(completions) == 2 and len(set(completions)) == 1
        finally:
            await stop(polling)


@pytest.mark.asyncio
async def test_unknown_agent_finishes_failed_and_notifies_without_running_model() -> None:
    stores = InMemoryRuntimeStores()

    async def resolve(agent_id: str, thread_id: str) -> AgentDefinition:
        raise KeyError(agent_id)

    completions: list[RunCompletion] = []

    async def complete(completion: RunCompletion) -> None:
        completions.append(completion)

    async with await WorkflowEnvironment.start_local() as env:
        runtime = AgentRuntime(
            client=env.client,
            stores=stores,
            config=TemporalRuntimeConfig(task_queue=uuid4().hex),
            resolve_agent=resolve,
            run_completion_handler=complete,
        )
        polling = asyncio.create_task(runtime.run_worker())
        try:
            workflow_id = await runtime.send_message("missing", "t", "hello")
            await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 30)
            [run] = await stores.runs.list_for_thread("missing", "t")
            assert run.status is RunStatus.FAILED
            assert run.stop_reason and "Unknown agent" in run.stop_reason
            assert len(completions) == 1 and completions[0].outcome == "failed"
        finally:
            await stop(polling)


@pytest.mark.asyncio
async def test_tool_asset_is_resolved_for_model_but_stored_as_reference() -> None:
    from actant.assets import AssetContext, AssetReference, MissingAsset, ResolvedImage

    block = AssetReference("images/tool.png", "image/png").to_block()

    @tool
    async def picture() -> ToolResult:
        return ToolResult(output="picture", content_blocks=[block])

    class Resolver:
        async def resolve(
            self, asset: AssetReference, context: AssetContext
        ) -> ResolvedImage | MissingAsset:
            assert asset.storage_key == "images/tool.png"
            assert context.thread_id == "t"
            return ResolvedImage("image/png", data=b"png")

    fake = FakeLLM(
        [
            FakeResponse(
                tool_calls=[
                    ToolCall(
                        id="picture-call",
                        function=ToolCallFunction(name="picture", arguments="{}"),
                    )
                ]
            ),
            FakeResponse(text="seen"),
        ]
    )
    agent = AgentDefinition(id="a", name="a", persona="", llm=fake, tools=ToolRegistry([picture]))

    async def resolve(agent_id: str, thread_id: str) -> AgentDefinition:
        return agent

    stores = InMemoryRuntimeStores()
    async with await WorkflowEnvironment.start_local() as env:
        runtime = AgentRuntime(
            client=env.client,
            stores=stores,
            event_sink=stores.publisher,
            resolve_agent=resolve,
            assets=Resolver(),
            config=TemporalRuntimeConfig(task_queue=uuid4().hex),
        )
        polling = asyncio.create_task(runtime.run_worker())
        try:
            workflow_id = await runtime.send_message("a", "t", "look")
            await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 30)
            stored = await runtime.thread("a", "t").messages()
            assert next(m for m in stored if m.role == "tool").content == [block]
            sent = next(m for m in fake.calls[1][1] if m.role == "tool")
            assert sent.content == [
                InlineImageBlock(source=Base64Source(media_type="image/png", data="cG5n"))
            ]
        finally:
            await stop(polling)


async def test_shutdown_waits_for_active_tool_and_returns_from_run_worker() -> None:
    from temporalio import activity

    started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    @tool
    async def finish_during_shutdown() -> str:
        started.set()
        await activity.wait_for_worker_shutdown()
        stopping.set()
        await release.wait()
        return "finished within grace"

    agent = AgentDefinition(
        id="a",
        name="a",
        persona="",
        tools=ToolRegistry([finish_during_shutdown]),
        llm=FakeLLM(
            [
                FakeResponse(
                    tool_calls=[
                        ToolCall(
                            id="c",
                            function=ToolCallFunction(
                                name="finish_during_shutdown", arguments="{}"
                            ),
                        )
                    ]
                )
            ]
        ),
    )

    async def resolve(agent_id: str, thread_id: str) -> AgentDefinition:
        return agent

    async with await WorkflowEnvironment.start_local() as env:
        stores = InMemoryRuntimeStores()
        runtime = AgentRuntime(
            client=env.client,
            stores=stores,
            event_sink=stores.publisher,
            resolve_agent=resolve,
            config=TemporalRuntimeConfig(
                task_queue=uuid4().hex,
                graceful_shutdown_timeout_seconds=10,
            ),
        )
        await runtime.shutdown()  # No worker yet.
        polling = asyncio.create_task(runtime.run_worker())
        shutdown: asyncio.Task[None] | None = None
        try:
            await runtime.send_message("a", "t", "go")
            await asyncio.wait_for(started.wait(), 20)
            shutdown = asyncio.create_task(runtime.shutdown())
            await asyncio.wait_for(stopping.wait(), 10)
            assert not shutdown.done() and not polling.done()
            release.set()
            await asyncio.wait_for(shutdown, 20)
            await asyncio.wait_for(polling, 20)
            record = await stores.tool_calls.get("c")
            assert record.result == {"tool_call_id": "c", "result": "finished within grace"}
            await runtime.shutdown()  # Already stopped.
        finally:
            release.set()
            if shutdown is not None:
                await shutdown
            await stop(polling)
