"""``AgentRuntime.spawn``: a thread receives its first message once."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from actant.agents import AgentDefinition
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import TemporalRuntimeActivities
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import TemporalRuntimeConfig
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.tools.registry import ToolRegistry
from runtime_fixtures import static_agents

_AGENT = "author"
_THREAD = "author-chair-1"
_PARENT = "planner"


def _agent(replies: int) -> AgentDefinition:
    return AgentDefinition(
        id=_AGENT,
        name="author",
        persona="test persona",
        llm=FakeLLM([FakeResponse(text=f"reply {i}") for i in range(replies)]),
        tools=ToolRegistry([]),
        tool_allowlist=set(),
    )


async def _wait_for(predicate: Callable[[], Awaitable[bool]], timeout: float = 30.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("predicate did not become true within timeout")


async def _with_runtime(
    stores: InMemoryRuntimeStores,
    agent: AgentDefinition,
    test: Callable[[AgentRuntime], Awaitable[None]],
) -> None:
    task_queue = f"test-spawn-{uuid.uuid4().hex[:8]}"
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({agent.id: agent}))
    )
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            await test(
                AgentRuntime(
                    client=env.client,
                    stores=stores,
                    config=TemporalRuntimeConfig(task_queue=task_queue),
                )
            )


async def _user_messages(stores: InMemoryRuntimeStores) -> list[str]:
    messages = await stores.messages.list_for_thread(_AGENT, _THREAD)
    return [str(m.content) for m in messages if m.role == "user"]


async def _answered(stores: InMemoryRuntimeStores) -> bool:
    messages = await stores.messages.list_for_thread(_AGENT, _THREAD)
    return any(m.role == "assistant" for m in messages)


@pytest.mark.asyncio
async def test_spawn_twice_while_starting_delivers_one_brief() -> None:
    stores = InMemoryRuntimeStores()

    async def body(runtime: AgentRuntime) -> None:
        first, second = await asyncio.gather(
            runtime.spawn(_AGENT, _THREAD, "brief", parent_thread_id=_PARENT),
            runtime.spawn(_AGENT, _THREAD, "brief", parent_thread_id=_PARENT),
        )
        assert sorted([first, second]) == [False, True]
        await _wait_for(lambda: _answered(stores))
        assert await _user_messages(stores) == ["brief"]
        thread = await stores.threads.get(_AGENT, _THREAD)
        assert thread.parent_thread_id == _PARENT

    await _with_runtime(stores, _agent(replies=2), body)


@pytest.mark.asyncio
async def test_spawn_after_the_thread_ran_is_ignored() -> None:
    stores = InMemoryRuntimeStores()

    async def body(runtime: AgentRuntime) -> None:
        assert await runtime.spawn(_AGENT, _THREAD, "brief", parent_thread_id=_PARENT)
        await _wait_for(lambda: _answered(stores))
        assert not await runtime.spawn(_AGENT, _THREAD, "brief", parent_thread_id=_PARENT)
        await asyncio.sleep(0.5)
        assert await _user_messages(stores) == ["brief"]

    await _with_runtime(stores, _agent(replies=2), body)


@pytest.mark.asyncio
async def test_spawn_is_refused_by_the_stores_after_temporal_forgets() -> None:
    """A fresh Temporal (a wiped server, a new worker) still sees the stored thread."""
    stores = InMemoryRuntimeStores()

    async def first(runtime: AgentRuntime) -> None:
        assert await runtime.spawn(_AGENT, _THREAD, "brief", parent_thread_id=_PARENT)
        await _wait_for(lambda: _answered(stores))

    async def again(runtime: AgentRuntime) -> None:
        assert not await runtime.spawn(_AGENT, _THREAD, "brief", parent_thread_id=_PARENT)

    await _with_runtime(stores, _agent(replies=2), first)
    await _with_runtime(stores, _agent(replies=2), again)
    assert await _user_messages(stores) == ["brief"]


@pytest.mark.asyncio
async def test_spawn_without_once_sends_every_time() -> None:
    stores = InMemoryRuntimeStores()

    async def body(runtime: AgentRuntime) -> None:
        assert await runtime.spawn(_AGENT, _THREAD, "one", once=False)
        await _wait_for(lambda: _answered(stores))
        assert await runtime.spawn(_AGENT, _THREAD, "two", once=False)
        await _wait_for(lambda: _count_user(stores, 2))
        assert await _user_messages(stores) == ["one", "two"]

    await _with_runtime(stores, _agent(replies=2), body)


async def _count_user(stores: InMemoryRuntimeStores, n: int) -> bool:
    return len(await _user_messages(stores)) >= n
