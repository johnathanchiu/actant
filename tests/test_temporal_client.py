"""TemporalRuntimeClient integration tests."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from actant.agents import AgentDefinition
from actant.core import JSONObject, new_id
from actant.llm.messages import ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime.temporal.activities import TemporalRuntimeActivities
from actant.runtime.temporal.client import TemporalRuntimeClient
from actant.runtime.temporal.types import TemporalRuntimeConfig
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools.admission import (
    ToolCallView,
    ToolDecision,
    ToolResolution,
    ToolWaitRequest,
    TurnContextView,
)
from actant.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    Tool,
    ToolResult,
    make_tool_schema,
)
from actant.tools.calls import ToolCallStatus
from actant.tools.registry import ToolRegistry
from actant.tools import tool

_AGENT = "test_agent"
_THREAD = "test_thread"


def _agent(llm: FakeLLM, tools: list[Tool] | None = None) -> AgentDefinition:
    return AgentDefinition(
        id=_AGENT,
        name="test",
        persona="test persona",
        llm=llm,
        tools=ToolRegistry(tools or []),
        tool_allowlist={t.name for t in (tools or [])},
    )


def _tool_call(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=new_id("tc"), function=ToolCallFunction(name=name, arguments=args))


async def _wait_for(
    predicate: Callable[[], Awaitable[bool]] | Callable[[], bool],
    *,
    timeout: float = 10.0,
    poll: float = 0.05,
) -> None:
    start = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start < timeout:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        await asyncio.sleep(poll)
    raise TimeoutError("predicate did not become true within timeout")


@dataclass
class _RunSetup:
    env: WorkflowEnvironment
    stores: InMemoryRuntimeStores
    runtime: TemporalRuntimeClient
    task_queue: str


async def _run(
    test: Callable[[_RunSetup], Awaitable[None]],
    *,
    agent: AgentDefinition,
) -> None:
    stores = InMemoryRuntimeStores()
    task_queue = f"test-client-{uuid.uuid4().hex[:8]}"
    activities = TemporalRuntimeActivities(stores=stores, agents={agent.id: agent})

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            runtime = TemporalRuntimeClient(
                stores=stores,
                agents={agent.id: agent},
                config=TemporalRuntimeConfig(
                    address=env.client.service_client.config.target_host,
                    namespace=env.client.namespace,
                    task_queue=task_queue,
                ),
            )
            await test(_RunSetup(env=env, stores=stores, runtime=runtime, task_queue=task_queue))


@pytest.mark.asyncio
async def test_temporal_client_send_message_signals_existing_workflow() -> None:
    agent = _agent(FakeLLM([FakeResponse(text="one"), FakeResponse(text="two")]))

    async def body(s: _RunSetup) -> None:
        await s.runtime.send_message(_AGENT, _THREAD, "first")

        async def one_assistant() -> bool:
            messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in messages if m.role == "assistant") >= 1

        await _wait_for(one_assistant)
        await s.runtime.send_message(_AGENT, _THREAD, "second")

        async def two_assistants() -> bool:
            messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in messages if m.role == "assistant") >= 2

        await _wait_for(two_assistants)
        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        assert [(m.role, m.content) for m in messages] == [
            ("user", "first"),
            ("assistant", "one"),
            ("user", "second"),
            ("assistant", "two"),
        ]

        await s.runtime.cancel_thread(_AGENT, _THREAD)

    await _run(body, agent=agent)


class _ApprovalInvocation(BaseToolInvocation[JSONObject, object]):
    async def execute(self) -> ToolResult:  # pragma: no cover - WAIT only
        return ToolResult.ok({"never": True})


class _ApprovalTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        super().__init__("needs_approval", make_tool_schema("needs_approval", "Needs approval"))

    async def can_execute(
        self,
        call: ToolCallView,
        invocation: object,
        context: TurnContextView,
    ) -> ToolDecision:
        del call, invocation, context
        return ToolDecision.wait(ToolWaitRequest(kind="approval", prompt="approve?", payload={}))

    async def on_resolve(
        self,
        call: ToolCallView,
        resolution: ToolResolution,
    ) -> ToolResult:
        return ToolResult.ok(
            {
                "resolved_call_id": call.id,
                "approved": resolution.approved,
                "answer": resolution.answer,
                "payload": resolution.payload,
            }
        )

    async def build(self, params: JSONObject) -> _ApprovalInvocation:
        return _ApprovalInvocation(params)


@pytest.mark.asyncio
async def test_temporal_client_resolves_deferred_tool_call() -> None:
    tool_call = _tool_call("needs_approval")
    agent = _agent(
        FakeLLM([FakeResponse(tool_calls=[tool_call]), FakeResponse(text="approved!")]),
        tools=[_ApprovalTool()],
    )

    async def body(s: _RunSetup) -> None:
        await s.runtime.send_message(_AGENT, _THREAD, "please approve")

        async def is_waiting() -> bool:
            try:
                record = await s.stores.tool_calls.get(tool_call.id)
            except KeyError:
                return False
            return record.status == ToolCallStatus.WAITING

        await _wait_for(is_waiting)
        await s.runtime.resolve_tool_call(
            _AGENT,
            _THREAD,
            tool_call.id,
            approved=True,
            answer="ok",
            payload={"source": "test"},
        )

        async def post_resolution_assistant() -> bool:
            messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in messages if m.role == "assistant") >= 2

        await _wait_for(post_resolution_assistant)
        record = await s.stores.tool_calls.get(tool_call.id)
        assert record.status == ToolCallStatus.COMPLETED
        assert record.result == {
            "result": {
                "resolved_call_id": tool_call.id,
                "approved": True,
                "answer": "ok",
                "payload": {"source": "test"},
            },
            "tool_call_id": tool_call.id,
        }

        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
        assert messages[-1].content == "approved!"

        await s.runtime.cancel_thread(_AGENT, _THREAD)

    await _run(body, agent=agent)


@pytest.mark.asyncio
async def test_function_tool_approval_uses_the_existing_deferred_workflow() -> None:
    tool_call = _tool_call("publish", '{"title":"Launch"}')
    publications: list[str] = []

    @tool(approval="Publish {title}?")
    async def publish(title: str) -> dict[str, str]:
        """Publish an update."""
        publications.append(title)
        return {"published": title}

    agent = _agent(
        FakeLLM([FakeResponse(tool_calls=[tool_call]), FakeResponse(text="published")]),
        tools=[publish],
    )

    async def body(s: _RunSetup) -> None:
        await s.runtime.send_message(_AGENT, _THREAD, "publish the launch")

        async def is_waiting() -> bool:
            try:
                return (
                    await s.stores.tool_calls.get(tool_call.id)
                ).status is ToolCallStatus.WAITING
            except KeyError:
                return False

        await _wait_for(is_waiting)
        assert publications == []
        await s.runtime.resolve_tool_call(
            _AGENT,
            _THREAD,
            tool_call.id,
            approved=True,
        )

        async def is_complete() -> bool:
            return (await s.stores.tool_calls.get(tool_call.id)).status is ToolCallStatus.COMPLETED

        await _wait_for(is_complete)
        record = await s.stores.tool_calls.get(tool_call.id)
        assert record.result == {
            "result": {"published": "Launch"},
            "tool_call_id": tool_call.id,
        }
        assert publications == ["Launch"]
        await s.runtime.cancel_thread(_AGENT, _THREAD)

    await _run(body, agent=agent)


@pytest.mark.asyncio
async def test_duplicate_resolutions_use_the_first_signal() -> None:
    tool_call = _tool_call("needs_approval")
    agent = _agent(
        FakeLLM([FakeResponse(tool_calls=[tool_call]), FakeResponse(text="done")]),
        tools=[_ApprovalTool()],
    )

    async def body(s: _RunSetup) -> None:
        await s.runtime.send_message(_AGENT, _THREAD, "please approve")

        async def is_waiting() -> bool:
            try:
                return (
                    await s.stores.tool_calls.get(tool_call.id)
                ).status is ToolCallStatus.WAITING
            except KeyError:
                return False

        await _wait_for(is_waiting)
        await s.runtime.resolve_tool_call(
            _AGENT, _THREAD, tool_call.id, approved=True, answer="first"
        )
        await s.runtime.resolve_tool_call(
            _AGENT, _THREAD, tool_call.id, approved=False, answer="second"
        )

        async def is_terminal() -> bool:
            return (await s.stores.tool_calls.get(tool_call.id)).status is ToolCallStatus.COMPLETED

        await _wait_for(is_terminal)
        record = await s.stores.tool_calls.get(tool_call.id)
        assert isinstance(record.result, dict)
        assert record.result["result"]["approved"] is True
        assert record.result["result"]["answer"] == "first"

        await s.runtime.cancel_thread(_AGENT, _THREAD)

    await _run(body, agent=agent)


@pytest.mark.asyncio
async def test_resolution_is_durable_while_no_worker_is_running() -> None:
    tool_call = _tool_call("needs_approval")
    agent = _agent(
        FakeLLM([FakeResponse(tool_calls=[tool_call]), FakeResponse(text="resumed")]),
        tools=[_ApprovalTool()],
    )
    stores = InMemoryRuntimeStores()
    task_queue = f"test-restart-{uuid.uuid4().hex[:8]}"
    activities = TemporalRuntimeActivities(stores=stores, agents={agent.id: agent})

    async with await WorkflowEnvironment.start_local() as env:
        runtime = TemporalRuntimeClient(
            stores=stores,
            agents={agent.id: agent},
            config=TemporalRuntimeConfig(
                address=env.client.service_client.config.target_host,
                namespace=env.client.namespace,
                task_queue=task_queue,
            ),
        )
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            await runtime.send_message(_AGENT, _THREAD, "please approve")

            async def is_waiting() -> bool:
                try:
                    return (
                        await stores.tool_calls.get(tool_call.id)
                    ).status is ToolCallStatus.WAITING
                except KeyError:
                    return False

            await _wait_for(is_waiting)

        # No worker is polling the task queue here. Temporal still accepts and
        # stores the signal; no Actant code has to remain alive.
        await runtime.resolve_tool_call(
            _AGENT, _THREAD, tool_call.id, approved=True, answer="offline"
        )
        assert (await stores.tool_calls.get(tool_call.id)).status is ToolCallStatus.WAITING

        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):

            async def resumed() -> bool:
                messages = await stores.messages.list_for_thread(_AGENT, _THREAD)
                return any(message.content == "resumed" for message in messages)

            await _wait_for(resumed)
            assert (await stores.tool_calls.get(tool_call.id)).status is ToolCallStatus.COMPLETED
            await runtime.cancel_thread(_AGENT, _THREAD)


@pytest.mark.asyncio
async def test_continue_as_new_preserves_thread_state_between_agent_runs() -> None:
    agent = _agent(FakeLLM([FakeResponse(text="one"), FakeResponse(text="two")]))
    stores = InMemoryRuntimeStores()
    task_queue = f"test-continue-{uuid.uuid4().hex[:8]}"
    activities = TemporalRuntimeActivities(stores=stores, agents={agent.id: agent})

    async with await WorkflowEnvironment.start_local() as env:
        runtime = TemporalRuntimeClient(
            stores=stores,
            agents={agent.id: agent},
            config=TemporalRuntimeConfig(
                address=env.client.service_client.config.target_host,
                namespace=env.client.namespace,
                task_queue=task_queue,
                history_size_threshold=1,
            ),
        )
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            await runtime.send_message(_AGENT, _THREAD, "first")

            async def first_run_rotated() -> bool:
                try:
                    state = await runtime.get_state(_AGENT, _THREAD)
                except Exception:  # continue-as-new transition is momentarily unqueryable
                    return False
                return state.current_run_id is None and state.turn_count_total == 1

            await _wait_for(first_run_rotated)
            await runtime.send_message(_AGENT, _THREAD, "second")

            async def second_run_finished() -> bool:
                state = await runtime.get_state(_AGENT, _THREAD)
                return state.current_run_id is None and state.turn_count_total == 2

            await _wait_for(second_run_finished)
            messages = await stores.messages.list_for_thread(_AGENT, _THREAD)
            assert [(message.role, message.content) for message in messages] == [
                ("user", "first"),
                ("assistant", "one"),
                ("user", "second"),
                ("assistant", "two"),
            ]
            await runtime.cancel_thread(_AGENT, _THREAD)
