"""A worker lost mid-activity: the run picks up where it left off.

A lost worker is simulated in-process: the first attempt does part of the work and then
hangs without heartbeating, which is all Temporal can see of a worker that died. The
workflow, retry policies and the real activities run the second attempt.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any
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
from actant.llm.messages import Message
from actant.runtime.events.streaming import StreamListener
from actant.runtime.temporal.types import ExecuteInput, ExecuteOutcome, RunTurnInput, TurnResult
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.types.threads import RunStatus
from actant.sandbox import Endpoint, RemoteRunner, host
from actant.sandbox.protocol import CallRequest, Route
from actant.tools import ToolRegistry, tools
from actant.tools.calls import ToolCallStatus
from runtime_fixtures import static_agents

#: Failure detection shortened so a lost attempt is noticed in a second.
_TIMEOUTS = ActivityTimeouts(tool_s=30, tool_heartbeat_s=1, turn_heartbeat_s=1)


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


class _Place:
    effects: list[str] = []
    gate = asyncio.Event()

    async def place(self) -> str:
        """Place the thing."""
        _Place.effects.append("placed")
        await _Place.gate.wait()
        return "placed"


class _Serving(host.Host):
    async def call(self, request: CallRequest) -> Any:
        if request.call_id in self.calls:
            _Place.gate.set()  # the lost call, sent again, finishes only once it attached
        return await super().call(request)


async def test_a_service_call_lost_with_its_worker_attaches_to_its_run_and_the_run_goes_on() -> (
    None
):
    _Place.effects, _Place.gate = [], asyncio.Event()
    serving = _Serving({"s": _Place})
    server = await asyncio.start_server(serving.connection, "127.0.0.1", 0)
    endpoint = Endpoint(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}")
    runner = RemoteRunner(endpoint, "s", "k")
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
    registry = ToolRegistry(tools(_Place, runner))
    agent = AgentDefinition(id="a", name="a", persona="", llm=fake, tools=registry)
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({"a": agent}))
    )

    @activity.defn(name="execute_tool")
    async def execute_tool(payload: ExecuteInput) -> ExecuteOutcome:
        if activity.info().attempt == 1:
            # The call reaches the host, then its worker dies without a heartbeat.
            call_id = f"{payload.thread_id}:{payload.tool_call_id}"
            request = CallRequest(service="s", key="k", method="place", call_id=call_id)
            body = request.model_dump_json().encode()
            asyncio.get_running_loop().run_in_executor(
                None, host.post, endpoint, Route.CALL, body, 30.0
            )
            await asyncio.Future()
        return await activities.tools.execute_tool(payload)

    try:
        await _run(agent, stores, activities, [execute_tool])
    finally:
        await runner.close()
        server.close()
        for writer in list(serving.writers):
            writer.close()

    [run] = await stores.runs.list_for_thread("a", "t")
    assert run.status is RunStatus.IDLE
    assert (await stores.tool_calls.get("call")).status is ToolCallStatus.COMPLETED
    assert _Place.effects == ["placed"]  # run once, its result delivered to the retry
    messages = await stores.messages.list_for_thread("a", "t")
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert "placed" in str(messages[2].content)


class _DiesOnFirstCall(FakeLLM):
    """The first model call never returns, as if its worker died mid-request."""

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: StreamListener | None = None,
        *,
        allowed_tools: tuple[str, ...] = (),
        max_output_tokens: int | None = None,
    ) -> Message:
        if not self.calls:
            self.calls.append((system, list(messages), tools))
            await asyncio.Future()
        return await super().complete(
            system,
            messages,
            tools,
            listener,
            allowed_tools=allowed_tools,
            max_output_tokens=max_output_tokens,
        )


async def _run(
    agent: AgentDefinition,
    stores: InMemoryRuntimeStores,
    activities: TemporalRuntimeActivities,
    replaced: list[Callable[..., Any]] | None = None,
) -> None:
    names = {fn.__name__ for fn in replaced or []}
    registered = [fn for fn in activities.all if fn.__name__ not in names]
    async with await WorkflowEnvironment.start_local() as env:
        config = TemporalRuntimeConfig(task_queue=uuid4().hex, activity_timeouts=_TIMEOUTS)
        async with Worker(
            env.client,
            task_queue=config.task_queue,
            workflows=[AgentThreadWorkflow],
            activities=[*registered, *(replaced or [])],
        ):
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            workflow_id = await runtime.send_message("a", "t", "go")
            await asyncio.wait_for(env.client.get_workflow_handle(workflow_id).result(), 30)


async def test_a_model_turn_lost_mid_call_is_asked_again_without_storing_its_input_twice() -> (
    None
):
    fake = _DiesOnFirstCall([FakeResponse(text="done")])
    agent = AgentDefinition(id="a", name="a", persona="", llm=fake, tools=ToolRegistry([]))
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({"a": agent}))
    )
    await _run(agent, stores, activities)

    [run] = await stores.runs.list_for_thread("a", "t")
    assert run.status is RunStatus.IDLE
    messages = await stores.messages.list_for_thread("a", "t")
    assert [(m.role, m.content) for m in messages] == [("user", "go"), ("assistant", "done")]
    assert len(fake.calls) == 2
    assert (await stores.threads.get("a", "t")).turn_count == 1


async def test_a_model_turn_lost_after_its_answer_was_stored_is_not_asked_again() -> None:
    placed: list[str] = []

    @tool
    async def place() -> str:
        """Place the thing."""
        placed.append("placed")
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

    @activity.defn(name="run_turn")
    async def run_turn(payload: RunTurnInput) -> TurnResult:
        result = await activities.runs.run_turn(payload)
        if activity.info().attempt == 1 and result.tool_calls:
            await asyncio.Future()  # the answer is stored; the worker dies before returning
        return result

    await _run(agent, stores, activities, [run_turn])

    [run] = await stores.runs.list_for_thread("a", "t")
    assert run.status is RunStatus.IDLE
    messages = await stores.messages.list_for_thread("a", "t")
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant"]
    assert len(fake.calls) == 2
    assert placed == ["placed"]
    assert (await stores.threads.get("a", "t")).turn_count == 2
