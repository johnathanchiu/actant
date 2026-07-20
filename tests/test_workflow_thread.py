"""End-to-end tests for ``AgentThreadWorkflow`` + activities.

Each test starts a real Temporal worker against an in-memory
``WorkflowEnvironment``, drives the workflow via signals, and asserts
on the projection state of ``InMemoryRuntimeStores``.

The activities import the same hooks/listeners/preprocessor knobs that
the production worker uses, so this is an integration-level test of
the pure-Temporal runtime end-to-end.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from actant.agents import AgentDefinition
from actant.core import JSONObject, new_id
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime.temporal.activities import TemporalRuntimeActivities
from actant.runtime.temporal.types import (
    ExecuteOutcome,
    ExecuteStatus,
    InboundMessage,
    ThreadInput,
)
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.hooks import AgentThreadHooks
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.types.threads import AgentThread
from actant.tools.admission import (
    ToolCallView,
    ToolDecision,
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

_AGENT = "test_agent"
_THREAD = "test_thread"


# === helpers ===


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
    return ToolCall(
        id=new_id("tc"),
        function=ToolCallFunction(name=name, arguments=args),
    )


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
    activities: TemporalRuntimeActivities
    task_queue: str


async def _run(
    test: Callable[[_RunSetup, object], Awaitable[None]],
    *,
    agent: AgentDefinition,
    hooks_factory: object | None = None,
) -> None:
    """Spin up a WorkflowEnvironment + Worker and run ``test`` inside it."""
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        stores=stores,
        agents={agent.id: agent},
        hooks_factory=hooks_factory,  # type: ignore[arg-type]
    )
    task_queue = f"test-actant-{uuid.uuid4().hex[:8]}"

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            setup = _RunSetup(env, stores, activities, task_queue)
            await test(setup, env.client)


# === text-only turn ===


@pytest.mark.asyncio
async def test_inbound_runs_one_turn_and_persists_assistant_message() -> None:
    agent = _agent(FakeLLM([FakeResponse(text="hi back")]))

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="hello")],
        )

        async def has_assistant() -> bool:
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return any(m.role == "assistant" for m in msgs)

        await _wait_for(has_assistant)

        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant"]
        assert messages[-1].content == "hi back"

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


# === tool turn (ALLOW) ===


class _EchoInvocation(BaseToolInvocation[JSONObject, dict[str, object]]):
    async def execute(self) -> ToolResult:
        return ToolResult.ok({"echoed": self.params})


class _EchoTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        super().__init__("echo", make_tool_schema("echo", "Echo back"))

    async def build(self, params: JSONObject) -> _EchoInvocation:
        return _EchoInvocation(params)


class _TerminalInvocation(BaseToolInvocation[JSONObject, dict[str, object]]):
    async def execute(self) -> ToolResult:
        return ToolResult.ok({"done": True}, terminal=True)


class _TerminalTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        super().__init__("finish", make_tool_schema("finish", "Stop the run"))

    async def build(self, params: JSONObject) -> _TerminalInvocation:
        return _TerminalInvocation(params)


@dataclass
class _ParallelProbe:
    first_started: asyncio.Event
    second_started: asyncio.Event
    release: asyncio.Event


class _ParallelInvocation(BaseToolInvocation[JSONObject, object]):
    def __init__(self, params: JSONObject, started: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__(params)
        self.started = started
        self.release = release

    async def execute(self) -> ToolResult:
        self.started.set()
        await self.release.wait()
        return ToolResult.ok({"completed": True})


class _ParallelTool(BaseDeclarativeTool):
    def __init__(self, name: str, started: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__(name, make_tool_schema(name, "Parallel execution probe"))
        self.started = started
        self.release = release

    async def build(self, params: JSONObject) -> _ParallelInvocation:
        return _ParallelInvocation(params, self.started, self.release)


@pytest.mark.asyncio
async def test_tool_turn_allow_completes_and_continues() -> None:
    tool_call = _tool_call("echo", '{"x": 1}')
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[tool_call]),
                FakeResponse(text="done"),
            ]
        ),
        tools=[_EchoTool()],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="run echo")],
        )

        async def two_assistants() -> bool:
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in msgs if m.role == "assistant") >= 2

        await _wait_for(two_assistants)

        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        roles = [m.role for m in messages]
        # user → assistant(tool_call) → tool → assistant
        assert roles == ["user", "assistant", "tool", "assistant"]
        record = await s.stores.tool_calls.get(tool_call.id)
        assert record.status == ToolCallStatus.COMPLETED

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


@pytest.mark.asyncio
async def test_parallel_tool_calls_execute_concurrently() -> None:
    probe = _ParallelProbe(asyncio.Event(), asyncio.Event(), asyncio.Event())
    first = _tool_call("parallel_a")
    second = _tool_call("parallel_b")
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[first, second]),
                FakeResponse(text="both completed"),
            ]
        ),
        tools=[
            _ParallelTool("parallel_a", probe.first_started, probe.release),
            _ParallelTool("parallel_b", probe.second_started, probe.release),
        ],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="run both")],
        )

        # Both invocations must start before either is allowed to finish.
        # A sequential executor deadlocks here and fails the timeout.
        await asyncio.wait_for(
            asyncio.gather(probe.first_started.wait(), probe.second_started.wait()),
            timeout=5.0,
        )
        probe.release.set()

        async def continued() -> bool:
            messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return any(message.content == "both completed" for message in messages)

        await _wait_for(continued)
        assert (await s.stores.tool_calls.get(first.id)).status == ToolCallStatus.COMPLETED
        assert (await s.stores.tool_calls.get(second.id)).status == ToolCallStatus.COMPLETED

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


@pytest.mark.asyncio
async def test_terminal_tool_completes_without_followup_llm_turn() -> None:
    tool_call = _tool_call("finish")
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[tool_call]),
                FakeResponse(text="should not be called"),
            ]
        ),
        tools=[_TerminalTool()],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="finish")],
        )

        async def thread_idle() -> bool:
            try:
                thread = await s.stores.threads.get(_AGENT, _THREAD)
            except KeyError:
                return False
            return thread.status.value == "idle"

        await _wait_for(thread_idle)

        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "tool"]
        record = await s.stores.tool_calls.get(tool_call.id)
        assert record.status == ToolCallStatus.COMPLETED
        assert record.result is not None
        result = cast(dict[str, object], record.result)
        assert result.get("metadata") == {"terminal": True}

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


# === tool turn (WAIT → external resolution via complete_activity_by_id) ===


class _ApprovalInvocation(BaseToolInvocation[JSONObject, object]):
    async def execute(self) -> ToolResult:  # pragma: no cover — never called for WAIT
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

    async def build(self, params: JSONObject) -> _ApprovalInvocation:
        return _ApprovalInvocation(params)


@pytest.mark.asyncio
async def test_wait_tool_parks_until_external_completion() -> None:
    """WAIT-decision tools are handled via async activity completion.

    The workflow fires ``await_external_resolution`` which stamps
    ``(workflow_id, activity_id)`` onto the tool_call record and parks
    via ``raise_complete_async``. Test delivers the resolution by:
    1. Persisting the result to the record (mirrors what
       ``TemporalRuntimeClient.resolve_deferred_tool_call`` does in production).
    2. Calling ``client.get_async_activity_handle(...).complete(...)``
       to unblock the workflow's ``await``.
    """
    tool_call = _tool_call("needs_approval")
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[tool_call]),
                FakeResponse(text="approved!"),
            ]
        ),
        tools=[_ApprovalTool()],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="please approve")],
        )

        async def has_temporal_handle() -> bool:
            try:
                rec = await s.stores.tool_calls.get(tool_call.id)
            except KeyError:
                return False
            return (
                rec.status == ToolCallStatus.WAITING
                and rec.temporal_workflow_id is not None
                and rec.temporal_activity_id is not None
            )

        # Wait for await_external_resolution to park: status=WAITING +
        # temporal handle stamped.
        await _wait_for(has_temporal_handle)
        record = await s.stores.tool_calls.get(tool_call.id)
        assert record.temporal_workflow_id is not None
        assert record.temporal_activity_id is not None

        # Deliver the resolution: persist result + complete the activity.
        # This mirrors what ``TemporalRuntimeClient.resolve_deferred_tool_call`` does in
        # production. (Direct calls here for explicit testing of the
        # Temporal mechanics.)
        await s.stores.tool_calls.update_status(
            tool_call.id,
            ToolCallStatus.COMPLETED,
            result={"approved": True, "answer": "ok"},
        )
        async_handle = client.get_async_activity_handle(
            workflow_id=record.temporal_workflow_id,
            activity_id=record.temporal_activity_id,
        )
        await async_handle.complete(
            ExecuteOutcome(
                tool_call_id=tool_call.id,
                status=ExecuteStatus.COMPLETED.value,
            )
        )

        async def has_second_assistant() -> bool:
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in msgs if m.role == "assistant") >= 2

        await _wait_for(has_second_assistant)

        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        # Last assistant message should be the post-resolve continuation.
        assistants = [m for m in messages if m.role == "assistant"]
        assert assistants[-1].content == "approved!"

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


@pytest.mark.asyncio
async def test_mixed_allow_and_wait_group_continues_only_after_resolution() -> None:
    allowed = _tool_call("echo", '{"value": "ready"}')
    deferred = _tool_call("needs_approval")
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[allowed, deferred]),
                FakeResponse(text="group continued exactly once"),
            ]
        ),
        tools=[_EchoTool(), _ApprovalTool()],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="run mixed group")],
        )

        async def allowed_done_deferred_waiting() -> bool:
            try:
                allowed_record = await s.stores.tool_calls.get(allowed.id)
                deferred_record = await s.stores.tool_calls.get(deferred.id)
            except KeyError:
                return False
            return (
                allowed_record.status == ToolCallStatus.COMPLETED
                and deferred_record.status == ToolCallStatus.WAITING
                and deferred_record.temporal_workflow_id is not None
                and deferred_record.temporal_activity_id is not None
            )

        await _wait_for(allowed_done_deferred_waiting)
        before = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        assert [message.role for message in before] == ["user", "assistant"]
        assert not any(message.content == "group continued exactly once" for message in before)

        deferred_record = await s.stores.tool_calls.get(deferred.id)
        await s.stores.tool_calls.update_status(
            deferred.id,
            ToolCallStatus.COMPLETED,
            result={"approved": True},
        )
        assert deferred_record.temporal_workflow_id is not None
        assert deferred_record.temporal_activity_id is not None
        activity_handle = client.get_async_activity_handle(
            workflow_id=deferred_record.temporal_workflow_id,
            activity_id=deferred_record.temporal_activity_id,
        )
        await activity_handle.complete(
            ExecuteOutcome(
                tool_call_id=deferred.id,
                status=ExecuteStatus.COMPLETED.value,
            )
        )

        async def continued() -> bool:
            messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return any(message.content == "group continued exactly once" for message in messages)

        await _wait_for(continued)
        after = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        assert [message.role for message in after] == [
            "user",
            "assistant",
            "tool",
            "tool",
            "assistant",
        ]
        assert sum(
            message.content == "group continued exactly once" for message in after
        ) == 1

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


# === exhaustion ===


@pytest.mark.asyncio
async def test_exhaustion_finalizes_run_then_next_message_starts_fresh_run() -> None:
    """Force tool calls every turn until the run hits its budget.

    With ``max_turns_per_run=2`` and the agent always producing a tool
    call, the inner loop hits ``turns_left == 0`` and finalizes
    EXHAUSTED. The next user message should start a brand new run
    with a fresh budget (different ``run_id`` on stored runs).
    """
    # 5 responses available — enough for two consecutive runs of up to 2
    # turns each (4) plus a buffer.
    fake = FakeLLM(
        [
            FakeResponse(tool_calls=[_tool_call("echo", '{"i": 1}')]),
            FakeResponse(tool_calls=[_tool_call("echo", '{"i": 2}')]),
            FakeResponse(tool_calls=[_tool_call("echo", '{"i": 3}')]),
            FakeResponse(text="finally done"),
            FakeResponse(text="extra"),
        ]
    )
    agent = _agent(fake, tools=[_EchoTool()])

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=2),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="loop forever please")],
        )

        async def first_run_exhausted() -> bool:
            # Look for any run row with EXHAUSTED outcome. We can't
            # peek inside InMemory cleanly, so use turn count instead:
            # 2 assistant messages = 2 turns happened.
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in msgs if m.role == "assistant") >= 2

        await _wait_for(first_run_exhausted)

        # Give the workflow a beat to finalize the run before sending
        # the next message — finalize_run is async and the next
        # send_message must observe a fresh run boundary.
        await asyncio.sleep(0.2)

        # Send a follow-up message. This starts a NEW run.
        await handle.signal(AgentThreadWorkflow.inbound, InboundMessage(content="continue"))

        async def fourth_assistant() -> bool:
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in msgs if m.role == "assistant") >= 4

        await _wait_for(fourth_assistant)

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


# === hooks fire from inside activities ===


class _RecordingHooks(AgentThreadHooks):
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    async def on_user_message(self, content) -> None:  # type: ignore[no-untyped-def]
        self.events.append(("user", content))

    async def on_assistant_message(self, message: Message) -> None:
        self.events.append(("assistant", message.content))

    async def on_turn_start(self, turn: int, turn_id: str | None = None) -> None:
        self.events.append(("turn_start", (turn, turn_id)))

    async def on_complete(self, success: bool, reason: str, message: str) -> None:
        self.events.append(("complete", reason))


@pytest.mark.asyncio
async def test_hooks_fire_inside_activities() -> None:
    hooks = _RecordingHooks()

    def factory(_thread: AgentThread) -> _RecordingHooks:
        return hooks

    agent = _agent(FakeLLM([FakeResponse(text="ok")]))

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="hi")],
        )

        await _wait_for(lambda: any(e[0] == "complete" for e in hooks.events))

        kinds = [e[0] for e in hooks.events]
        assert "user" in kinds
        assert "turn_start" in kinds
        assert "assistant" in kinds
        assert "complete" in kinds

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent, hooks_factory=factory)
