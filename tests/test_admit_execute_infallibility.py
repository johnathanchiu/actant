"""Activity-boundary catch tests.

The whole point of splitting ``admit_tool`` and ``execute_tool`` into
their own activities is that each catches at its boundary — they
ALWAYS return a structured outcome, even when the tool's user code
(``can_execute``, ``build``, ``execute``) raises an unhandled
exception. The workflow can then orchestrate without try/except.

These tests pin that contract: an exploding tool must not propagate
an exception out of either activity. The orphan-tool-call bug this
guards against happens when an admission failure skips tool-group
finalization, leaving an assistant tool call without a matching tool
result. The new contract makes that impossible.
"""

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
from actant.runtime.temporal.types import (
    InboundMessage,
    ThreadInput,
)
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools.admission import (
    ToolCallView,
    ToolDecision,
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


# === helpers (same shape as test_workflow_thread) ===


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
    activities: TemporalRuntimeActivities
    task_queue: str


async def _run(
    test: Callable[[_RunSetup, object], Awaitable[None]],
    *,
    agent: AgentDefinition,
) -> None:
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(stores=stores, agents={agent.id: agent})
    task_queue = f"test-infallible-{uuid.uuid4().hex[:8]}"

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            setup = _RunSetup(env, stores, activities, task_queue)
            await test(setup, env.client)


# === can_execute that raises ===


class _BoomInvocation(BaseToolInvocation[JSONObject, object]):
    async def execute(self) -> ToolResult:  # pragma: no cover — never reached
        return ToolResult.ok({})


class _CanExecuteRaisesTool(BaseDeclarativeTool):
    """Tool whose ``can_execute`` raises — simulates the orphan-result bug
    where a constraint check threw out of the admission activity."""

    def __init__(self) -> None:
        super().__init__("can_execute_raises", make_tool_schema("can_execute_raises", "kaboom"))

    async def can_execute(
        self,
        call: ToolCallView,
        invocation: object,
        context: TurnContextView,
    ) -> ToolDecision:
        del call, invocation, context
        raise RuntimeError("simulated can_execute crash")

    async def build(self, params: JSONObject) -> _BoomInvocation:
        return _BoomInvocation(params)


@pytest.mark.asyncio
async def test_admit_tool_swallows_can_execute_exception() -> None:
    """``can_execute`` raising must NOT break the workflow.

    The buggy old behavior: gather() re-raises → _run_tool_group
    returns FAILED → finalize_tool_group is skipped → orphan tool_call
    in the message log → next LLM call 400s on the orphan forever.

    New behavior: admit_tool's outermost catch maps the exception to
    decision=BLOCK with reason="admission_error: ...". The workflow
    keeps going. ``finalize_tool_group`` runs and appends the
    tool_result message, closing the transcript invariant. The agent
    can recover on the next turn.
    """
    boom = _tool_call("can_execute_raises")
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[boom]),
                FakeResponse(text="recovered"),
            ]
        ),
        tools=[_CanExecuteRaisesTool()],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="boom")],
        )

        async def two_assistants() -> bool:
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in msgs if m.role == "assistant") >= 2

        # The whole point: workflow recovers, second agent turn fires.
        await _wait_for(two_assistants)

        # Tool call is in a terminal status (BLOCKED via admission_error path).
        record = await s.stores.tool_calls.get(boom.id)
        assert record.status == ToolCallStatus.BLOCKED
        assert isinstance(record.result, dict)
        assert "admission_error" in str(record.result.get("error", ""))

        # Transcript is whole — every tool_call has a matching tool_result.
        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "tool", "assistant"]

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


# === execute that raises ===


class _ExecuteRaisesInvocation(BaseToolInvocation[JSONObject, object]):
    async def execute(self) -> ToolResult:
        raise RuntimeError("simulated execute crash")


class _ExecuteRaisesTool(BaseDeclarativeTool):
    """Tool whose ``execute`` raises — admission ALLOWs, then execution
    explodes inside the activity. Should still come back cleanly."""

    def __init__(self) -> None:
        super().__init__("execute_raises", make_tool_schema("execute_raises", "kaboom"))

    async def build(self, params: JSONObject) -> _ExecuteRaisesInvocation:
        return _ExecuteRaisesInvocation(params)


@pytest.mark.asyncio
async def test_execute_tool_swallows_execute_exception() -> None:
    """An exception in ``invocation.execute()`` is caught inside
    ``execute_tool`` and converted to ``status=FAILED`` with the
    error captured. Workflow keeps moving."""
    boom = _tool_call("execute_raises")
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[boom]),
                FakeResponse(text="recovered"),
            ]
        ),
        tools=[_ExecuteRaisesTool()],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="boom")],
        )

        async def two_assistants() -> bool:
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in msgs if m.role == "assistant") >= 2

        await _wait_for(two_assistants)

        record = await s.stores.tool_calls.get(boom.id)
        assert record.status == ToolCallStatus.FAILED
        assert isinstance(record.result, dict)
        assert "execution error" in str(record.result.get("error", "")).lower()

        # Transcript whole.
        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "tool", "assistant"]

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)


# === mixed group: one ok, one explodes ===


class _OkInvocation(BaseToolInvocation[JSONObject, dict[str, object]]):
    async def execute(self) -> ToolResult:
        return ToolResult.ok({"echoed": self.params})


class _OkTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        super().__init__("ok_tool", make_tool_schema("ok_tool", "always works"))

    async def build(self, params: JSONObject) -> _OkInvocation:
        return _OkInvocation(params)


@pytest.mark.asyncio
async def test_mixed_group_one_explodes_one_succeeds() -> None:
    """Critical case the old design got wrong: when one tool's admit
    raises in a multi-tool group, the OTHER tools in the group
    previously got cancelled by ``asyncio.gather`` and the whole
    group's finalize was skipped — leaving orphans for every tool.

    With the activity-boundary catch, each tool's outcome is
    independent. The exploding one gets BLOCKED + admission_error;
    the working one runs to COMPLETED. Both end up in the transcript."""
    boom = _tool_call("can_execute_raises")
    ok = _tool_call("ok_tool", '{"x": 1}')
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[boom, ok]),
                FakeResponse(text="recovered"),
            ]
        ),
        tools=[_CanExecuteRaisesTool(), _OkTool()],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="mixed")],
        )

        async def two_assistants() -> bool:
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return sum(1 for m in msgs if m.role == "assistant") >= 2

        await _wait_for(two_assistants)

        boom_rec = await s.stores.tool_calls.get(boom.id)
        ok_rec = await s.stores.tool_calls.get(ok.id)
        assert boom_rec.status == ToolCallStatus.BLOCKED
        assert ok_rec.status == ToolCallStatus.COMPLETED

        # Both tools have matching tool_result messages — transcript whole.
        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        tool_msgs = [m for m in messages if m.role == "tool"]
        tool_call_ids = {m.tool_call_id for m in tool_msgs}
        assert tool_call_ids == {boom.id, ok.id}

        await handle.signal(AgentThreadWorkflow.cancel)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(body, agent=agent)
