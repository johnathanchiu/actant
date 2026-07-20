"""Cancellation finalization — tool_call placeholders + thread.status.

Two gaps the rewrite left open and this module pins down:

1. ``cancel_thread`` on a workflow with in-flight tool_calls must
   write cancellation placeholder results, otherwise the LLM
   transcript is broken (every tool_call needs a matching
   tool_result, and provider 400s without one).

2. ``thread.status`` after a finalized run must reflect the run
   outcome — terminal (FAILED / CANCELLED) or alive (IDLE). The
   pre-fix ``finalize_run`` always wrote IDLE, losing the
   distinction a UI run card depends on.

Both tests run against a real Temporal worker via
``WorkflowEnvironment.start_local()``.
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
from actant.runtime.types.threads import ThreadStatus
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


# === helpers (mirrors test_workflow_thread.py) ===


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
) -> None:
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        stores=stores,
        agents={agent.id: agent},
    )
    task_queue = f"test-cancel-{uuid.uuid4().hex[:8]}"

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            setup = _RunSetup(env, stores, activities, task_queue)
            await test(setup, env.client)


# === Gap 1: cancel writes tool_call placeholders ===


class _ParkInvocation(BaseToolInvocation[JSONObject, object]):
    async def execute(self) -> ToolResult:  # pragma: no cover — never called for WAIT
        return ToolResult.ok({})


class _ParkTool(BaseDeclarativeTool):
    """Always parks via WAIT — never resolves on its own. The test
    cancels the workflow while this tool is in WAITING status."""

    def __init__(self) -> None:
        super().__init__("park", make_tool_schema("park", "Always parks"))

    async def can_execute(
        self,
        call: ToolCallView,
        invocation: object,
        context: TurnContextView,
    ) -> ToolDecision:
        del call, invocation, context
        return ToolDecision.wait(ToolWaitRequest(kind="approval", prompt="approve?", payload={}))

    async def build(self, params: JSONObject) -> _ParkInvocation:
        return _ParkInvocation(params)


@pytest.mark.asyncio
async def test_cancel_writes_placeholder_for_waiting_tool_calls() -> None:
    """Cancel-while-tool-WAITING repairs BOTH projection AND transcript.

    Pre-fix: tool_call stayed WAITING in the projection table forever,
    AND its tool_result message never made it into the message log —
    the next provider call 400s on the orphan in the transcript.

    Post-fix: ``apply_thread_cancellation`` does two repairs:
    1. Stamps open tool_call records to COMPLETED with the
       ``session_cancelled`` placeholder (projection invariant).
    2. Appends placeholder ``tool_result`` messages for any
       tool_call_id that has no matching tool_result in the message
       log (LLM transcript invariant).
    Both repairs are idempotent.
    """
    tool_call = _tool_call("park")
    agent = _agent(
        FakeLLM([FakeResponse(tool_calls=[tool_call])]),
        tools=[_ParkTool()],
    )

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="trigger park")],
        )

        async def is_waiting() -> bool:
            try:
                rec = await s.stores.tool_calls.get(tool_call.id)
            except KeyError:
                return False
            return rec.status == ToolCallStatus.WAITING

        await _wait_for(is_waiting)

        # Cancel the workflow while the tool is parked.
        await handle.cancel()

        # Workflow handle.result() raises on CancelledError; we don't
        # care, we just want the finalize_run side-effects.
        try:
            await asyncio.wait_for(handle.result(), timeout=5.0)
        except Exception:
            pass

        # 1. Projection: tool_call record is COMPLETED with placeholder.
        record = await s.stores.tool_calls.get(tool_call.id)
        assert record.status == ToolCallStatus.COMPLETED
        assert isinstance(record.result, dict)
        assert record.result.get("status") == "cancelled"
        assert record.result.get("reason") == "session_cancelled"

        # 2. Transcript: every tool_call in the message log has a
        #    matching tool_result. No orphans. Without the transcript
        #    repair this assertion fails and the next LLM call would
        #    400 on the orphan.
        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        tool_call_ids = {
            tc.id
            for m in messages
            if m.role == "assistant" and m.tool_calls
            for tc in m.tool_calls
        }
        tool_result_ids = {
            m.tool_call_id
            for m in messages
            if m.role == "tool" and m.tool_call_id is not None
        }
        assert tool_call_ids == tool_result_ids, (
            f"orphans found: {tool_call_ids - tool_result_ids}"
        )

    await _run(body, agent=agent)


# === Gap 2: thread.status reflects outcome ===


@pytest.mark.asyncio
async def test_thread_status_is_cancelled_after_workflow_cancel() -> None:
    """Pre-fix: thread.status was always IDLE post-finalize. UIs use
    status to distinguish "between turns" from "this thread
    is done" — so a cancelled session must read CANCELLED."""
    agent = _agent(FakeLLM([FakeResponse(text="hi")]))

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="hi")],
        )

        async def has_assistant() -> bool:
            msgs = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
            return any(m.role == "assistant" for m in msgs)

        await _wait_for(has_assistant)

        # Cancel after the first turn lands but before the workflow
        # naturally exits — the next finalize will flip thread.status.
        await handle.cancel()
        try:
            await asyncio.wait_for(handle.result(), timeout=5.0)
        except Exception:
            pass

        thread = await s.stores.threads.get(_AGENT, _THREAD)
        assert thread.status == ThreadStatus.CANCELLED
        assert thread.active_run_id is None

    await _run(body, agent=agent)


@pytest.mark.asyncio
async def test_thread_status_returns_to_idle_after_normal_completion() -> None:
    """COMPLETED outcome → thread.status=IDLE so the workflow can
    accept the next user message. Sanity check that the new mapping
    didn't break the happy path."""
    agent = _agent(FakeLLM([FakeResponse(text="done")]))

    async def body(s: _RunSetup, client) -> None:  # type: ignore[no-untyped-def]
        handle = await client.start_workflow(
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=5),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=s.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content="ping")],
        )

        async def thread_idle() -> bool:
            try:
                t = await s.stores.threads.get(_AGENT, _THREAD)
            except KeyError:
                return False
            return t.status == ThreadStatus.IDLE and t.active_run_id is None

        await _wait_for(thread_idle)

        thread = await s.stores.threads.get(_AGENT, _THREAD)
        assert thread.status == ThreadStatus.IDLE
        assert thread.active_run_id is None

        await handle.signal(AgentThreadWorkflow.cancel)
        try:
            await asyncio.wait_for(handle.result(), timeout=5.0)
        except Exception:
            pass

    await _run(body, agent=agent)
