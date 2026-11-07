"""A finished sub-thread reports to its parent, under the workflow and in process alike.

The parent hears once per child run, on its inbox, whether it is parked mid-run or already
closed; the report says which child, how its run ended, and what it said.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer

from actant.agents import AgentDefinition
from actant.blocks import TextBlock
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime
from actant.runtime.local import LocalThreadRuntime
from actant.runtime.runtime import WORKFLOW_RUNNER
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.subagents import SUBAGENT_REPORT_TAG, parent_report
from actant.runtime.temporal.activities import RunActivities
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import (
    ReportToParentInput,
    RunOutcome,
    TemporalRuntimeConfig,
    ThreadInput,
)
from actant.runtime.temporal.workflow import WORKFLOWS
from actant.runtime.types.threads import ThreadStatus
from actant.tools import ToolRegistry, tool
from actant.tools.admission import ToolCallView, ToolDecision, ToolWaitRequest, TurnContextView
from actant.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    CallContext,
    Tool,
    ToolResult,
    make_tool_schema,
)
from actant.core import JSONObject
from runtime_fixtures import static_agents

pytestmark = pytest.mark.asyncio

_PARENT, _CHILD = "planner", "author"


def _agent(agent_id: str, replies: list[FakeResponse], tools: list[Tool] | None = None):
    return AgentDefinition(
        id=agent_id,
        name=agent_id,
        persona="",
        llm=FakeLLM(replies),
        tools=ToolRegistry(tools or []),
        tool_allowlist={t.name for t in tools or []},
    )


def _call(name: str) -> FakeResponse:
    return FakeResponse(
        tool_calls=[ToolCall(id=f"tc_{name}", function=ToolCallFunction(name, "{}"))]
    )


def _text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content
    return "".join(b.text for b in message.content or [] if isinstance(b, TextBlock))


async def _reports(stores: InMemoryRuntimeStores, thread_id: str = "p") -> list[str]:
    messages = await stores.messages.list_for_thread(_PARENT, thread_id)
    return [_text(m) for m in messages if m.tag == SUBAGENT_REPORT_TAG]


async def _said(stores: InMemoryRuntimeStores, text: str, thread_id: str = "p") -> bool:
    messages = await stores.messages.list_for_thread(_PARENT, thread_id)
    return any(m.role == "assistant" and _text(m) == text for m in messages)


async def _wait_for(predicate: Callable[[], Awaitable[bool]], timeout: float = 30.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not await predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError("predicate did not become true within timeout")
        await asyncio.sleep(0.05)


# === the report ===


async def _linked(stores: InMemoryRuntimeStores, parent_status: ThreadStatus) -> None:
    parent = await stores.threads.get_or_create(_PARENT, "p")
    parent.status = parent_status
    await stores.threads.update(parent)
    child = await stores.threads.get_or_create(_CHILD, "c")
    child.parent_thread_id = "p"
    await stores.threads.update(child)
    await stores.messages.append_assistant(_CHILD, "c", "turn", Message("assistant", "done it"))


async def test_a_cancelled_parent_is_not_told() -> None:
    """A cancel that reached the child through its parent must not start the parent again."""
    stores = InMemoryRuntimeStores()
    await _linked(stores, ThreadStatus.CANCELLED)
    report = ReportToParentInput(_CHILD, "c", "r1", RunOutcome.CANCELLED.value)
    assert await parent_report(stores, report) is None


async def test_a_report_attempted_again_after_the_parent_stored_it_is_not_sent() -> None:
    stores = InMemoryRuntimeStores()
    await _linked(stores, ThreadStatus.IDLE)
    sent: list[tuple[str, str, str]] = []

    async def message_thread(agent_id: str, thread_id: str, msg) -> None:
        sent.append((agent_id, thread_id, str(msg.content)))
        await stores.messages.append_user(agent_id, thread_id, msg.content, tag=msg.tag)

    runs = RunActivities(ActivityContext(stores=stores, message_thread=message_thread))
    report = ReportToParentInput(_CHILD, "c", "r1", RunOutcome.COMPLETED.value)
    await runs.report_to_parent(report)
    await runs.report_to_parent(report)
    assert [(a, t) for a, t, _ in sent] == [(_PARENT, "p")]
    [text] = await _reports(stores)
    assert 'thread_id="c"' in text and 'status="completed"' in text and "done it" in text


# === under the workflow ===


class _Approval(BaseDeclarativeTool):
    """Parks the parent mid-run until a person answers."""

    def __init__(self) -> None:
        super().__init__("approve", make_tool_schema("approve", "Needs approval"))

    async def can_execute(
        self, call: ToolCallView, invocation: object, context: TurnContextView
    ) -> ToolDecision:
        return ToolDecision.await_human(ToolWaitRequest(kind="approval", prompt="ok?", payload={}))

    async def build(self, params: JSONObject, ctx: CallContext) -> BaseToolInvocation:
        class _Never(BaseToolInvocation[JSONObject, object]):
            async def execute(self) -> ToolResult:  # pragma: no cover -- never runs
                return ToolResult.ok()

        return _Never(params)


@asynccontextmanager
async def _temporal(
    stores: InMemoryRuntimeStores, *agents: AgentDefinition
) -> AsyncIterator[tuple[AgentRuntime, Client]]:
    async with await WorkflowEnvironment.start_local() as env:
        runtime = AgentRuntime(
            client=env.client,
            stores=stores,
            config=TemporalRuntimeConfig(task_queue=f"test-report-{uuid.uuid4().hex[:8]}"),
            resolve_agent=static_agents({a.id: a for a in agents}),
        )
        worker = asyncio.create_task(runtime.run_worker())
        try:
            yield runtime, env.client
        finally:
            await runtime.shutdown()
            await worker


async def test_a_completed_child_wakes_its_closed_parent_once_and_replays() -> None:
    stores = InMemoryRuntimeStores()
    parent = _agent(_PARENT, [FakeResponse(text="hello"), FakeResponse(text="got it")])
    child = _agent(_CHILD, [FakeResponse(text="the chair is done")])
    async with _temporal(stores, parent, child) as (runtime, client):
        await runtime.send_message(_PARENT, "p", "start")
        await client.get_workflow_handle(runtime._workflow_id(_PARENT, "p")).result()

        await runtime.spawn(_CHILD, "c", "build a chair", parent_thread_id="p")
        await _wait_for(lambda: _said(stores, "got it"))

        [report] = await _reports(stores)
        assert 'thread_id="c"' in report and f'agent_id="{_CHILD}"' in report
        assert 'status="completed"' in report and "the chair is done" in report

        child_handle = client.get_workflow_handle(runtime._workflow_id(_CHILD, "c"))
        await child_handle.result()
        replayer = Replayer(workflows=WORKFLOWS, workflow_runner=WORKFLOW_RUNNER)
        await replayer.replay_workflow(await child_handle.fetch_history())
        assert len(await _reports(stores)) == 1


async def test_a_failed_child_reaches_its_parked_parent_once() -> None:
    stores = InMemoryRuntimeStores()
    parent = _agent(
        _PARENT,
        [_call("approve"), FakeResponse(text="approved"), FakeResponse(text="noted the failure")],
        tools=[_Approval()],
    )
    child = _agent(_CHILD, [])  # no reply queued: the child's model call fails
    async with _temporal(stores, parent, child) as (runtime, _client):
        await runtime.send_message(_PARENT, "p", "start")

        async def parked() -> bool:
            try:
                return (await stores.tool_calls.get("tc_approve")).status.value == "waiting"
            except KeyError:
                return False

        await _wait_for(parked)
        await runtime.spawn(_CHILD, "c", "build a chair", parent_thread_id="p")

        async def failed() -> bool:
            thread = await stores.threads.get_or_create(_CHILD, "c")
            return thread.status is ThreadStatus.FAILED

        await _wait_for(failed)
        [run] = await stores.runs.list_for_thread(_CHILD, "c")
        # The report attempted again while the parent still holds the first in its inbox.
        await RunActivities(runtime._context).report_to_parent(
            ReportToParentInput(_CHILD, "c", run.id, RunOutcome.FAILED.value, run.stop_reason)
        )
        await runtime.resolve_tool_call(_PARENT, "p", "tc_approve", approved=True)
        await _wait_for(lambda: _said(stores, "noted the failure"))

        [report] = await _reports(stores)
        assert 'status="failed"' in report and "FakeLLM has no queued responses" in report


# === in process ===


async def test_in_process_a_child_wakes_its_closed_parent() -> None:
    stores = InMemoryRuntimeStores()
    parent = _agent(_PARENT, [FakeResponse(text="hello"), FakeResponse(text="got it")])
    child = _agent(_CHILD, [FakeResponse(text="the chair is done")])
    runtime = LocalThreadRuntime(
        ActivityContext(
            stores=stores, resolve_agent=static_agents({_PARENT: parent, _CHILD: child})
        )
    )
    await runtime.run(ThreadInput(_PARENT, "p"))
    await runtime.run(ThreadInput(_CHILD, "c", parent_thread_id="p"))
    await runtime.idle()

    assert await _said(stores, "got it")
    [report] = await _reports(stores)
    assert 'status="completed"' in report and "the chair is done" in report


async def test_in_process_a_failed_child_reaches_its_running_parent_after_its_run() -> None:
    stores = InMemoryRuntimeStores()
    runtimes: list[LocalThreadRuntime] = []

    @tool
    async def delegate() -> str:
        """Hand the chair to the author."""
        await runtimes[0].run(ThreadInput(_CHILD, "c", parent_thread_id="p"))
        return "delegated"

    parent = _agent(
        _PARENT,
        [_call("delegate"), FakeResponse(text="waiting"), FakeResponse(text="noted the failure")],
        tools=[delegate],
    )
    child = _agent(_CHILD, [])
    runtimes.append(
        LocalThreadRuntime(
            ActivityContext(
                stores=stores, resolve_agent=static_agents({_PARENT: parent, _CHILD: child})
            )
        )
    )
    outcome = await runtimes[0].run(ThreadInput(_PARENT, "p"))

    assert outcome.outcome is RunOutcome.COMPLETED
    assert await _said(stores, "waiting") and await _said(stores, "noted the failure")
    [report] = await _reports(stores)
    assert 'status="failed"' in report and "FakeLLM has no queued responses" in report
