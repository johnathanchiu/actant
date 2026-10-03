"""A message sent to a running thread, with and without ``interleave_inbox``.

Each scenario parks a run on a tool that awaits a person, sends a message
while it waits, then answers the tool. The message is therefore in the inbox
before the run's next turn, and the only question is which turn sees it.

The replay tests hold the flag to its promise: a workflow change that is off
by default must leave every history recorded without it replaying unchanged.
"""

from __future__ import annotations

import asyncio
import uuid

from temporalio.client import Client, WorkflowHandle, WorkflowHistory
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker

from actant.llm.messages import Message
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import TemporalRuntimeActivities
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.types import (
    TemporalRuntimeConfig,
)
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.tools.calls import ToolCallStatus
from runtime_fixtures import static_agents
from test_workflow_thread import _AGENT, _THREAD, _ApprovalTool, _agent, _tool_call, _wait_for


def _roles_and_text(messages: list[Message]) -> list[tuple[str, str | None]]:
    return [
        (m.role, m.content if isinstance(m.content, str) and m.role != "tool" else None)
        for m in messages
    ]


async def _message_while_a_tool_waits(
    *, interleave_inbox: bool
) -> tuple[InMemoryRuntimeStores, FakeLLM, WorkflowHistory]:
    """Run the scenario through ``AgentRuntime`` and return what it left."""
    call = _tool_call("needs_approval")
    llm = FakeLLM(
        [
            FakeResponse(tool_calls=[call]),
            FakeResponse(text="after tool"),
            FakeResponse(text="second run"),
        ]
    )
    agent = _agent(llm, tools=[_ApprovalTool()])
    stores = InMemoryRuntimeStores()
    activities = TemporalRuntimeActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({agent.id: agent}))
    )
    task_queue = f"test-actant-{uuid.uuid4().hex[:8]}"

    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            config = TemporalRuntimeConfig(
                task_queue=task_queue,
                address=env.client.service_client.config.target_host,
                namespace=env.client.namespace,
                interleave_inbox=interleave_inbox,
            )
            runtime = AgentRuntime(client=env.client, stores=stores, config=config)
            await runtime.send_message(_AGENT, _THREAD, "first")

            async def waiting() -> bool:
                try:
                    record = await stores.tool_calls.get(call.id)
                except KeyError:
                    return False
                return record.status == ToolCallStatus.WAITING

            await _wait_for(waiting)
            await runtime.send_message(_AGENT, _THREAD, "mid-run")
            await runtime.resolve_tool_call(_AGENT, _THREAD, call.id, approved=True)

            handle = _handle(env.client, config)
            await asyncio.wait_for(handle.result(), timeout=20.0)
            history = await handle.fetch_history()
    return stores, llm, history


def _handle(client: Client, config: TemporalRuntimeConfig) -> WorkflowHandle[object, object]:
    return client.get_workflow_handle(f"{config.workflow_id_prefix}-{_AGENT}-{_THREAD}")


async def test_a_message_sent_mid_run_reaches_the_next_turn_of_that_run() -> None:
    stores, llm, _ = await _message_while_a_tool_waits(interleave_inbox=True)

    # Two model calls, one run: the message did not wait for a second run.
    assert len(llm.calls) == 2
    assert len(await stores.runs.list_for_thread(_AGENT, _THREAD)) == 1

    # The second call saw it, after the tool result of the first.
    _, seen, _ = llm.calls[1]
    assert _roles_and_text(seen) == [
        ("user", "first"),
        ("assistant", None),
        ("tool", None),
        ("user", "mid-run"),
    ]
    stored = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert _roles_and_text(stored) == [*_roles_and_text(seen), ("assistant", "after tool")]


async def test_a_mid_run_message_follows_every_tool_result_of_the_turn_before() -> None:
    stores, llm, _ = await _message_while_a_tool_waits(interleave_inbox=True)

    _, seen, _ = llm.calls[1]
    calls = seen[1].tool_calls or []
    first_user_after_calls = next(i for i, m in enumerate(seen) if i > 1 and m.role == "user")
    answered = {m.tool_call_id for m in seen[2:first_user_after_calls] if m.role == "tool"}
    assert {c.id for c in calls} == answered


async def test_without_the_flag_a_mid_run_message_waits_for_the_next_run() -> None:
    stores, llm, _ = await _message_while_a_tool_waits(interleave_inbox=False)

    assert len(llm.calls) == 3
    assert len(await stores.runs.list_for_thread(_AGENT, _THREAD)) == 2
    _, seen, _ = llm.calls[1]
    assert all(m.content != "mid-run" for m in seen)
    stored = await stores.messages.list_for_thread(_AGENT, _THREAD)
    assert _roles_and_text(stored) == [
        ("user", "first"),
        ("assistant", None),
        ("tool", None),
        ("assistant", "after tool"),
        ("user", "mid-run"),
        ("assistant", "second run"),
    ]


async def test_histories_with_and_without_the_flag_replay() -> None:
    for interleave_inbox in (False, True):
        _, _, history = await _message_while_a_tool_waits(interleave_inbox=interleave_inbox)
        await Replayer(workflows=[AgentThreadWorkflow]).replay_workflow(history)
