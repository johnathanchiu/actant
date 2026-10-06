"""A cancelled workflow's surviving model activity must not poison its next run."""

import asyncio

import pytest
from runtime_fixtures import static_agents
from test_workflow_thread import _agent, _tool_call

from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.llm.providers.openai import OpenAIProvider
from actant.runtime.cancellation import record_cancelled
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.activities.runs import RunActivities
from actant.runtime.temporal.types import RunTurnInput
from actant.runtime.types.threads import RunStatus, ThreadStatus


class LateLLM(FakeLLM):
    def __init__(self):
        super().__init__([FakeResponse(tool_calls=[_tool_call("echo")])])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, *args, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().complete(*args, **kwargs)


@pytest.mark.parametrize("during_append", [False, True])
@pytest.mark.parametrize("restarted", [False, True])
@pytest.mark.xfail(strict=True, reason="Late model activity commits after cancellation repair")
async def test_cancelled_turn_cannot_leave_orphan_call(during_append, restarted):
    llm = LateLLM()
    agent = _agent(llm)
    stores = InMemoryRuntimeStores()
    await stores.runs.create(agent.id, "thread", run_id="old", max_turns=5)
    thread = await stores.threads.get_or_create(agent.id, "thread")
    thread.active_run_id = "old"
    thread.status = ThreadStatus.ACTIVE
    await stores.threads.update(thread)
    activities = RunActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({agent.id: agent}))
    )

    async def cancel_and_restart():
        await stores.runs.finish("old", RunStatus.CANCELLED)
        await record_cancelled(stores, agent.id, "thread")
        if restarted:
            await stores.runs.create(agent.id, "thread", run_id="new", max_turns=5)
            thread = await stores.threads.get_or_create(agent.id, "thread")
            thread.active_run_id = "new"
            thread.status = ThreadStatus.ACTIVE
            await stores.threads.update(thread)

    if during_append:
        append = stores.messages.append_assistant_with_tool_calls

        async def delayed_append(*args, **kwargs):
            # Cancellation sweeps before the old activity commits its new call.
            await cancel_and_restart()
            return await append(*args, **kwargs)

        stores.messages.append_assistant_with_tool_calls = delayed_append

    task = asyncio.create_task(
        activities.run_turn(RunTurnInput(agent.id, "thread", "old", "turn", 1))
    )
    await llm.started.wait()
    if not during_append:
        await cancel_and_restart()
    llm.release.set()
    await task

    messages = await stores.messages.list_for_thread(agent.id, "thread")
    wire = OpenAIProvider.convert_messages(messages)
    calls = {item["call_id"] for item in wire if item["type"] == "function_call"}
    outputs = {item["call_id"] for item in wire if item["type"] == "function_call_output"}
    assert calls == outputs
    assert (await stores.runs.get("old")).status == RunStatus.CANCELLED
    thread = await stores.threads.get_or_create(agent.id, "thread")
    assert thread.active_run_id == ("new" if restarted else None)
    assert thread.status == (ThreadStatus.ACTIVE if restarted else ThreadStatus.CANCELLED)
