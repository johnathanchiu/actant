"""A tool's durable call identity and uncertain drains survive activity boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from temporalio.exceptions import ApplicationError

from actant.agents import AgentDefinition
from actant.llm.providers.fake import FakeLLM
from actant.runtime.stores.in_memory import InMemoryRuntimeStores
from actant.runtime.temporal.activities.context import ActivityContext
from actant.runtime.temporal.activities.tools import ToolActivities
from actant.runtime.temporal.types import (
    AdmitInput,
    DeferredToolResolution,
    ExecuteInput,
    ResolveToolInput,
)
from actant.sandbox.base import Sandbox
from actant.sandbox.protocol import CallResponse
from actant.sandbox.service import ServiceDrainError, service_call_id
from actant.tools.base import CallContext
from actant.tools.calls import ToolCallRecord, ToolCallStatus
from actant.tools.registry import ToolRegistry
from actant.tools.service import ServiceTool
from runtime_fixtures import static_agents


class Ping:
    async def ping(self) -> str:
        return "pong"


class RecordingRunner:
    needs_sandbox = False

    def __init__(self, error: BaseException | None = None) -> None:
        self.ids: list[str | None] = []
        self.error = error

    async def call(
        self,
        method: str,
        args: Mapping[str, object],
        *,
        key: str | None = None,
        sandbox: Sandbox | None = None,
    ) -> CallResponse:
        identity = service_call_id.get()
        self.ids.append(identity)
        await asyncio.sleep(0)
        assert service_call_id.get() == identity
        if self.error is not None:
            raise self.error
        return CallResponse(text="pong")


@pytest.mark.parametrize("error", [None, RuntimeError("failure"), asyncio.CancelledError()])
async def test_identity_repeats_across_attempts_and_resets_even_when_cancelled(error):
    runner = RecordingRunner(error)
    tool = ServiceTool(Ping, "ping", runner)
    context = CallContext("a", "thread", "run", "call", "turn")
    token = service_call_id.set("outer-call")
    try:
        for _ in range(2):
            invocation = await tool.build({}, context)
            if error is None:
                assert (await invocation.execute()).output == "pong"
            else:
                with pytest.raises(type(error)):
                    await invocation.execute()
            assert service_call_id.get() == "outer-call"
        assert runner.ids == ["thread:call", "thread:call"]
    finally:
        service_call_id.reset(token)


async def test_concurrent_service_calls_keep_distinct_context_ids():
    runner = RecordingRunner()
    tool = ServiceTool(Ping, "ping", runner)
    first = await tool.build({}, CallContext("a", "first", "r", "call", "turn"))
    second = await tool.build({}, CallContext("a", "second", "r", "call", "turn"))
    await asyncio.gather(first.execute(), second.execute())
    assert set(runner.ids) == {"first:call", "second:call"}
    assert service_call_id.get() is None


@pytest.mark.parametrize("phase", ["build", "execute", "interrupted_retry"])
async def test_uncertain_drain_is_nonretryable_infrastructure_failure(phase, monkeypatch):
    runner = RecordingRunner(ServiceDrainError("remote effects still running"))
    tool = ServiceTool(Ping, "ping", runner)
    if phase == "build":
        monkeypatch.setattr(tool, "build", AsyncMock(side_effect=runner.error))
    elif phase == "interrupted_retry":
        from actant.runtime.temporal.activities import tools as module

        monkeypatch.setattr(module.activity, "in_activity", lambda: True)
        monkeypatch.setattr(module.activity, "info", lambda: SimpleNamespace(attempt=2))
    agent = AgentDefinition(
        id="a", name="test", persona="", llm=FakeLLM([]), tools=ToolRegistry([tool])
    )
    stores = InMemoryRuntimeStores()
    await stores.threads.get_or_create("a", "thread")
    record = ToolCallRecord(
        id="call",
        group_id="group",
        run_id="run",
        agent_id="a",
        thread_id="thread",
        turn_id="turn",
        turn_index=1,
        name="ping",
        args={},
        status=ToolCallStatus.RUNNING,
    )
    await stores.tool_calls.save(record)
    activities = ToolActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({"a": agent}))
    )
    with pytest.raises(ApplicationError) as raised:
        await activities.execute_tool(
            ExecuteInput(agent_id="a", thread_id="thread", run_id="run", tool_call_id="call")
        )
    assert raised.value.type == "ServiceDrainError" and raised.value.non_retryable
    assert isinstance(raised.value.__cause__, ServiceDrainError)
    persisted = await stores.tool_calls.get("call")
    assert persisted.status == ToolCallStatus.RUNNING and persisted.result is None
    if phase == "interrupted_retry":
        assert not runner.ids


@pytest.mark.parametrize("boundary", ["admit", "resolve"])
async def test_admission_and_resolution_do_not_hide_uncertain_service_drain(boundary, monkeypatch):
    tool = ServiceTool(Ping, "ping", RecordingRunner())
    error = ServiceDrainError("callback remote drain unconfirmed")
    if boundary == "admit":
        monkeypatch.setattr(tool, "build", AsyncMock(side_effect=error))
    else:
        monkeypatch.setattr(tool, "on_resolve", AsyncMock(side_effect=error), raising=False)
    agent = AgentDefinition(
        id="a", name="test", persona="", llm=FakeLLM([]), tools=ToolRegistry([tool])
    )
    stores = InMemoryRuntimeStores()
    await stores.threads.get_or_create("a", "thread")
    status = ToolCallStatus.REQUESTED if boundary == "admit" else ToolCallStatus.WAITING
    await stores.tool_calls.save(
        ToolCallRecord(
            id="call",
            group_id="group",
            run_id="run",
            agent_id="a",
            thread_id="thread",
            turn_id="turn",
            turn_index=1,
            name="ping",
            args={},
            status=status,
        )
    )
    activities = ToolActivities(
        ActivityContext(stores=stores, resolve_agent=static_agents({"a": agent}))
    )
    with pytest.raises(ApplicationError) as raised:
        if boundary == "admit":
            await activities.admit_tool(
                AdmitInput(agent_id="a", thread_id="thread", run_id="run", tool_call_id="call")
            )
        else:
            await activities.resolve_tool(
                ResolveToolInput(
                    agent_id="a",
                    thread_id="thread",
                    run_id="run",
                    tool_call_id="call",
                    resolution=DeferredToolResolution(tool_call_id="call", approved=True),
                )
            )
    assert raised.value.type == "ServiceDrainError" and raised.value.non_retryable
    record = await stores.tool_calls.get("call")
    assert record.status == status and record.result is None
