"""TaskTool's parent_thread_id resolution, and what a spawn returns.

``parent_thread_id`` is optional at construction: the tool falls back to
``call.thread_id`` from the per-call ``ToolCallView``, so one ``TaskTool``
can be shared across many threads in one ``AgentDefinition``.

Admission only validates. The spawn itself happens in ``execute``, and
returns the sub-thread's id rather than its answer -- the parent gets a
handle it can supervise instead of stopping until the subagent is done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from typing import Any, cast

from actant.core import JSONObject
from actant.runtime.events.lifecycle import PublishingThreadHooks
from actant.tools.admission import ToolDecisionKind
from actant.tools.task import TaskTool
from actant.tools.base import CallContext


def _ctx(thread_id: str = "thread-1", **overrides: object) -> CallContext:
    values: dict[str, object] = {
        "agent_id": "demo",
        "thread_id": thread_id,
        "run_id": "run-1",
        "tool_call_id": "tc-1",
        "turn_id": "turn-1",
    }
    values.update(overrides)
    return CallContext(**values)  # type: ignore[arg-type]


@dataclass
class _RecordedSpawn:
    name: str
    message: str
    context: JSONObject
    parent_thread_id: str


@dataclass
class _CapturingSpawner:
    """Test double for SubagentSpawner that records every spawn call."""

    spawns: list[_RecordedSpawn] = field(default_factory=list)

    async def spawn(
        self,
        *,
        name: str,
        message: str,
        context: JSONObject,
        parent_thread_id: str,
    ) -> str:
        self.spawns.append(
            _RecordedSpawn(
                name=name,
                message=message,
                context=context,
                parent_thread_id=parent_thread_id,
            )
        )
        return f"sub_{len(self.spawns)}"


@dataclass
class _FakeCall:
    """Minimal ToolCallView for can_execute."""

    id: str
    thread_id: str
    agent_id: str = "demo"
    group_id: str = "g_1"
    run_id: str = "r_1"
    turn_id: str = "turn_1"
    turn_index: int = 0
    name: str = "task"
    args: JSONObject = field(default_factory=dict)


async def test_construction_time_parent_thread_id_wins() -> None:
    """If the app pins parent_thread_id at construction, that value is
    used regardless of ``call.thread_id``. Backwards-compatible with
    pre-v0.2 per-thread agents."""
    spawner = _CapturingSpawner()
    tool = TaskTool(spawner=spawner, parent_thread_id="thread_constructed")
    call = _FakeCall(
        id="tc_1",
        thread_id="thread_different",  # different from construction
        args={"subagent": "researcher", "message": "do a thing"},
    )
    decision = await tool.can_execute(call, None, None)
    assert decision.kind == ToolDecisionKind.EXECUTE
    # Admission does not spawn: it runs for calls that never execute.
    assert spawner.spawns == []

    result = await (await tool.build(call.args, _ctx(call.thread_id))).execute()
    assert len(spawner.spawns) == 1
    assert spawner.spawns[0].parent_thread_id == "thread_constructed"
    assert result.output == {
        "subagent": "researcher",
        "thread_id": "sub_1",
        "sub_thread_id": "sub_1",
        "status": "running",
    }
    # In the output rather than metadata on purpose: the tool_result event
    # carries only output, so a viewer cannot see metadata at all.
    assert result.metadata == {}


async def test_per_call_thread_id_fallback() -> None:
    """When parent_thread_id is unset at construction, the tool reads
    ``call.thread_id`` from each invocation. Enables a single
    AgentDefinition to be shared across many threads."""
    spawner = _CapturingSpawner()
    tool = TaskTool(spawner=spawner)  # no parent_thread_id
    call_a = _FakeCall(
        id="tc_a",
        thread_id="thread_alpha",
        args={"subagent": "researcher", "message": "task A"},
    )
    call_b = _FakeCall(
        id="tc_b",
        thread_id="thread_beta",
        args={"subagent": "researcher", "message": "task B"},
    )
    assert (await tool.can_execute(call_a, None, None)).kind == ToolDecisionKind.EXECUTE
    assert (await tool.can_execute(call_b, None, None)).kind == ToolDecisionKind.EXECUTE

    await (await tool.build(call_a.args, _ctx(call_a.thread_id))).execute()
    await (await tool.build(call_b.args, _ctx(call_b.thread_id))).execute()
    assert len(spawner.spawns) == 2
    assert spawner.spawns[0].parent_thread_id == "thread_alpha"
    assert spawner.spawns[1].parent_thread_id == "thread_beta"


async def test_blocks_when_thread_id_missing_everywhere() -> None:
    """Defensive: if neither construction-time nor call-time thread_id
    is available, the tool blocks the call with a clear reason instead
    of crashing or spawning into the void."""
    spawner = _CapturingSpawner()
    tool = TaskTool(spawner=spawner)

    @dataclass
    class _NoThreadCall:
        id: str = "tc_x"
        thread_id: str = ""  # explicitly empty
        agent_id: str = "demo"
        group_id: str = "g_1"
        run_id: str = "r_1"
        turn_id: str = "turn_1"
        turn_index: int = 0
        name: str = "task"
        args: JSONObject = field(default_factory=lambda: {"subagent": "x", "message": "y"})

    decision = await tool.can_execute(_NoThreadCall(), None, None)
    assert decision.kind == ToolDecisionKind.DENY
    assert "parent_thread_id" in decision.reason
    assert len(spawner.spawns) == 0


async def test_sync_mode_still_allows_no_parent_thread_id() -> None:
    """Sync mode (invoker, not spawner) never needed parent_thread_id —
    still doesn't."""

    class _Invoker:
        async def invoke(self, name: str, message: str, context: JSONObject):
            from actant.tools.base import ToolResult

            return ToolResult.ok({"name": name, "message": message})

    tool = TaskTool(invoker=_Invoker())
    call = _FakeCall(
        id="tc_sync",
        thread_id="thread_sync",
        args={"subagent": "x", "message": "y"},
    )
    decision = await tool.can_execute(call, None, None)
    assert decision.kind == ToolDecisionKind.EXECUTE


@pytest.mark.asyncio
async def test_the_sub_thread_id_survives_the_event_a_viewer_sees() -> None:
    """A viewer must be able to link the child to the call that started it.

    This asserts through the event, not the ToolResult, because that is
    where it broke: ``on_tool_result`` publishes ``output`` and ``error``
    and nothing else, so anything put in ``metadata`` is invisible to a UI
    until the page is reloaded and history is read from the store instead.
    """
    published: list[JSONObject] = []

    class _Publisher:
        async def publish(self, channel: str, event: JSONObject) -> None:
            published.append(event)

    spawner = _CapturingSpawner()
    tool = TaskTool(spawner=spawner, parent_thread_id="thread_1")
    call = _FakeCall(id="tc_1", thread_id="thread_1", args={"subagent": "r", "message": "go"})

    result = await (await tool.build(call.args, _ctx(call.thread_id))).execute()
    hooks = PublishingThreadHooks("thread_1", _Publisher())
    await hooks.on_tool_result("tc_1", result)

    assert published, "the tool result was published"
    # The key, not the value: ``thread_id`` carries the same string, so
    # asserting on the value alone passes even when sub_thread_id is absent.
    # A viewer looks for this key to attach the child to the call.
    assert "sub_thread_id" in str(published[0]["data"]), (
        "a viewer can find the sub-thread id in the event it actually receives"
    )


@pytest.mark.asyncio
async def test_the_runtime_hands_a_tool_its_call_context() -> None:
    """Every build receives the call's context, so delegation knows its thread.

    Exercised through the activity's own context builder rather than
    ``TaskTool.build`` directly: if the runtime stopped passing the thread,
    every delegation would fail at runtime against a fully green suite.
    """
    from actant.runtime.stores.in_memory import InMemoryRuntimeStores
    from actant.runtime.temporal.activities.tools import ToolActivities
    from actant.tools.calls import ToolCallRecord

    spawner = _CapturingSpawner()
    tool = TaskTool(spawner=spawner)
    record = _FakeCall(
        id="tc_1",
        thread_id="thread_from_the_record",
        args={"subagent": "researcher", "message": "go"},
    )
    stores = InMemoryRuntimeStores()
    await stores.threads.get_or_create("demo", "thread_from_the_record")
    activities = ToolActivities(stores=stores, agents={})
    agent = cast(Any, None)  # no sandbox is requested, so the agent is never consulted

    ctx = await activities._call_context(agent, tool, cast(ToolCallRecord, record))
    invocation = await tool.build(record.args, ctx)
    result = await invocation.execute()

    assert result.error is None, result.error
    assert spawner.spawns[0].parent_thread_id == "thread_from_the_record", (
        "the runtime handed the tool its call context, not just the arguments"
    )


@pytest.mark.asyncio
async def test_a_subagent_cannot_spawn_subagents() -> None:
    """Delegation is one level deep: a thread with a parent is refused."""
    tool = TaskTool(spawner=_CapturingSpawner())
    with pytest.raises(ValueError, match="cannot spawn"):
        await tool.build({"name": "x", "message": "y"}, _ctx("child-1", parent_thread_id="root"))
