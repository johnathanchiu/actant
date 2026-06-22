"""TaskTool's parent_thread_id resolution.

v0.2 makes ``parent_thread_id`` optional at construction. The tool
falls back to ``call.thread_id`` from the per-call ``ToolCallView``,
so a single ``TaskTool`` instance can be shared across many threads
in one ``AgentDefinition``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from actant.core import JSONObject
from actant.tools.admission import ToolDecisionKind
from actant.tools.task import TaskTool


@dataclass
class _RecordedSpawn:
    name: str
    message: str
    context: JSONObject
    parent_thread_id: str
    parent_tool_call_id: str


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
        parent_tool_call_id: str,
    ) -> None:
        self.spawns.append(
            _RecordedSpawn(
                name=name,
                message=message,
                context=context,
                parent_thread_id=parent_thread_id,
                parent_tool_call_id=parent_tool_call_id,
            )
        )


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
    assert decision.kind == ToolDecisionKind.WAIT
    assert len(spawner.spawns) == 1
    assert spawner.spawns[0].parent_thread_id == "thread_constructed"


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
    decision_a = await tool.can_execute(call_a, None, None)
    decision_b = await tool.can_execute(call_b, None, None)
    assert decision_a.kind == ToolDecisionKind.WAIT
    assert decision_b.kind == ToolDecisionKind.WAIT
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
    assert decision.kind == ToolDecisionKind.BLOCK
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
    assert decision.kind == ToolDecisionKind.ALLOW
