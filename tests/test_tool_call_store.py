"""``ToolCallStore`` — tool_call lifecycle persistence.

Tool_calls are the unit the orchestrator schedules and the executor
runs. Their status transitions (REQUESTED → RUNNING → COMPLETED |
FAILED | BLOCKED, plus REQUESTED → WAITING for deferred tools) are
the spine of group continuation. The orphan bug we're hunting
shows up here as REQUESTED records that nobody promotes to RUNNING.

One behavior per test.
"""

from __future__ import annotations

import pytest

from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools.calls import ToolCallRecord, ToolCallStatus


def _record(
    tc_id: str = "tc_1",
    *,
    group_id: str = "g_1",
    run_id: str = "run_1",
    thread_id: str = "t",
    turn_id: str = "turn_1",
    agent_id: str = "a",
    name: str = "echo",
    status: ToolCallStatus = ToolCallStatus.REQUESTED,
) -> ToolCallRecord:
    return ToolCallRecord(
        id=tc_id,
        group_id=group_id,
        run_id=run_id,
        agent_id=agent_id,
        thread_id=thread_id,
        turn_id=turn_id,
        turn_index=1,
        name=name,
        args={},
        status=status,
    )


# --- save / get round-trip ---


@pytest.mark.asyncio
async def test_save_then_get_returns_same_record() -> None:
    stores = InMemoryRuntimeStores()
    rec = _record()
    await stores.tool_calls.save(rec)

    fetched = await stores.tool_calls.get("tc_1")

    assert fetched.id == "tc_1"
    assert fetched.status == ToolCallStatus.REQUESTED
    assert fetched.name == "echo"


@pytest.mark.asyncio
async def test_default_status_is_requested() -> None:
    """The dataclass default is REQUESTED — the runtime relies on
    every freshly-written record being orchestrator-schedulable
    until update_status moves it forward."""
    rec = ToolCallRecord(
        id="tc",
        group_id="g",
        run_id="r",
        agent_id="a",
        thread_id="t",
        turn_id="turn",
        turn_index=1,
        name="x",
        args={},
    )
    assert rec.status == ToolCallStatus.REQUESTED


# --- status transitions ---


@pytest.mark.asyncio
async def test_update_status_promotes_requested_to_running() -> None:
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record())

    await stores.tool_calls.update_status("tc_1", ToolCallStatus.RUNNING)

    fetched = await stores.tool_calls.get("tc_1")
    assert fetched.status == ToolCallStatus.RUNNING


@pytest.mark.asyncio
async def test_update_status_to_completed_sets_result() -> None:
    """The orchestrator continuation reads ``record.result`` to
    append it as the tool_result message — has to land via
    update_status with the result kwarg."""
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record(status=ToolCallStatus.RUNNING))

    await stores.tool_calls.update_status(
        "tc_1",
        ToolCallStatus.COMPLETED,
        result={"output": "ok"},
    )

    fetched = await stores.tool_calls.get("tc_1")
    assert fetched.status == ToolCallStatus.COMPLETED
    assert fetched.result == {"output": "ok"}


@pytest.mark.asyncio
async def test_update_status_to_waiting_records_prompt_and_wait_request() -> None:
    """Deferred tools (the WAIT decision path) stash both a human
    prompt and the structured wait_request. Both have to round-trip."""
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record(status=ToolCallStatus.RUNNING))

    await stores.tool_calls.update_status(
        "tc_1",
        ToolCallStatus.WAITING,
        prompt="approve?",
        wait_request={"kind": "approval"},
    )

    fetched = await stores.tool_calls.get("tc_1")
    assert fetched.status == ToolCallStatus.WAITING
    assert fetched.prompt == "approve?"
    assert fetched.wait_request == {"kind": "approval"}


@pytest.mark.asyncio
async def test_update_status_keeps_optional_fields_none_by_default() -> None:
    """A plain status update doesn't accidentally clear prior result/
    prompt/wait_request — only sets them when explicitly provided."""
    stores = InMemoryRuntimeStores()
    rec = _record()
    rec.result = {"prior": True}
    await stores.tool_calls.save(rec)

    await stores.tool_calls.update_status("tc_1", ToolCallStatus.COMPLETED)

    fetched = await stores.tool_calls.get("tc_1")
    assert fetched.status == ToolCallStatus.COMPLETED
    assert fetched.result == {"prior": True}


# --- group / run / open queries ---


@pytest.mark.asyncio
async def test_get_group_returns_only_records_in_group() -> None:
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record("tc_a", group_id="g_1"))
    await stores.tool_calls.save(_record("tc_b", group_id="g_1"))
    await stores.tool_calls.save(_record("tc_c", group_id="g_2"))

    group_1 = await stores.tool_calls.get_group("g_1")

    assert sorted(r.id for r in group_1) == ["tc_a", "tc_b"]


@pytest.mark.asyncio
async def test_get_by_run_filters_by_run_id() -> None:
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record("tc_a", run_id="run_1"))
    await stores.tool_calls.save(_record("tc_b", run_id="run_1"))
    await stores.tool_calls.save(_record("tc_c", run_id="run_2"))

    run_1 = await stores.tool_calls.get_by_run("run_1")

    assert sorted(r.id for r in run_1) == ["tc_a", "tc_b"]


# --- open_for_thread (used by terminate_thread) ---


@pytest.mark.asyncio
async def test_get_open_for_thread_includes_requested_running_waiting() -> None:
    """terminate_thread needs to find every non-terminal call to
    write a placeholder result for it. Open states are exactly
    REQUESTED, RUNNING, WAITING."""
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record("tc_req", status=ToolCallStatus.REQUESTED))
    await stores.tool_calls.save(_record("tc_run", status=ToolCallStatus.RUNNING))
    await stores.tool_calls.save(_record("tc_wait", status=ToolCallStatus.WAITING))

    open_calls = await stores.tool_calls.get_open_for_thread("a", "t")

    assert sorted(r.id for r in open_calls) == ["tc_req", "tc_run", "tc_wait"]


@pytest.mark.asyncio
async def test_get_open_for_thread_excludes_terminal_states() -> None:
    """Completed/failed/blocked already have results — terminate_thread
    must skip them or it'd overwrite real outputs with placeholders."""
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record("tc_done", status=ToolCallStatus.COMPLETED))
    await stores.tool_calls.save(_record("tc_fail", status=ToolCallStatus.FAILED))
    await stores.tool_calls.save(_record("tc_block", status=ToolCallStatus.BLOCKED))

    open_calls = await stores.tool_calls.get_open_for_thread("a", "t")

    assert open_calls == []


@pytest.mark.asyncio
async def test_get_open_for_thread_filters_by_agent_and_thread() -> None:
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record("tc_match", agent_id="a", thread_id="t"))
    await stores.tool_calls.save(_record("tc_other_agent", agent_id="b", thread_id="t"))
    await stores.tool_calls.save(_record("tc_other_thread", agent_id="a", thread_id="other"))

    open_calls = await stores.tool_calls.get_open_for_thread("a", "t")

    assert [r.id for r in open_calls] == ["tc_match"]
