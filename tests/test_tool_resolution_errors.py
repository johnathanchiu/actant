from __future__ import annotations

from typing import cast
from temporalio.client import Client
from unittest.mock import Mock

import pytest

from actant.runtime.exceptions import ToolCallNotFoundError, ToolCallNotWaitingError
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime import AgentRuntime
from actant.tools.calls import ToolCallRecord, ToolCallStatus


def _record(*, status: ToolCallStatus) -> ToolCallRecord:
    return ToolCallRecord(
        id="tc_1",
        group_id="group_1",
        run_id="run_1",
        agent_id="agent_1",
        thread_id="thread_1",
        turn_id="turn_1",
        turn_index=1,
        name="approval",
        args={},
        status=status,
    )


def _client(stores: InMemoryRuntimeStores) -> AgentRuntime:
    return AgentRuntime(client=cast(Client, Mock()), stores=stores)


@pytest.mark.asyncio
async def test_missing_tool_call_is_typed_not_found() -> None:
    stores = InMemoryRuntimeStores()
    with pytest.raises(ToolCallNotFoundError):
        await _client(stores).resolve_tool_call("agent_1", "thread_1", "missing")


@pytest.mark.asyncio
async def test_wrong_thread_is_indistinguishable_from_not_found() -> None:
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record(status=ToolCallStatus.WAITING))
    with pytest.raises(ToolCallNotFoundError):
        await _client(stores).resolve_tool_call("agent_1", "another_thread", "tc_1")


@pytest.mark.asyncio
async def test_non_waiting_tool_call_is_a_typed_conflict() -> None:
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record(status=ToolCallStatus.RUNNING))
    with pytest.raises(ToolCallNotWaitingError) as error:
        await _client(stores).resolve_tool_call("agent_1", "thread_1", "tc_1")
    assert error.value.status is ToolCallStatus.RUNNING


@pytest.mark.asyncio
async def test_terminal_resolution_is_idempotent_for_owning_thread() -> None:
    stores = InMemoryRuntimeStores()
    await stores.tool_calls.save(_record(status=ToolCallStatus.COMPLETED))
    await _client(stores).resolve_tool_call("agent_1", "thread_1", "tc_1")
