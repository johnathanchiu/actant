"""Cancellation follows stored lineage, not the worker that spawned a child."""

from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from temporalio.client import Client
from temporalio.service import RPCError, RPCStatusCode

from actant.runtime import AgentRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities.threads import ThreadActivities
from actant.runtime.temporal.types import ApplyThreadCancellationInput
from actant.runtime.types.threads import ThreadStatus


@pytest.mark.parametrize("cleanup", [False, True])
async def test_cancellation_reaches_descendants_on_a_fresh_runtime(cleanup: bool) -> None:
    stores = InMemoryRuntimeStores()
    for name, parent, status in [
        ("root", None, ThreadStatus.ACTIVE),
        ("child", "root", ThreadStatus.ACTIVE),
        ("finished", "root", ThreadStatus.IDLE),
        ("grandchild", "finished", ThreadStatus.ACTIVE),
        ("unrelated", None, ThreadStatus.ACTIVE),
    ]:
        thread = await stores.threads.get_or_create(name, name)
        thread.parent_thread_id = parent
        thread.status = status
        await stores.threads.update(thread)
    handles = {name: Mock(cancel=AsyncMock()) for name in ["root", "child", "grandchild"]}
    # A closed parent must not hide its still-running children.
    handles["root"].cancel.side_effect = RPCError("closed", RPCStatusCode.NOT_FOUND, b"")
    client = Mock()
    runtime = AgentRuntime(client=cast(Client, client), stores=stores)
    client.get_workflow_handle.side_effect = lambda wid: handles[wid.rsplit("-", 1)[-1]]
    if cleanup:
        await ThreadActivities(runtime._context).apply_thread_cancellation(
            ApplyThreadCancellationInput(agent_id="root", thread_id="root")
        )
    else:
        await runtime.cancel_thread("root", "root")
    handles["child"].cancel.assert_awaited_once()
    handles["grandchild"].cancel.assert_awaited_once()
    assert (await stores.threads.get("finished", "finished")).status == ThreadStatus.IDLE
    assert (await stores.threads.get("unrelated", "unrelated")).status == ThreadStatus.ACTIVE
    children = await stores.threads.list_children("root")
    children[0].parent_thread_id = None
    assert len(await stores.threads.list_children("root")) == 2
