"""Thread-scoped runtime API tests."""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from actant.runtime.runtime import AgentRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.thread import ThreadHandle, ThreadRuntime


def _handle() -> tuple[ThreadHandle, Mock, InMemoryRuntimeStores]:
    stores = InMemoryRuntimeStores()
    runtime = Mock()
    runtime.stores = stores
    runtime.event_source = stores.publisher
    runtime.send_message = AsyncMock(return_value="workflow-id")
    runtime.resolve_tool_call = AsyncMock()
    runtime.cancel_thread = AsyncMock()
    runtime.get_state = AsyncMock()
    handle = ThreadHandle(cast(ThreadRuntime, runtime), "assistant", "thread-id")
    return handle, runtime, stores


@pytest.mark.asyncio
async def test_thread_handle_scopes_runtime_commands() -> None:
    handle, runtime, _stores = _handle()

    workflow_id = await handle.send("Hello")
    await handle.resolve("tool-id", approved=True)
    await handle.cancel()

    assert workflow_id == "workflow-id"
    runtime.send_message.assert_awaited_once_with("assistant", "thread-id", "Hello")
    runtime.resolve_tool_call.assert_awaited_once_with(
        "assistant",
        "thread-id",
        "tool-id",
        approved=True,
        answer="",
        payload=None,
    )
    runtime.cancel_thread.assert_awaited_once_with("assistant", "thread-id")


@pytest.mark.asyncio
async def test_thread_handle_consumes_typed_events() -> None:
    handle, _runtime, stores = _handle()

    event_task = asyncio.ensure_future(anext(handle.events()))
    await asyncio.sleep(0)
    await stores.publisher.publish(
        "thread:thread-id",
        {
            "type": "text_delta",
            "thread_id": "thread-id",
            "data": {"delta": "Hello"},
        },
    )

    event = await asyncio.wait_for(event_task, timeout=1)
    assert event.type == "text_delta"
    assert event.text == "Hello"


def test_runtime_thread_handle_accepts_uuid() -> None:
    thread_id = uuid4()
    runtime = AgentRuntime(stores=InMemoryRuntimeStores(), agents={})

    handle = runtime.thread("assistant", thread_id)

    assert handle.thread_id == str(thread_id)
