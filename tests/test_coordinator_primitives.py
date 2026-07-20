"""Tests for the coordinator primitives.

Covers:
- SubThreadRegistry round-trip + lookup
- publishing_hooks_factory: top-level threads vs sub-threads
- publishing_listener_factory: same
- resolve_deferred_tool_call: thin wrapper passes args through; surfaces
  ToolResolutionStaleError when runtime reconciles a stale activity
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from actant.runtime.coordinator import (
    SubThreadLink,
    SubThreadRegistry,
    publishing_hooks_factory,
    publishing_listener_factory,
    resolve_deferred_tool_call,
)
from actant.runtime.exceptions import ToolResolutionStaleError
from actant.runtime.hooks import PublishingThreadHooks
from actant.runtime.stores.in_memory import InMemoryEventPublisher
from actant.runtime.types.threads import AgentThread


def _thread(thread_id: str, agent_id: str = "demo") -> AgentThread:
    return AgentThread(id=thread_id, agent_id=agent_id)


# ─── SubThreadRegistry ──────────────────────────────────────────────


def test_registry_register_get_pop_roundtrip() -> None:
    reg = SubThreadRegistry()
    link = SubThreadLink(
        sub_thread_id="sub_1",
        parent_thread_id="thread_a",
        parent_tool_call_id="tc_1",
        sub_agent_id="researcher",
        subagent_name="researcher",
    )
    assert reg.get("sub_1") is None
    reg.register(link)
    assert reg.get("sub_1") is link
    assert "sub_1" in reg
    assert len(reg) == 1
    popped = reg.pop("sub_1")
    assert popped is link
    assert reg.get("sub_1") is None
    assert len(reg) == 0


def test_registry_find_by_parent_tool_call() -> None:
    reg = SubThreadRegistry()
    link_a = SubThreadLink("sub_a", "thread_x", "tc_alpha", "researcher")
    link_b = SubThreadLink("sub_b", "thread_x", "tc_beta", "researcher")
    reg.register(link_a)
    reg.register(link_b)
    assert reg.find_by_parent_tool_call("tc_alpha") is link_a
    assert reg.find_by_parent_tool_call("tc_beta") is link_b
    assert reg.find_by_parent_tool_call("tc_unknown") is None


def test_registry_register_is_idempotent() -> None:
    reg = SubThreadRegistry()
    link_v1 = SubThreadLink("sub_1", "thread_a", "tc_1", "researcher")
    link_v2 = SubThreadLink(
        "sub_1", "thread_a", "tc_1", "researcher", subagent_name="researcher"
    )
    reg.register(link_v1)
    reg.register(link_v2)  # overwrite
    assert reg.get("sub_1") is link_v2
    assert len(reg) == 1


# ─── publishing_hooks_factory ───────────────────────────────────────


async def test_hooks_factory_top_level_publishes_once() -> None:
    """Top-level thread → events go to thread:<id> only, no parent dual-publish."""
    publisher = InMemoryEventPublisher()
    factory = publishing_hooks_factory(publisher, registry=None)
    hooks = factory(_thread("thread_top"))
    assert isinstance(hooks, PublishingThreadHooks)
    await hooks.emit("custom", {"x": 1})
    own = publisher.events.get("thread:thread_top", [])
    assert len(own) == 1
    assert own[0]["type"] == "custom"
    # No other channel got it.
    assert list(publisher.events.keys()) == ["thread:thread_top"]


async def test_hooks_factory_subthread_dual_publishes() -> None:
    """Sub-thread → events go to BOTH thread:<sub> AND thread:<parent>
    with parent metadata stamped."""
    publisher = InMemoryEventPublisher()
    registry = SubThreadRegistry()
    registry.register(
        SubThreadLink(
            sub_thread_id="sub_1",
            parent_thread_id="thread_parent",
            parent_tool_call_id="tc_task_1",
            sub_agent_id="researcher",
            subagent_name="researcher",
        )
    )
    factory = publishing_hooks_factory(publisher, registry=registry)
    hooks = factory(_thread("sub_1", agent_id="researcher"))
    assert isinstance(hooks, PublishingThreadHooks)
    await hooks.emit("custom", {"x": 1})

    # Own channel: verbatim event.
    own = publisher.events["thread:sub_1"]
    assert len(own) == 1
    assert own[0]["type"] == "custom"
    assert own[0]["thread_id"] == "sub_1"
    assert "parent_thread_id" not in own[0]

    # Parent channel: stamped with parent metadata.
    parent = publisher.events["thread:thread_parent"]
    assert len(parent) == 1
    assert parent[0]["type"] == "custom"
    assert parent[0]["thread_id"] == "sub_1"
    assert parent[0]["parent_thread_id"] == "thread_parent"
    assert parent[0]["parent_tool_call_id"] == "tc_task_1"
    assert parent[0]["subagent"] == "researcher"


async def test_hooks_factory_no_registry_means_no_dual_publish() -> None:
    """registry=None → behaves exactly like un-wired PublishingThreadHooks."""
    publisher = InMemoryEventPublisher()
    factory = publishing_hooks_factory(publisher)  # default registry=None
    hooks = factory(_thread("any"))
    assert isinstance(hooks, PublishingThreadHooks)
    await hooks.emit("custom", {})
    assert list(publisher.events.keys()) == ["thread:any"]


# ─── publishing_listener_factory ────────────────────────────────────


async def test_listener_factory_top_level() -> None:
    publisher = InMemoryEventPublisher()
    factory = publishing_listener_factory(publisher)
    listener = factory(_thread("thread_solo"))
    await listener.on_text_delta("hello")
    own = publisher.events.get("thread:thread_solo", [])
    assert len(own) == 1
    assert own[0]["type"] == "text_delta"


async def test_listener_factory_subthread_dual_publishes() -> None:
    publisher = InMemoryEventPublisher()
    registry = SubThreadRegistry()
    registry.register(
        SubThreadLink(
            sub_thread_id="sub_x",
            parent_thread_id="thread_y",
            parent_tool_call_id="tc_xy",
            sub_agent_id="researcher",
            subagent_name="researcher",
        )
    )
    factory = publishing_listener_factory(publisher, registry=registry)
    listener = factory(_thread("sub_x", agent_id="researcher"))
    await listener.on_text_delta("hi")
    own = publisher.events["thread:sub_x"]
    parent = publisher.events["thread:thread_y"]
    assert len(own) == 1
    assert len(parent) == 1
    assert parent[0]["parent_thread_id"] == "thread_y"
    assert parent[0]["subagent"] == "researcher"


# ─── resolve_deferred_tool_call ─────────────────────────────────────


async def test_resolve_deferred_tool_call_delegates_to_runtime() -> None:
    """Thin wrapper passes all args to runtime deferred resolution."""
    runtime = AsyncMock()
    await resolve_deferred_tool_call(
        runtime,
        agent_id="demo",
        thread_id="thread_1",
        tool_call_id="tc_1",
        approved=True,
        answer="ok",
        payload={"extra": 1},
    )
    runtime.resolve_deferred_tool_call.assert_awaited_once_with(
        "demo",
        "thread_1",
        "tc_1",
        approved=True,
        answer="ok",
        payload={"extra": 1},
    )


async def test_resolve_deferred_tool_call_propagates_stale_error() -> None:
    """When the runtime raises ToolResolutionStaleError (Temporal lost
    the activity), the wrapper propagates verbatim — apps catch it
    above this level."""
    runtime = AsyncMock()
    runtime.resolve_deferred_tool_call.side_effect = ToolResolutionStaleError("tc_x", "gone")
    with pytest.raises(ToolResolutionStaleError):
        await resolve_deferred_tool_call(
            runtime,
            agent_id="demo",
            thread_id="thread_1",
            tool_call_id="tc_x",
            approved=True,
            answer="ok",
        )
