"""Coordinator primitives for multi-thread + sub-agent apps.

This module ships the smallest set of pieces a non-trivial actant app
needs to wire together. It is INTENTIONALLY a set of building blocks,
not a complete coordinator class — apps compose these into their own
thin coordinator and own all the policy decisions (per-thread agent
construction, harvest semantics, cancellation behavior, etc).

If you're building a single-agent chatbot with no sub-agents, you
don't need any of this — use :class:`AgentRuntime` directly.

If you're building anything more complex (multiple agents,
``task()``-style delegation, robust state recovery), the canonical
pattern is:

1. Create a :class:`SubThreadRegistry` that lives for the process.
2. Use :func:`publishing_hooks_factory` + :func:`publishing_listener_factory`
   to get factories that auto-route sub-thread events to the parent's
   SSE channel.
3. Implement your own ``Coordinator`` class that:
   - Owns the registry.
   - Builds :class:`AgentDefinition`\\s per thread (your policy).
   - Implements a ``spawn_subagent`` method that registers a new
     :class:`SubThreadLink` BEFORE calling ``runtime.send_message``
     (so the hook factory sees the relationship synchronously).
   - Routes both user-driven and sub-thread-completion resolutions through
     ``runtime.resolve_tool_call``.

See ``docs/coordinator-guide.md`` for the full pattern with code,
and the ``examples/demo/`` directory in the actant repo for the canonical
reference implementation (``DemoCoordinator``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from actant.core import JSONObject
from actant.runtime.events import (
    AgentThreadHooks,
    PublishingStreamListener,
    PublishingThreadHooks,
    StreamListener,
)
from actant.runtime.events import EventPublisher
from actant.runtime.types.threads import AgentThread

__all__ = [
    "HookFactory",
    "ListenerFactory",
    "SubThreadLink",
    "SubThreadRegistry",
    "publishing_hooks_factory",
    "publishing_listener_factory",
]


HookFactory = Callable[[AgentThread], AgentThreadHooks]
ListenerFactory = Callable[[AgentThread], StreamListener]


@dataclass(frozen=True)
class SubThreadLink:
    """Describes a single parent ↔ sub-thread relationship.

    Apps register a link BEFORE sending the sub-thread's first
    message; the hook factories (built via this module's helpers)
    consult the registry at hook-construction time to enable
    dual-publishing.

    The link is immutable. App-specific bookkeeping that needs to
    mutate (e.g. "the latest assistant text the sub-thread emitted")
    belongs in app state, not on the link.
    """

    sub_thread_id: str
    parent_thread_id: str
    parent_tool_call_id: str
    sub_agent_id: str
    subagent_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SubThreadRegistry:
    """In-memory map of active sub-threads keyed by ``sub_thread_id``.

    Owned by the app's coordinator. Lifetime is process-scoped; applications
    rebuild it from persisted thread parent fields on startup. Durable parent
    resolution must use a ``RunCompletionHandler`` and projection data rather
    than relying on registry presence.

    Thread-safety: registrations and lookups happen from coroutines
    on a single event loop. The registry is NOT safe for use across
    OS threads.
    """

    def __init__(self) -> None:
        self._links: dict[str, SubThreadLink] = {}

    def register(self, link: SubThreadLink) -> None:
        """Add a link. Idempotent — re-registering with the same
        ``sub_thread_id`` overwrites the previous entry."""
        self._links[link.sub_thread_id] = link

    def get(self, sub_thread_id: str) -> SubThreadLink | None:
        return self._links.get(sub_thread_id)

    def pop(self, sub_thread_id: str) -> SubThreadLink | None:
        """Remove and return the link. Used at sub-thread completion
        time so the registry only holds active sub-threads."""
        return self._links.pop(sub_thread_id, None)

    def find_by_parent_tool_call(self, parent_tool_call_id: str) -> SubThreadLink | None:
        """Look up a sub-thread by the tool call that spawned it.
        Useful when the parent's deferred resolution flow needs to
        find the sub-thread it's waiting on."""
        for link in self._links.values():
            if link.parent_tool_call_id == parent_tool_call_id:
                return link
        return None

    def __contains__(self, sub_thread_id: str) -> bool:
        return sub_thread_id in self._links

    def __len__(self) -> int:
        return len(self._links)


def _parent_metadata_for(link: SubThreadLink) -> JSONObject:
    """Stamps a sub-thread event with parent context so the FE can
    attribute it to the parent's ``task()`` row."""
    metadata: JSONObject = {
        "parent_thread_id": link.parent_thread_id,
        "parent_tool_call_id": link.parent_tool_call_id,
    }
    if link.subagent_name is not None:
        metadata["subagent"] = link.subagent_name
    return metadata


def publishing_hooks_factory(
    publisher: EventPublisher,
    registry: SubThreadRegistry | None = None,
) -> HookFactory:
    """Return a HookFactory that yields :class:`PublishingThreadHooks`
    for every thread, with dual-publish to the parent's channel
    automatically enabled for sub-threads in ``registry``.

    Apps wire this into ``AgentRuntime``'s ``hooks_factory`` parameter
    instead of constructing hooks manually. The factory is the
    single point of "should this thread dual-publish?" decision-
    making — every thread's hooks consult the registry at
    construction time.
    """

    def factory(thread: AgentThread) -> AgentThreadHooks:
        link = registry.get(thread.id) if registry is not None else None
        if link is None:
            return PublishingThreadHooks(thread.id, publisher=publisher)
        return PublishingThreadHooks(
            thread.id,
            publisher=publisher,
            parent_channel=f"thread:{link.parent_thread_id}",
            parent_metadata=_parent_metadata_for(link),
        )

    return factory


def publishing_listener_factory(
    publisher: EventPublisher,
    registry: SubThreadRegistry | None = None,
) -> ListenerFactory:
    """Stream-listener counterpart of :func:`publishing_hooks_factory`.

    Use both together when wiring ``AgentRuntime`` so streaming
    deltas (text_delta, thinking_delta, tool_call_args_delta) ALSO
    dual-publish to the parent for sub-threads. Without this, parents
    would only see assistant_message/tool_result events from sub-threads
    — they'd miss the streaming.
    """

    def factory(thread: AgentThread) -> StreamListener:
        link = registry.get(thread.id) if registry is not None else None
        if link is None:
            return PublishingStreamListener(thread.id, publisher=publisher)
        return PublishingStreamListener(
            thread.id,
            publisher=publisher,
            parent_channel=f"thread:{link.parent_thread_id}",
            parent_metadata=_parent_metadata_for(link),
        )

    return factory
