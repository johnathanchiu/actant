"""Transport-neutral runtime event protocols."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from actant.core import JSONObject


class EventSink(Protocol):
    """Worker-side destination for live runtime events."""

    async def publish(self, channel: str, event: JSONObject) -> None: ...


class EventSource(Protocol):
    """Application-side source of live runtime events."""

    def subscribe(self, channel: str) -> AsyncIterator[JSONObject]: ...


class EventPublisher(EventSink, EventSource, Protocol):
    """Combined event sink/source retained for in-process brokers.

    The Temporal runtime emits hook events from inside activities (turn
    deltas, tool results, completion). Apps wire their own publisher
    (Redis pubsub, SSE bus, websockets) to receive them.
    """
