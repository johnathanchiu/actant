"""Transport-neutral runtime event protocols."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol
import logging

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


log = logging.getLogger(__name__)


class ScopedEventSink:
    def __init__(
        self,
        sink: EventSink | None,
        agent_id: str,
        run_id: str | None,
        turn_id: str | None,
        turn_index: int | None = None,
    ) -> None:
        self.sink = sink
        self.identity: JSONObject = {"agent_id": agent_id}
        if run_id is not None:
            self.identity["run_id"] = run_id
        if turn_id is not None:
            self.identity.update({"turn_id": turn_id, "turn_uid": turn_id})

        if turn_index is not None:
            self.identity["turn_index"] = turn_index

    async def publish(self, channel: str, event: JSONObject) -> None:
        if self.sink is None:
            return
        data = event.get("data")
        payload = {**event, "data": {**self.identity, **(data if isinstance(data, dict) else {})}}
        try:
            await self.sink.publish(channel, payload)
        except Exception:
            log.exception("runtime event publication failed")
