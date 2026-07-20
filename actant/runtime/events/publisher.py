"""Transport-neutral publisher protocol for runtime events."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from actant.core import JSONObject


class EventPublisher(Protocol):
    """Pub/sub fan-out for runtime hooks.

    The Temporal runtime emits hook events from inside activities (turn
    deltas, tool results, completion). Apps wire their own publisher
    (Redis pubsub, SSE bus, websockets) to receive them.
    """

    async def publish(self, channel: str, event: JSONObject) -> None: ...

    def subscribe(self, channel: str) -> AsyncIterator[JSONObject]: ...
