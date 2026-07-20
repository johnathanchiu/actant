"""Low-latency model-stream observers and event publication."""

from __future__ import annotations

from actant.core import JSONObject
from actant.runtime.events.publisher import EventPublisher


class StreamListener:
    """Per-call sink for token-level deltas from an LLM provider.

    Stream events precede the canonical assistant-message write and may be
    lost or duplicated. Implementations should be lightweight, non-blocking,
    and safe to abandon when a client disconnects.
    """

    async def on_text_delta(self, delta: str) -> None:
        pass

    async def on_thinking_delta(self, delta: str) -> None:
        pass

    async def on_tool_call_start(self, tool_call_id: str, name: str) -> None:
        """Report the opening of a streamed tool-use content block."""
        pass

    async def on_tool_call_args_delta(self, tool_call_id: str, delta: str) -> None:
        """Report one partial JSON argument fragment."""
        pass

    async def on_tool_call_args_complete(self, tool_call_id: str) -> None:
        """Report that a streamed tool-use content block has closed."""
        pass

    def cancel_requested(self) -> bool:
        return False


class PublishingStreamListener(StreamListener):
    """Publish model deltas to the thread and, optionally, its parent."""

    def __init__(
        self,
        thread_id: str,
        publisher: EventPublisher,
        *,
        channel: str | None = None,
        parent_channel: str | None = None,
        parent_metadata: JSONObject | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.publisher = publisher
        self.channel = channel or f"thread:{thread_id}"
        self.parent_channel = parent_channel
        self.parent_metadata = parent_metadata

    async def _emit(self, event_type: str, data: JSONObject) -> None:
        await self.publisher.publish(
            self.channel,
            {"type": event_type, "thread_id": self.thread_id, "data": data},
        )
        if self.parent_channel is not None and self.parent_metadata is not None:
            payload: JSONObject = {
                "type": event_type,
                "thread_id": self.thread_id,
                "data": data,
            }
            payload.update(self.parent_metadata)
            await self.publisher.publish(self.parent_channel, payload)

    async def on_text_delta(self, delta: str) -> None:
        await self._emit("text_delta", {"delta": delta})

    async def on_thinking_delta(self, delta: str) -> None:
        await self._emit("thinking_delta", {"delta": delta})

    async def on_tool_call_start(self, tool_call_id: str, name: str) -> None:
        await self._emit("tool_call_start", {"tool_call_id": tool_call_id, "name": name})

    async def on_tool_call_args_delta(self, tool_call_id: str, delta: str) -> None:
        await self._emit("tool_call_args_delta", {"tool_call_id": tool_call_id, "delta": delta})

    async def on_tool_call_args_complete(self, tool_call_id: str) -> None:
        await self._emit("tool_call_args_complete", {"tool_call_id": tool_call_id})
