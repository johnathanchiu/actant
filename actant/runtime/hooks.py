"""Thread hook interfaces."""

from __future__ import annotations

from typing import cast

from actant.core import JSONObject, JSONValue
from actant.llm.messages import Message
from actant.runtime.interfaces.events import EventPublisher
from actant.tools.base import ToolResult


class StreamListener:
    """Per-call sink for token-level deltas from an LLM provider."""

    async def on_text_delta(self, delta: str) -> None:
        pass

    async def on_thinking_delta(self, delta: str) -> None:
        pass

    async def on_tool_call_start(self, tool_call_id: str, name: str) -> None:
        """Fired when a tool_use content block opens during streaming.

        ``tool_call_id`` matches the eventual ``on_tool_call`` arg, so
        downstream sinks can correlate the streamed args with the
        assembled call that lands at turn end.
        """
        pass

    async def on_tool_call_args_delta(self, tool_call_id: str, delta: str) -> None:
        """Fired for each ``input_json_delta`` chunk inside an open
        tool_use block. ``delta`` is a partial JSON string fragment
        (concatenation of all deltas yields the final ``args`` JSON)."""
        pass

    async def on_tool_call_args_complete(self, tool_call_id: str) -> None:
        """Fired when a tool_use content block closes during streaming."""
        pass

    def cancel_requested(self) -> bool:
        return False


class PublishingStreamListener(StreamListener):
    """Stream listener that re-emits deltas onto an EventPublisher channel.

    Mirrors :class:`PublishingThreadHooks`' sub-thread dual-publish
    behavior: when ``parent_channel`` + ``parent_metadata`` are set,
    every delta lands on both the thread's own channel AND the parent
    channel with metadata stamped. See ``actant.runtime.coordinator``
    for the canonical wiring.
    """

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
        await self._emit(
            "tool_call_args_delta", {"tool_call_id": tool_call_id, "delta": delta}
        )

    async def on_tool_call_args_complete(self, tool_call_id: str) -> None:
        await self._emit("tool_call_args_complete", {"tool_call_id": tool_call_id})


class AgentThreadHooks:
    """Async callbacks fired by turn execution and coordinator lifecycle events."""

    async def on_user_message(self, content: str | list[dict[str, object]]) -> None:
        """Fired when a user turn is being persisted.

        ``content`` is either a plain string (text-only turn) or a list
        of content blocks (multimodal turn — text + asset blocks).
        Hooks that only care about text can ``isinstance``-check.
        """
        pass

    async def on_assistant_message(self, message: Message) -> None:
        pass

    async def on_turn_start(self, turn: int, turn_id: str | None = None) -> None:
        del turn_id
        pass

    async def on_tool_call(self, tool_call_id: str, name: str, args: JSONObject) -> None:
        pass

    async def on_tool_result(
        self, tool_call_id: str, result: ToolResult, turn_id: str | None = None
    ) -> None:
        del turn_id
        pass

    async def on_tool_waiting(
        self,
        tool_call_id: str,
        prompt: str,
        turn_id: str | None = None,
        wait_request: JSONObject | None = None,
    ) -> None:
        del turn_id
        del wait_request
        pass

    async def on_tool_resolved(
        self, tool_call_id: str, result: ToolResult, turn_id: str | None = None
    ) -> None:
        del turn_id
        pass

    async def on_complete(self, success: bool, reason: str, message: str) -> None:
        pass

    async def on_error(self, error: Exception) -> None:
        pass


class PublishingThreadHooks(AgentThreadHooks):
    """Re-emit runtime lifecycle events onto an EventPublisher channel.

    Conversation state is persisted by the runtime through whichever
    ``MessageStore`` impl is plugged in — these hooks are pure
    observability / SSE-bus broadcasting and do not write themselves.
    Routing both the runtime AND the hooks at the message store would
    double-write the same row from two callers; the runtime is the
    single thread-safe writer, hooks just announce.

    **Sub-thread dual publishing** (since v0.2): if ``parent_channel``
    + ``parent_metadata`` are set, every event is published to TWO
    channels: the thread's own channel (verbatim) AND the parent
    channel (with ``parent_thread_id`` / ``parent_tool_call_id`` /
    ``subagent`` stamped into the envelope). Apps building on actant
    use this to surface sub-agent activity in the parent thread's
    SSE stream — see :class:`SubThreadRegistry` and
    :func:`publishing_hooks_factory` in
    ``actant.runtime.coordinator`` for the canonical wiring.
    """

    def __init__(
        self,
        thread_id: str,
        publisher: EventPublisher | None = None,
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

    async def emit(self, event_type: str, data: JSONObject) -> None:
        if self.publisher is None:
            return
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

    async def on_assistant_message(self, message: Message) -> None:
        tool_calls = cast(list[JSONValue], [tc.to_dict() for tc in (message.tool_calls or [])])
        payload: JSONObject = {
            "content": cast(JSONValue, message.content),
            "thought_summary": message.thought_summary,
            "tool_calls": tool_calls,
        }
        await self.emit("assistant_message", payload)

    async def on_turn_start(self, turn: int, turn_id: str | None = None) -> None:
        payload: JSONObject = {"turn": turn}
        if turn_id is not None:
            payload["turn_id"] = turn_id
            payload["turn_uid"] = turn_id
        await self.emit("turn_start", payload)

    async def on_tool_call(self, tool_call_id: str, name: str, args: JSONObject) -> None:
        await self.emit("tool_call", {"tool_call_id": tool_call_id, "name": name, "args": args})

    async def on_tool_result(
        self, tool_call_id: str, result: ToolResult, turn_id: str | None = None
    ) -> None:
        payload: JSONObject = {
            "tool_call_id": tool_call_id,
            "output": str(result.output) if result.output is not None else None,
            "error": result.error,
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
            payload["turn_uid"] = turn_id
        await self.emit("tool_result", payload)

    async def on_tool_waiting(
        self,
        tool_call_id: str,
        prompt: str,
        turn_id: str | None = None,
        wait_request: JSONObject | None = None,
    ) -> None:
        payload: JSONObject = {"tool_call_id": tool_call_id, "prompt": prompt}
        if wait_request is not None:
            payload["wait_request"] = wait_request
            kind = wait_request.get("kind")
            payload["wait_kind"] = kind if isinstance(kind, str) else None
            wait_payload = wait_request.get("payload")
            payload["wait_payload"] = wait_payload if isinstance(wait_payload, dict) else {}
        if turn_id is not None:
            payload["turn_id"] = turn_id
            payload["turn_uid"] = turn_id
        await self.emit("tool_waiting", payload)

    async def on_tool_resolved(
        self, tool_call_id: str, result: ToolResult, turn_id: str | None = None
    ) -> None:
        payload: JSONObject = {
            "tool_call_id": tool_call_id,
            "output": str(result.output) if result.output is not None else None,
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
            payload["turn_uid"] = turn_id
        await self.emit("tool_resolved", payload)

    async def on_complete(self, success: bool, reason: str, message: str) -> None:
        await self.emit(
            "complete",
            {"success": success, "reason": reason, "message": message},
        )

    async def on_error(self, error: Exception) -> None:
        await self.emit("error", {"message": str(error)})
