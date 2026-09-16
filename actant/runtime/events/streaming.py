"""Low-latency model-stream observers and event publication."""

from __future__ import annotations

from actant.core import JSONObject


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

    async def on_usage(self, response_id: str, model: str, usage: JSONObject, status: str) -> None:
        """Report provider usage once available; consumers deduplicate by response_id."""
        pass

    async def on_stream_reset(self) -> None:
        """Discard every delta since the call began: a failed attempt is being retried
        and its partial text, thinking, and tool calls will not be committed."""
        pass

    def cancel_requested(self) -> bool:
        return False
