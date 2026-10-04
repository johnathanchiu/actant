"""Fake LLM for tests and examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall

if TYPE_CHECKING:
    from actant.runtime.events.streaming import StreamListener


@dataclass
class FakeResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text_chunks: list[str] | None = None
    thinking_chunks: list[str] | None = None
    #: Usage to report, as a provider would on the assistant message.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None


class FakeLLM:
    supports_allowed_tools = True

    def __init__(
        self,
        responses: list[FakeResponse],
        model_id: str = "fake",
        *,
        context_window_tokens: int | None = None,
        max_images_per_request: int | None = None,
    ) -> None:
        self._responses = list(responses)
        self.model_id = model_id
        self.context_window_tokens = context_window_tokens
        self.max_images_per_request = max_images_per_request
        self.calls: list[tuple[str, list[Message], list[dict]]] = []
        self.allowed_tools: list[tuple[str, ...]] = []
        self.max_output_tokens: list[int | None] = []

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None" = None,
        *,
        allowed_tools: tuple[str, ...] = (),
        max_output_tokens: int | None = None,
    ) -> Message:
        self.calls.append((system, list(messages), tools))
        self.allowed_tools.append(allowed_tools)
        self.max_output_tokens.append(max_output_tokens)
        if not self._responses:
            raise RuntimeError("FakeLLM has no queued responses")
        response = self._responses.pop(0)
        if allowed_tools and (
            not response.tool_calls
            or any(call.function.name not in allowed_tools for call in response.tool_calls)
        ):
            raise ValueError("FakeResponse must call only allowed_tools on a constrained turn")

        if listener is not None:
            for chunk in response.thinking_chunks or []:
                if listener.cancel_requested():
                    raise StreamCancelled
                await listener.on_thinking_delta(chunk)
            for chunk in response.text_chunks or []:
                if listener.cancel_requested():
                    raise StreamCancelled
                await listener.on_text_delta(chunk)

        return Message(
            role="assistant",
            content=response.text or None,
            tool_calls=response.tool_calls or None,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cache_read_tokens=response.cache_read_tokens,
            cache_write_tokens=response.cache_write_tokens,
        )
