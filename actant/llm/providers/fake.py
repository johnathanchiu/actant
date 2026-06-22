"""Fake LLM for tests and examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall

if TYPE_CHECKING:
    from actant.runtime.hooks import StreamListener


@dataclass
class FakeResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text_chunks: list[str] | None = None
    thinking_chunks: list[str] | None = None


class FakeLLM:
    def __init__(
        self,
        responses: list[FakeResponse],
        model_id: str = "fake",
    ) -> None:
        self._responses = list(responses)
        self.model_id = model_id
        self.calls: list[tuple[str, list[Message], list[dict]]] = []

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None" = None,
    ) -> Message:
        self.calls.append((system, list(messages), tools))
        if not self._responses:
            raise RuntimeError("FakeLLM has no queued responses")
        response = self._responses.pop(0)

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
        )
