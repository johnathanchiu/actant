"""LLM client protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from actant.llm.messages import Message

if TYPE_CHECKING:
    from actant.runtime.events.streaming import StreamListener


class LLMClient(Protocol):
    model_id: str

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None" = None,
    ) -> Message: ...
