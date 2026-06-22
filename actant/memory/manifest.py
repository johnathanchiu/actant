"""Prompt-cache friendly memory manifests."""

from __future__ import annotations

from dataclasses import dataclass

from actant.core import JSONObject
from actant.memory.cards import MemoryCardRef


@dataclass(frozen=True)
class MemoryManifest:
    namespace: str
    cards: tuple[MemoryCardRef, ...]

    def to_dict(self) -> JSONObject:
        return {
            "namespace": self.namespace,
            "cards": [card.to_dict() for card in self.cards],
        }

    def to_prompt_text(self) -> str:
        lines = [f"Memory namespace: {self.namespace}", "Available memory cards:"]
        if not self.cards:
            lines.append("- none")
            return "\n".join(lines)

        for card in self.cards:
            tags = f" tags={','.join(card.tags)}" if card.tags else ""
            lines.append(f"- {card.id}: {card.title}{tags}")
        return "\n".join(lines)
