"""Memory context helpers."""

from __future__ import annotations

from dataclasses import dataclass

from actant.memory.cards import MemoryCard
from actant.memory.manifest import MemoryManifest
from actant.memory.store import MemoryStore


@dataclass(frozen=True)
class MemoryContext:
    manifest: MemoryManifest
    selected_cards: tuple[MemoryCard, ...] = ()

    def to_prompt_text(self) -> str:
        sections = [self.manifest.to_prompt_text()]
        if self.selected_cards:
            sections.append("Selected memory cards:")
            for card in self.selected_cards:
                sections.append(f"## {card.id}: {card.title}\n{card.body}")
        return "\n\n".join(sections)


async def build_memory_context(
    store: MemoryStore,
    namespace: str,
    *,
    selected_card_ids: tuple[str, ...] = (),
) -> MemoryContext:
    refs = tuple(await store.list(namespace))
    selected: list[MemoryCard] = []
    for card_id in selected_card_ids:
        card = await store.get(namespace, card_id)
        if card is not None:
            selected.append(card)
    return MemoryContext(
        manifest=MemoryManifest(namespace=namespace, cards=refs),
        selected_cards=tuple(selected),
    )
