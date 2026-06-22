"""Memory store + memory context primitives.

End-to-end memory-tool usage inside an agent loop is covered by the
workflow tests; here we exercise the store and prompt-context builder
in isolation.
"""

from __future__ import annotations

import pytest

from actant.memory import MemoryCard, build_memory_context
from actant.runtime.stores import InMemoryMemoryStore


@pytest.mark.asyncio
async def test_memory_store_lists_reads_searches_and_appends() -> None:
    store = InMemoryMemoryStore()
    await store.put(
        MemoryCard(
            id="nvda_watch",
            namespace="agent_1",
            title="NVDA Watch",
            body="Track valuation risk and datacenter demand.",
            tags=("watchlist", "semis"),
        )
    )

    refs = await store.list("agent_1")
    assert [ref.id for ref in refs] == ["nvda_watch"]

    card = await store.get("agent_1", "nvda_watch")
    assert card is not None
    assert card.title == "NVDA Watch"

    results = await store.search("agent_1", "valuation")
    assert len(results) == 1
    assert results[0].card.id == "nvda_watch"

    updated = await store.append("agent_1", "nvda_watch", "New note.")
    assert updated is not None
    assert updated.version == 2
    assert updated.body.endswith("New note.")


@pytest.mark.asyncio
async def test_memory_context_builds_manifest_and_selected_cards() -> None:
    store = InMemoryMemoryStore()
    await store.put(
        MemoryCard(
            id="principles",
            namespace="agent_1",
            title="Principles",
            body="Prefer simple decisions.",
        )
    )

    context = await build_memory_context(
        store,
        "agent_1",
        selected_card_ids=("principles",),
    )

    prompt = context.to_prompt_text()
    assert "principles: Principles" in prompt
    assert "Prefer simple decisions." in prompt
