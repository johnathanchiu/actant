"""Memory store protocols."""

from __future__ import annotations

from typing import Protocol

from actant.memory.cards import MemoryCard, MemoryCardRef, MemorySearchResult


class MemoryStore(Protocol):
    async def put(self, card: MemoryCard) -> MemoryCard: ...

    async def get(self, namespace: str, card_id: str) -> MemoryCard | None: ...

    async def delete(self, namespace: str, card_id: str) -> bool: ...

    async def list(self, namespace: str) -> list[MemoryCardRef]: ...

    async def search(
        self,
        namespace: str,
        query: str,
        *,
        limit: int = 10,
    ) -> list[MemorySearchResult]: ...

    async def append(self, namespace: str, card_id: str, body: str) -> MemoryCard | None: ...

    async def replace(self, namespace: str, card_id: str, body: str) -> MemoryCard | None: ...
