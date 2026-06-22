"""Memory card models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from actant.core import JSONObject


@dataclass(frozen=True)
class MemoryCardRef:
    id: str
    namespace: str
    title: str
    tags: tuple[str, ...] = ()
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> JSONObject:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "title": self.title,
            "tags": list(self.tags),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class MemoryCard:
    id: str
    namespace: str
    title: str
    body: str
    tags: tuple[str, ...] = ()
    metadata: JSONObject = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    version: int = 1

    def ref(self) -> MemoryCardRef:
        return MemoryCardRef(
            id=self.id,
            namespace=self.namespace,
            title=self.title,
            tags=self.tags,
            updated_at=self.updated_at,
        )

    def to_dict(self, *, include_body: bool = True) -> JSONObject:
        data: JSONObject = {
            "id": self.id,
            "namespace": self.namespace,
            "title": self.title,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }
        if include_body:
            data["body"] = self.body
        return data


@dataclass(frozen=True)
class MemorySearchResult:
    card: MemoryCardRef
    score: float
    snippet: str

    def to_dict(self) -> JSONObject:
        return {
            "card": self.card.to_dict(),
            "score": self.score,
            "snippet": self.snippet,
        }
