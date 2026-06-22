"""Durable inbox message models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from actant.core import JSONObject


@dataclass(frozen=True)
class InboxMessage:
    id: str
    agent_id: str
    thread_id: str
    payload: JSONObject
    source: str = "user"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    visible_at: datetime | None = None
    correlation_id: str | None = None
    environment_id: str | None = None
