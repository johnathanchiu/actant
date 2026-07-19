"""Agent thread lifecycle models.

Temporal owns single-writer execution, so projection rows only need
enough state to answer "is this thread alive, and how did the last run
end."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class ThreadStatus(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    """Run lifecycle.

    ``ACTIVE`` while a turn is in flight, ``IDLE`` after a normal
    completion, terminal otherwise.
    """

    ACTIVE = "active"
    IDLE = "idle"
    EXHAUSTED = "exhausted"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class AgentThread:
    id: str
    agent_id: str
    status: ThreadStatus = ThreadStatus.IDLE
    turn_count: int = 0
    active_run_id: str | None = None
    parent_thread_id: str | None = None
    parent_turn_id: str | None = None
    parent_tool_call_id: str | None = None


@dataclass
class AgentRun:
    id: str
    agent_id: str
    thread_id: str
    status: RunStatus = RunStatus.ACTIVE
    turn_count: int = 0
    max_turns: int = 25

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self.turn_count)


@dataclass(frozen=True)
class MessageRecord:
    id: str
    agent_id: str
    thread_id: str
    message: object
