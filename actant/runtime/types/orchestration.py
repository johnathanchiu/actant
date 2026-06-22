"""Runtime orchestration data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StepStatus(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    NOOP = "noop"
    PARKED = "parked"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class StepResult:
    status: StepStatus
    thread_id: str | None = None
    turns_executed: int = 0
    # ``True`` when the orchestrator wants the caller (``run_one``) to
    # release the wake back to the queue instead of acking it.
    # Currently set only when claim_thread refused due to a TRANSIENT
    # non-claimable state (another driver holds the thread); the
    # released wake will be retried after the holder releases.
    # Terminal-state refusals (CANCELLED / FAILED) and stale-wake
    # NOOPs leave this False — the wake should be consumed.
    requeue: bool = False
