"""Retryable application integration at a persisted run boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RunCompletion:
    """Facts available after a run and its thread projection are finalized."""

    agent_id: str
    thread_id: str
    run_id: str
    outcome: str

    @property
    def succeeded(self) -> bool:
        return self.outcome == "completed"


class RunCompletionHandler(Protocol):
    """Durable integration invoked inside the retryable finalization activity.

    Unlike lifecycle hooks, failure is significant: an exception keeps the
    Temporal activity incomplete and causes it to retry. Implementations must
    therefore be idempotent.
    """

    async def __call__(self, completion: RunCompletion) -> None: ...
