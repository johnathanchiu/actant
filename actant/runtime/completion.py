"""Retryable application integration at a persisted run boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class RunCompletion:
    """Facts available after a run and its thread projection are finalized.

    ``artifacts`` are the deliverables terminal tool results named, as the
    product's artifact sink stored them (``ArtifactRef.to_dict()`` shape).
    ``reason`` is set when the outcome alone does not explain the end.
    """

    agent_id: str
    thread_id: str
    run_id: str
    outcome: str
    reason: str | None = None
    artifacts: tuple[dict[str, object], ...] = field(default_factory=tuple)

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
