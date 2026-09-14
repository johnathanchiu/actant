"""Host admission at the start of every model turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TurnStart:
    """The turn about to call the model."""

    agent_id: str
    thread_id: str
    run_id: str
    turn_id: str
    turn_index: int


class TurnGate(Protocol):
    """Worker-side check consulted before each model call.

    Return ``None`` to let the turn call the model. Return a reason to end the
    run instead: the model is not called, the run finalizes ``exhausted`` with
    that reason as its ``stop_reason``, and ``RunCompletion`` and
    ``on_complete`` receive it. Inbound messages are persisted first, so a
    stopped run loses nothing the user sent.

    Unlike ``on_turn_start``, the gate decides. An exception fails the turn
    and with it the run.
    """

    async def __call__(self, turn: TurnStart) -> str | None: ...
