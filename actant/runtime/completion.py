"""Retryable application integration at a persisted run boundary."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from actant.blocks import TextBlock
from actant.core import JSONObject
from actant.llm.messages import Message


@dataclass(frozen=True)
class LastCall:
    """The last tool call a run made, its arguments as the model wrote them."""

    name: str
    arguments: JSONObject


@dataclass(frozen=True)
class RunCompletion:
    """Facts available after a run and its thread projection are finalized.

    ``artifacts`` are the deliverables terminal tool results named, as the
    product's artifact sink stored them (``ArtifactRef.to_dict()`` shape).
    ``stop_reason`` is set when the outcome alone does not explain the end.
    ``reply`` and ``last_call`` are the run's last word since the thread's last
    inbound message (:func:`last_word`); ``tags`` are the tags of every inbound
    message on the thread, in order.
    """

    agent_id: str
    thread_id: str
    run_id: str
    outcome: str
    stop_reason: str | None = None
    artifacts: tuple[dict[str, object], ...] = field(default_factory=tuple)
    reply: str = ""
    last_call: LastCall | None = None
    tags: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.outcome == "completed"


def last_word(messages: Sequence[Message]) -> tuple[str, LastCall | None]:
    """What the agent last said after the last inbound message: its last text, and its
    last tool call whose arguments parse (a garbled call carries nothing)."""
    start = max(
        (i for i, m in enumerate(messages) if m.role == "user" and m.kind == "message"),
        default=-1,
    )
    reply, call = "", None
    for message in reversed(messages[start + 1 :]):
        if message.role != "assistant":
            continue
        call = call or _last_call(message)
        if not reply:
            reply = _text(message)
        if reply and call is not None:
            break
    return reply, call


def inbound_tags(messages: Sequence[Message]) -> tuple[str, ...]:
    """The tags of the inbound messages in ``messages``, in order."""
    return tuple(m.tag for m in messages if m.role == "user" and m.kind == "message" and m.tag)


def _last_call(message: Message) -> LastCall | None:
    for tool_call in reversed(message.tool_calls or []):
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict):
            return LastCall(tool_call.function.name, arguments)
    return None


def _text(message: Message) -> str:
    if isinstance(message.content, str):
        return message.content if message.content.strip() else ""
    text = "".join(b.text for b in message.content or [] if isinstance(b, TextBlock))
    return text if text.strip() else ""


class RunCompletionHandler(Protocol):
    """Durable integration invoked inside the retryable finalization activity.

    Unlike lifecycle hooks, failure is significant: an exception keeps the
    Temporal activity incomplete and causes it to retry. Implementations must
    therefore be idempotent.
    """

    async def __call__(self, completion: RunCompletion) -> None: ...
