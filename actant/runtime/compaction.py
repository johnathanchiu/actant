"""Model-context compaction: the one place Actant ever reduces what the model sees.

Nothing is truncated and no stored message is rewritten. When the next request
would cross a limit (a fraction of the context window, or the provider's image
cap per request), the workflow runs one extra turn on the same model, with no
tools, that summarizes the conversation so far. The thread then continues in a
fresh context:

    system prompt, the summary, the app's pinned content, then what was pending

Every message stays in the store. A :class:`CompactionRecord` marks the
boundary, and the model's request is built from the boundary onward.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from actant.blocks import AssetBlock, Block, InlineImageBlock, TextBlock, UrlImageBlock
from actant.llm.messages import Message

COMPACTION_PROMPT = """\
<compaction_request>
Your context is nearly full. Write a summary of this conversation that will \
replace everything before this point. You will continue from the summary alone, \
with the system prompt, so anything you leave out is gone.

Cover, in this order:
1. The goal: what the user asked for, and any constraints or preferences they stated.
2. Decisions made, and why.
3. What was verified, and how (tests run, outputs checked, results observed).
4. Open items: every piece of unfinished work, unanswered question, or pending \
tool result. Do not omit any.
5. Next steps, in order.
6. Key facts to keep exactly: identifiers, paths, names, numbers, commands, \
errors, and anything the user must not be asked again.

Write plain prose and lists. Do not call tools. Do not address the user.
</compaction_request>"""

SUMMARY_PREFIX = "<context_summary>\nThis conversation was compacted. It continues from this summary of everything before it:\n\n"
SUMMARY_SUFFIX = "\n</context_summary>"

#: Characters per token for the part of a request no provider has counted yet.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Compaction:
    """What an ``on_compact`` hook receives."""

    agent_id: str
    thread_id: str
    run_id: str
    reason: str
    summary: str


#: Returns blocks to pin after the summary, read from the app's own source of truth
#: (a checklist, the current file, images for open items). An empty list pins nothing.
#: Registered by name in ``pin_providers`` and named in ``CompactionConfig.pin``; the
#: same signature serves the ``on_compact`` hook, the escape hatch that runs last.
PinProvider = Callable[[Compaction], Awaitable[list[Block]]]
CompactionHook = PinProvider


def pinned_item(name: str, blocks: Sequence[Block]) -> list[Block]:
    """One named pinned item, verbatim between two marker blocks."""
    return [TextBlock(text=f'<pinned name="{name}">'), *blocks, TextBlock(text="</pinned>")]


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class CompactionRecord:
    """One compaction of a thread's model context.

    ``boundary`` counts the stored messages (in ``list_for_thread`` order) that
    the summary replaces in the model's view; they stay in the store untouched.
    ``tokens_before`` and ``images_before`` measured the request that would
    have crossed; ``tokens_after`` (estimated) and ``images_after`` (exact)
    measure the fresh context, before the model has reported on it.
    """

    id: str
    agent_id: str
    thread_id: str
    run_id: str
    turn_id: str
    boundary: int
    reason: str
    summary: str
    tokens_before: int
    images_before: int
    tokens_after: int
    images_after: int
    pinned: list[Block] = field(default_factory=list)
    context_window_tokens: int | None = None
    max_images_per_request: int | None = None
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class ContextMeasure:
    tokens: int
    images: int


def summary_message(summary: str) -> Message:
    return Message(role="user", content=f"{SUMMARY_PREFIX}{summary}{SUMMARY_SUFFIX}")


def fresh_prefix(record: CompactionRecord) -> list[Message]:
    """What replaces the messages before the boundary: the summary, then the pinned blocks."""
    prefix = [summary_message(record.summary)]
    if record.pinned:
        prefix.append(Message(role="user", content=list(record.pinned)))
    return prefix


def context_messages(
    stored: Sequence[Message],
    record: CompactionRecord | None,
    *,
    end: int | None = None,
) -> list[Message]:
    """The model's view of a thread: the latest compaction's prefix, then the
    stored messages from its boundary on (up to ``end``, a stored index)."""
    start = record.boundary if record is not None else 0
    tail = list(stored[start:end])
    return [*fresh_prefix(record), *tail] if record is not None else tail


def usage_start(record: CompactionRecord | None) -> int:
    """First view index whose reported usage describes this context.

    The first kept message may be the assistant turn whose tool results were
    pending; its usage measured the pre-compaction request, so it is skipped.
    """
    return len(fresh_prefix(record)) + 1 if record is not None else 0


def pending_start(view: Sequence[Message], floor: int = 0) -> int:
    """Where the not-yet-answered part of a view begins.

    That is everything after the last assistant message, or that message
    itself when it made tool calls, so a call is never separated from its
    results. ``floor`` when no assistant message follows it.
    """
    for index in range(len(view) - 1, floor - 1, -1):
        message = view[index]
        if message.role == "assistant":
            return index if message.tool_calls else index + 1
    return floor


def count_images(messages: Sequence[Message]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, InlineImageBlock | UrlImageBlock)
        or (isinstance(block, AssetBlock) and block.mime.startswith("image/"))
    )


def estimate_tokens(messages: Sequence[Message], system: str = "") -> int:
    """A character estimate of text a provider has not counted. Images are not
    estimated here; they are counted against the image limit exactly."""
    chars = len(system)
    for message in messages:
        if isinstance(message.content, str):
            chars += len(message.content)
        elif isinstance(message.content, list):
            chars += sum(len(b.text) for b in message.content if isinstance(b, TextBlock))
        chars += len(message.thought_summary or "")
        for call in message.tool_calls or []:
            chars += len(call.function.name) + len(call.function.arguments)
        if message.reasoning_items:
            chars += len(json.dumps(message.reasoning_items, default=str))
    return math.ceil(chars / CHARS_PER_TOKEN)


def measure_request(
    system: str, view: Sequence[Message], *, usage_from: int = 0
) -> ContextMeasure:
    """The last turn's reported usage plus an estimate of what was added since.

    Without a reported turn in this context, the whole request is estimated.
    """
    for index in range(len(view) - 1, usage_from - 1, -1):
        message = view[index]
        if message.role == "assistant" and message.input_tokens is not None:
            reported = message.input_tokens + (message.output_tokens or 0)
            return ContextMeasure(
                reported + estimate_tokens(view[index + 1 :]), count_images(view)
            )
    return ContextMeasure(estimate_tokens(view, system), count_images(view))


def crossed_limits(
    measure: ContextMeasure,
    *,
    context_window_tokens: int | None,
    max_images_per_request: int | None,
    threshold: float,
) -> str | None:
    """``"tokens"``, ``"images"``, both joined by a comma, or ``None`` below both."""
    reasons = []
    if context_window_tokens is not None and measure.tokens > threshold * context_window_tokens:
        reasons.append("tokens")
    if max_images_per_request is not None and measure.images > max_images_per_request:
        reasons.append("images")
    return ",".join(reasons) or None


def compaction_request(view: Sequence[Message], instructions: str = "") -> list[Message]:
    """The summarizing turn's messages: the context being replaced, then the prompt."""
    prompt = COMPACTION_PROMPT if not instructions else f"{COMPACTION_PROMPT}\n\n{instructions}"
    return [*view, Message(role="user", content=prompt)]


__all__ = [
    "COMPACTION_PROMPT",
    "Compaction",
    "CompactionHook",
    "CompactionRecord",
    "PinProvider",
    "ContextMeasure",
    "compaction_request",
    "context_messages",
    "count_images",
    "crossed_limits",
    "estimate_tokens",
    "fresh_prefix",
    "measure_request",
    "pinned_item",
    "pending_start",
    "summary_message",
    "usage_start",
]
