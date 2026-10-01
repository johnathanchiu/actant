"""Model-context compaction: the one place Actant ever reduces what the model sees.

Nothing is truncated and no stored message is rewritten. When the next request would
cross a limit (a fraction of the context window, or the provider's image cap), one call
on the same model, with no tools, summarizes the conversation, and a compaction row
(``kind="compaction"``, one :class:`~actant.blocks.CompactionBlock`) is appended. The
model then sees:

    system prompt, the summary, the kept messages, the rows after the compaction row

The kept messages are the latest one of each tag in ``CompactionConfig.keep``, and the
turn that was still open (an assistant tool call and its results), carried whole.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

from actant.blocks import (
    AssetBlock,
    CompactionBlock,
    InlineImageBlock,
    PromptBlock,
    TextBlock,
    UrlImageBlock,
)
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
7. Image index: images before this point will not be shown again. For each \
notable one, its id (from the [image id=...] label before it) and a one-line \
caption. `recall_image(id)` shows it again.

Write plain prose and lists. Do not call tools. Do not address the user.
</compaction_request>"""

SUMMARY_PREFIX = (
    "<context_summary>\nThis conversation was compacted. It continues from this "
    "summary of everything before it:\n\n"
)
SUMMARY_SUFFIX = "\n</context_summary>"

#: Characters per token for the part of a request no provider has counted yet.
CHARS_PER_TOKEN = 4
#: The most a summary may write. Less when the window's remaining margin is smaller.
SUMMARY_MAX_OUTPUT_TOKENS = 16_000


@dataclass(frozen=True)
class ContextMeasure:
    tokens: int
    images: int


@dataclass
class ModelView:
    """The messages the model is sent, each with its stored message id (``None`` for
    the rendered summary), and where reported usage describes this context."""

    messages: list[Message]
    ids: list[str | None]
    usage_from: int = 0
    compaction: CompactionBlock | None = None
    compaction_id: str | None = None


def compaction_of(message: Message) -> CompactionBlock | None:
    if message.kind != "compaction" or not isinstance(message.content, list):
        return None
    return cast(CompactionBlock, message.content[0])


def summary_message(summary: str) -> Message:
    return Message(role="user", content=f"{SUMMARY_PREFIX}{summary}{SUMMARY_SUFFIX}")


def retained(message: Message) -> Message:
    """A kept tagged message whose turn the summary replaced.

    A user message is sent as it is. A tool result would be rejected without the
    assistant call before it, and that call belongs to a replaced turn (with its
    parallel siblings, reasoning items and thinking signatures), so replaying the
    pair out of place is not safe either: the result becomes a user message that
    says what it is, its content verbatim, images included.
    """
    if message.role == "user":
        return message
    label = TextBlock(
        text=f'<retained tag="{message.tag}">Kept verbatim across compaction: the latest '
        f"{message.role} message with this tag.</retained>"
    )
    body: list[PromptBlock] = (
        list(message.content)
        if isinstance(message.content, list)
        else [TextBlock(text=message.content or "")]
    )
    return Message(role="user", content=[label, *body], tag=message.tag)


def build_view(rows: Sequence[Message]) -> ModelView:
    """The model's view from ``list_for_model`` rows: the latest compaction rendered as
    its summary, kept rows (tool results whose call is not kept, as :func:`retained`),
    then every row after it. Compaction rows never reach a provider."""
    at = next((i for i in range(len(rows) - 1, -1, -1) if rows[i].kind == "compaction"), None)
    if at is None:
        return ModelView(list(rows), [m.id for m in rows])
    block = compaction_of(rows[at])
    assert block is not None
    view = ModelView(
        [summary_message(block.summary)], [None], compaction=block, compaction_id=rows[at].id
    )
    calls = set[str]()
    for position, message in enumerate(rows):
        if message.kind == "compaction":
            continue
        if position < at and message.role == "tool" and message.tool_call_id not in calls:
            message = retained(message)
        calls.update(c.id for c in message.tool_calls or [])
        view.messages.append(message)
        view.ids.append(rows[position].id)
    # Usage reported before the compaction row measured the old context.
    view.usage_from = len(view.messages) - (len(rows) - 1 - at)
    return view


def kept_ids(history: Sequence[Message], before: int, tags: Sequence[str]) -> list[str]:
    """The ids of the latest message of each tag before position ``before``."""
    latest = {
        m.tag: m.id
        for m in history[:before]
        if m.tag in tags and m.kind == "message" and m.id is not None
    }
    return list(latest.values())


def pending_start(view: Sequence[Message], floor: int = 0) -> int:
    """Where the not-yet-answered part of a view begins: after the last assistant
    message, or at it when it made tool calls, so a call is never separated from its
    results. ``floor`` when no assistant message follows it."""
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
    """The last turn's reported usage plus an estimate of what was added since; the
    whole request estimated when no turn in this context reported usage."""
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
    image_threshold: float = 1.0,
) -> str | None:
    """``"tokens"``, ``"images"``, both joined by a comma, or ``None`` below both."""
    reasons = []
    if context_window_tokens is not None and measure.tokens > threshold * context_window_tokens:
        reasons.append("tokens")
    if (
        max_images_per_request is not None
        and measure.images > image_threshold * max_images_per_request
    ):
        reasons.append("images")
    return ",".join(reasons) or None


def compaction_request(
    view: Sequence[Message], instructions: str = "", prompt: str | None = None
) -> list[Message]:
    """The summary call's messages: the context being replaced, each stored image
    labelled with its id for the image index, then the prompt (``prompt``, else
    ``COMPACTION_PROMPT``) with ``instructions`` after it."""
    labelled: list[Message] = []
    for message in view:
        if isinstance(message.content, list) and any(
            isinstance(b, AssetBlock) for b in message.content
        ):
            blocks: list[PromptBlock] = []
            for block in message.content:
                if isinstance(block, AssetBlock) and block.mime.startswith("image/"):
                    blocks.append(TextBlock(text=f"[image id={image_id(block)}]"))
                blocks.append(block)
            message = replace(message, content=blocks)
        labelled.append(message)
    text = COMPACTION_PROMPT if prompt is None else prompt
    if instructions:
        text = f"{text}\n\n{instructions}"
    return [*labelled, Message(role="user", content=text)]


def image_id(block: AssetBlock) -> str:
    return block.asset_public_id or block.storage_key


__all__ = [
    "COMPACTION_PROMPT",
    "ContextMeasure",
    "ModelView",
    "build_view",
    "compaction_of",
    "compaction_request",
    "count_images",
    "crossed_limits",
    "estimate_tokens",
    "image_id",
    "kept_ids",
    "measure_request",
    "pending_start",
    "retained",
    "summary_message",
]
