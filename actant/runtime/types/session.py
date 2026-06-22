"""Session message data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from actant.core import JSONObject


class PartKind(StrEnum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    USER_PROMPT = "user_prompt"


class WaitStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


@dataclass
class MessagePart:
    kind: PartKind
    content: str | None = None
    # Multimodal content lives here on USER_PROMPT and TOOL_RESULT parts
    # (matches pydantic-ai's UserPromptPart and ToolReturnPart). When set,
    # supersedes ``content``. Block shape: ``{"type": "text", "text": ...}``
    # or ``{"type": "asset", "storage_key": ..., "mime": ..., "asset_public_id": ...}``.
    content_blocks: list[dict[str, object]] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    args: JSONObject | None = None
    result: dict[str, object] | None = None
    wait_status: WaitStatus | None = None
    # Provider-specific continuation signature. Used by thinking parts
    # for Anthropic/OpenAI reasoning and by Gemini tool_call parts for
    # required function-call thought signatures.
    signature: str | None = None
    reasoning_items: list[object] | None = None
