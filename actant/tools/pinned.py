"""``pin_note``: the agent's own durable notes, kept across context compaction.

A compaction replaces the conversation with a summary, and a summary can drop
things. A note pinned here is re-injected verbatim after every summary, however
many compactions the thread goes through. Give the tool the runtime's
``stores.pinned_notes``.
"""

from __future__ import annotations

from typing import Protocol

from actant.core import JSONObject
from actant.tools.base import (
    BaseDeclarativeTool,
    BaseToolInvocation,
    CallContext,
    ToolInvocation,
    ToolResult,
    make_tool_schema,
)


class NoteWriter(Protocol):
    """The part of ``actant.runtime.interfaces.stores.PinnedNoteStore`` the tool uses."""

    async def pin(self, agent_id: str, thread_id: str, key: str, content: str) -> None: ...

    async def unpin(self, agent_id: str, thread_id: str, key: str) -> None: ...


DESCRIPTION = (
    "Pin a note that must survive when your context is compacted into a summary: a "
    "checklist, decisions, constraints, open items. The note under `key` is replaced "
    "with `text` and shown to you verbatim after every summary. An empty `text` unpins it."
)


class _PinNoteInvocation(BaseToolInvocation[JSONObject, JSONObject]):
    def __init__(self, params: JSONObject, store: NoteWriter, ctx: CallContext) -> None:
        super().__init__(params)
        self._store = store
        self._ctx = ctx

    def get_description(self) -> str:
        return "Pinning a note"

    async def execute(self) -> ToolResult:
        key, text = self.params.get("key"), self.params.get("text")
        if not isinstance(key, str) or not key.strip() or not isinstance(text, str):
            return ToolResult.fail("`key` must be a non-empty string and `text` a string")
        if text:
            await self._store.pin(self._ctx.agent_id, self._ctx.thread_id, key, text)
            return ToolResult.ok({"pinned": key})
        await self._store.unpin(self._ctx.agent_id, self._ctx.thread_id, key)
        return ToolResult.ok({"unpinned": key})


class PinNoteTool(BaseDeclarativeTool):
    def __init__(self, store: NoteWriter) -> None:
        super().__init__(
            "pin_note",
            make_tool_schema(
                "pin_note",
                DESCRIPTION,
                parameters={
                    "key": {"type": "string", "description": "The note's name."},
                    "text": {"type": "string", "description": "The whole note."},
                },
                required=["key", "text"],
            ),
        )
        self._store = store

    async def build(self, params: JSONObject, ctx: CallContext) -> ToolInvocation:
        return _PinNoteInvocation(params, self._store, ctx)
