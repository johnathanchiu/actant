"""Session persistence interface."""

from __future__ import annotations

from typing import Protocol

from actant.llm.messages import Message
from actant.runtime.types.session import MessagePart, WaitStatus


class SessionStore(Protocol):
    async def save_user_message(
        self, thread_id: str, content: str | list[dict[str, object]]
    ) -> str:
        """Persist a user message — string for plain text, block list for
        multimodal (text + asset blocks)."""
        ...

    async def save_assistant_message(self, thread_id: str, parts: list[MessagePart]) -> str: ...

    async def update_tool_result(
        self, thread_id: str, tool_call_id: str, result: dict[str, object]
    ) -> None: ...

    async def update_wait_status(
        self, thread_id: str, tool_call_id: str, status: WaitStatus
    ) -> None: ...

    async def get_conversation(self, thread_id: str) -> list[Message]: ...
