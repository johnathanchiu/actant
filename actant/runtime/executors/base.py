"""Runtime executor boundary."""

from __future__ import annotations

from typing import Any, Protocol


class RuntimeExecutor(Protocol):
    async def send_message(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> str: ...

    async def cancel_thread(self, agent_id: str, thread_id: str) -> None: ...

    async def resolve_tool(
        self,
        agent_id: str,
        thread_id: str,
        tool_call_id: str,
        *,
        approved: bool | None = None,
        answer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None: ...

    async def get_state(self, agent_id: str, thread_id: str) -> object: ...
