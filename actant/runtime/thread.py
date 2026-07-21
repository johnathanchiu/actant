"""Thread-scoped convenience API for applications."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from actant.llm.messages import Message
from actant.runtime.events.publisher import EventSource
from actant.runtime.events.types import ThreadEvent
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.temporal.types import ThreadStateView
from actant.tools.calls import ToolCallRecord, ToolCallStatus


class ThreadRuntime(Protocol):
    """Runtime capabilities used by a thread-scoped handle."""

    stores: RuntimeStores
    event_source: EventSource | None

    async def send_message(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> str: ...

    async def resolve_tool_call(
        self,
        agent_id: str,
        thread_id: str,
        tool_call_id: str,
        *,
        approved: bool | None = None,
        answer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None: ...

    async def cancel_thread(self, agent_id: str, thread_id: str) -> None: ...

    async def get_state(self, agent_id: str, thread_id: str) -> ThreadStateView: ...


@dataclass(frozen=True)
class ThreadHandle:
    """Commands and observations scoped to one ``(agent_id, thread_id)``."""

    runtime: ThreadRuntime
    agent_id: str
    thread_id: str

    async def send(self, content: str | list[dict[str, object]]) -> str:
        """Durably submit a user message and return the Temporal workflow id."""
        return await self.runtime.send_message(self.agent_id, self.thread_id, content)

    async def resolve(
        self,
        tool_call_id: str,
        *,
        approved: bool | None = None,
        answer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Resolve one waiting tool call on this thread."""
        await self.runtime.resolve_tool_call(
            self.agent_id,
            self.thread_id,
            tool_call_id,
            approved=approved,
            answer=answer,
            payload=payload,
        )

    async def cancel(self) -> None:
        """Cancel the thread's Temporal workflow."""
        await self.runtime.cancel_thread(self.agent_id, self.thread_id)

    async def state(self) -> ThreadStateView:
        return await self.runtime.get_state(self.agent_id, self.thread_id)

    async def messages(self) -> list[Message]:
        return await self.runtime.stores.messages.list_for_thread(self.agent_id, self.thread_id)

    async def waiting_tools(self) -> list[ToolCallRecord]:
        records = await self.runtime.stores.tool_calls.get_open_for_thread(
            self.agent_id, self.thread_id
        )
        return [record for record in records if record.status is ToolCallStatus.WAITING]

    async def events(self) -> AsyncIterator[ThreadEvent]:
        """Consume live events; reload projections after reconnecting.

        The source is observational and is never required for workflow
        correctness. Subscribe before ``send`` when the earliest streaming
        deltas matter.
        """
        source = self.runtime.event_source
        if source is None:
            raise RuntimeError("AgentRuntime has no event source configured")
        async for payload in source.subscribe(f"thread:{self.thread_id}"):
            yield ThreadEvent.from_dict(payload)
