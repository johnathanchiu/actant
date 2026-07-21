"""Friendly runtime wiring.

``AgentRuntime`` is the public client-side facade. It connects to a
Temporal cluster (via ``TemporalRuntimeClient``) and exposes thread
operations: ``send_message``, ``resolve_tool_call``, ``cancel_thread``,
``get_state``. Application code never holds an orchestrator or worker
reference — those live on the worker process started via
``TemporalRuntimeWorker``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from uuid import UUID

from actant.agents import AgentDefinition
from actant.llm.messages import Message
from actant.runtime.events.lifecycle import AgentThreadHooks
from actant.runtime.events.publisher import EventSource
from actant.runtime.events.streaming import StreamListener
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.temporal.client import TemporalRuntimeClient
from actant.runtime.temporal.types import ThreadStateView
from actant.runtime.thread import ThreadHandle
from actant.runtime.types.threads import AgentThread

HookFactory = Callable[[AgentThread], AgentThreadHooks]
ListenerFactory = Callable[[AgentThread], StreamListener]
MessagePreprocessor = Callable[[list[Message]], Awaitable[list[Message]]]


class AgentRuntime:
    """Runtime facade. Drives a Temporal-backed thread workflow per ``(agent_id, thread_id)``."""

    def __init__(
        self,
        *,
        stores: RuntimeStores,
        agents: Mapping[str, AgentDefinition],
        hooks_factory: HookFactory | None = None,
        listener_factory: ListenerFactory | None = None,
        message_preprocessor: MessagePreprocessor | None = None,
        event_source: EventSource | None = None,
        temporal: object | None = None,
    ) -> None:
        self.stores = stores
        self.agents = agents
        self.hooks_factory = hooks_factory
        self.listener_factory = listener_factory
        self.message_preprocessor = message_preprocessor
        self.event_source = event_source or getattr(stores, "publisher", None)
        self._client = TemporalRuntimeClient(
            stores=self.stores,
            agents=self.agents,
            hooks_factory=self.hooks_factory,
            listener_factory=self.listener_factory,
            message_preprocessor=self.message_preprocessor,
            config=temporal,
        )

    async def send_message(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> str:
        return await self._client.send_message(agent_id, thread_id, content)

    async def cancel_thread(self, agent_id: str, thread_id: str) -> None:
        await self._client.cancel_thread(agent_id, thread_id)

    async def resolve_tool_call(
        self,
        agent_id: str,
        thread_id: str,
        tool_call_id: str,
        *,
        approved: bool | None = None,
        answer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self._client.resolve_tool_call(
            agent_id,
            thread_id,
            tool_call_id,
            approved=approved,
            answer=answer,
            payload=payload,
        )

    async def get_state(self, agent_id: str, thread_id: str) -> ThreadStateView:
        return await self._client.get_state(agent_id, thread_id)

    def thread(self, agent_id: str, thread_id: str | UUID) -> ThreadHandle:
        """Return a thread-scoped command and observation handle."""
        return ThreadHandle(self, agent_id=agent_id, thread_id=str(thread_id))


def default_hooks_factory(_thread: AgentThread) -> AgentThreadHooks:
    return AgentThreadHooks()
