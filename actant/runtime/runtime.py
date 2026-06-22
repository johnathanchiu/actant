"""Friendly runtime wiring.

``AgentRuntime`` is the public client-side facade. It connects to a
Temporal cluster (via ``TemporalExecutor``) and exposes thread
operations: ``send_message``, ``cancel_thread``, ``resolve_tool``,
``get_state``. Application code never holds an orchestrator or worker
reference — those live on the worker process started via
``TemporalRuntimeWorker``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal

from actant.agents import AgentDefinition
from actant.llm.messages import Message
from actant.runtime.executors.base import RuntimeExecutor
from actant.runtime.executors.temporal import TemporalExecutor
from actant.runtime.hooks import AgentThreadHooks, StreamListener
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.types.orchestration import StepResult
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
        executor: Literal["temporal"] | RuntimeExecutor = "temporal",
        hooks_factory: HookFactory | None = None,
        listener_factory: ListenerFactory | None = None,
        message_preprocessor: MessagePreprocessor | None = None,
        temporal: object | None = None,
    ) -> None:
        self.stores = stores
        self.agents = agents
        self.hooks_factory = hooks_factory
        self.listener_factory = listener_factory
        self.message_preprocessor = message_preprocessor
        self.executor = self._build_executor(executor, temporal=temporal)

    def _build_executor(
        self,
        executor: Literal["temporal"] | RuntimeExecutor,
        *,
        temporal: object | None,
    ) -> RuntimeExecutor:
        if not isinstance(executor, str):
            return executor
        if executor == "temporal":
            return TemporalExecutor(
                stores=self.stores,
                agents=self.agents,
                hooks_factory=self.hooks_factory,
                listener_factory=self.listener_factory,
                message_preprocessor=self.message_preprocessor,
                config=temporal,
            )
        raise ValueError(f"Unsupported runtime executor: {executor}")

    async def send_message(
        self,
        agent_id: str,
        thread_id: str,
        content: str | list[dict[str, object]],
    ) -> str:
        return await self.executor.send_message(agent_id, thread_id, content)

    async def run_one(self) -> StepResult:
        return await self.executor.run_one()

    async def run_forever(self, *, idle_sleep: float = 0.1) -> None:
        await self.executor.run_forever(idle_sleep=idle_sleep)

    async def run_until_idle(
        self, agent_id: str, thread_id: str, max_steps: int = 25
    ) -> StepResult:
        return await self.executor.run_until_idle(agent_id, thread_id, max_steps)

    async def cancel_thread(self, agent_id: str, thread_id: str) -> None:
        cancel = getattr(self.executor, "cancel_thread", None)
        if cancel is None:
            raise NotImplementedError("Active executor does not support cancel_thread")
        await cancel(agent_id, thread_id)

    async def resolve_tool(
        self,
        agent_id: str,
        thread_id: str,
        tool_call_id: str,
        *,
        approved: bool | None = None,
        answer: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        resolve = getattr(self.executor, "resolve_tool", None)
        if resolve is None:
            raise NotImplementedError("Active executor does not support resolve_tool")
        await resolve(
            agent_id,
            thread_id,
            tool_call_id,
            approved=approved,
            answer=answer,
            payload=payload,
        )

    async def get_state(self, agent_id: str, thread_id: str) -> object:
        getter = getattr(self.executor, "get_state", None)
        if getter is None:
            raise NotImplementedError("Active executor does not support get_state")
        return await getter(agent_id, thread_id)

    def stop(self) -> None:
        self.executor.stop()


def default_hooks_factory(_thread: AgentThread) -> AgentThreadHooks:
    return AgentThreadHooks()
