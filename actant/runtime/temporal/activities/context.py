"""Dependencies shared by worker-bound Temporal activities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from temporalio.exceptions import ApplicationError

from actant.agents import AgentDefinition
from actant.assets import AssetResolver
from actant.runtime.events.publisher import EventSink
from actant.runtime.events.publisher import ScopedEventSink
from actant.runtime.events.runtime import RuntimeEvents
from actant.llm.messages import Message
from actant.runtime.completion import RunCompletionHandler
from actant.runtime.gate import TurnGate
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.types.threads import AgentThread
from actant.sandbox.base import ArtifactSink, Sandbox
from actant.sandbox.registry import SandboxRegistry

AgentResolver = Callable[[str, str], Awaitable[AgentDefinition]]

MessagePreprocessor = Callable[[list[Message]], Awaitable[list[Message]]]


class ActivityContext:
    """Worker dependencies available to every activity group."""

    def __init__(
        self,
        *,
        stores: RuntimeStores,
        resolve_agent: AgentResolver | None = None,
        assets: AssetResolver | None = None,
        event_sink: EventSink | None = None,
        message_preprocessor: MessagePreprocessor | None = None,
        run_completion_handler: RunCompletionHandler | None = None,
        turn_gate: TurnGate | None = None,
        sandboxes: SandboxRegistry | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> None:
        self.stores = stores
        self.resolve_agent = resolve_agent
        self.assets = assets
        self.event_sink = event_sink
        self.message_preprocessor = message_preprocessor
        self.run_completion_handler = run_completion_handler
        self.turn_gate = turn_gate
        self.sandboxes = sandboxes
        self.artifact_sink = artifact_sink

    async def agent(self, agent_id: str, thread_id: str) -> AgentDefinition:
        if self.resolve_agent is None:
            raise ApplicationError("agent resolution is not configured", non_retryable=True)
        try:
            agent = await self.resolve_agent(agent_id, thread_id)
        except KeyError as exc:
            raise ApplicationError(f"Unknown agent: {agent_id}", non_retryable=True) from exc
        if agent.id != agent_id:
            raise ApplicationError("resolver returned a different agent id", non_retryable=True)
        return agent

    async def sandbox_for(self, agent: AgentDefinition, thread_id: str) -> Sandbox:
        """The thread's sandbox, opened on first use. Refused, not retried, when
        the agent declares no spec or the worker registered no providers."""
        if agent.sandbox is None:
            raise ApplicationError(
                f"agent {agent.id!r} has a tool that needs a sandbox but declares no "
                "AgentDefinition.sandbox",
                non_retryable=True,
            )
        if self.sandboxes is None:
            raise ApplicationError(
                "a tool needs a sandbox but the worker registered no sandbox providers",
                non_retryable=True,
            )
        return await self.sandboxes.for_thread(agent.sandbox, agent.id, thread_id)

    def events(
        self,
        thread: AgentThread,
        *,
        run_id: str | None = None,
        turn_id: str | None = None,
        turn_index: int | None = None,
    ) -> RuntimeEvents:
        return RuntimeEvents(
            thread.id,
            ScopedEventSink(self.event_sink, thread.agent_id, run_id, turn_id, turn_index),
        )
