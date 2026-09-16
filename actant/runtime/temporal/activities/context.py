"""Dependencies shared by worker-bound Temporal activities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import logging

from temporalio.exceptions import ApplicationError

from actant.agents import AgentDefinition
from actant.assets import AssetResolver
from actant.runtime.events.publisher import EventSink
from actant.runtime.events.observers import ObservedHooks, ObservedStream, ScopedEventSink
from actant.runtime.events.lifecycle import PublishingThreadHooks
from actant.runtime.events.streaming import PublishingStreamListener
from actant.llm.messages import Message
from actant.runtime.completion import RunCompletionHandler
from actant.runtime.events.lifecycle import AgentThreadHooks
from actant.runtime.events.streaming import StreamListener
from actant.runtime.gate import TurnGate
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.types.threads import AgentThread
from actant.sandbox.base import ArtifactSink, Sandbox
from actant.sandbox.registry import SandboxRegistry

AgentResolver = Callable[[str, str], Awaitable[AgentDefinition]]

HookFactory = Callable[[AgentThread], AgentThreadHooks]
ListenerFactory = Callable[[AgentThread], StreamListener]
MessagePreprocessor = Callable[[list[Message]], Awaitable[list[Message]]]


log = logging.getLogger(__name__)


class ActivityContext:
    """Worker dependencies available to every activity group."""

    def __init__(
        self,
        *,
        stores: RuntimeStores,
        resolve_agent: AgentResolver | None = None,
        assets: AssetResolver | None = None,
        event_sink: EventSink | None = None,
        hooks_factory: HookFactory | None = None,
        listener_factory: ListenerFactory | None = None,
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
        self.hooks_factory = hooks_factory
        self.listener_factory = listener_factory
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

    async def _sandbox_for(self, agent: AgentDefinition, thread_id: str) -> Sandbox:
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

    def _hooks(
        self, thread: AgentThread, *, run_id: str | None = None, turn_id: str | None = None
    ) -> AgentThreadHooks:
        if self.hooks_factory is not None:
            try:
                return ObservedHooks(self.hooks_factory(thread))
            except Exception:
                log.exception("runtime hook factory failed")
                return AgentThreadHooks()
        if self.event_sink is None:
            return AgentThreadHooks()
        return PublishingThreadHooks(
            thread.id, publisher=ScopedEventSink(self.event_sink, thread.agent_id, run_id, turn_id)
        )

    def _listener(
        self, thread: AgentThread, *, run_id: str | None = None, turn_id: str | None = None
    ) -> StreamListener:
        if self.listener_factory is not None:
            try:
                return ObservedStream(self.listener_factory(thread))
            except Exception:
                log.exception("runtime listener factory failed")
                return StreamListener()
        if self.event_sink is None:
            return StreamListener()
        return PublishingStreamListener(
            thread.id, publisher=ScopedEventSink(self.event_sink, thread.agent_id, run_id, turn_id)
        )
