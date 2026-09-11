"""Dependencies shared by worker-bound Temporal activities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from temporalio.exceptions import ApplicationError

from actant.agents import AgentDefinition
from actant.llm.messages import Message
from actant.runtime.completion import RunCompletionHandler
from actant.runtime.events.lifecycle import AgentThreadHooks
from actant.runtime.events.streaming import StreamListener
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.types.threads import AgentThread
from actant.sandbox.base import ArtifactSink, Sandbox
from actant.sandbox.registry import SandboxRegistry

HookFactory = Callable[[AgentThread], AgentThreadHooks]
ListenerFactory = Callable[[AgentThread], StreamListener]
MessagePreprocessor = Callable[[list[Message]], Awaitable[list[Message]]]


class ActivityContext:
    """Worker dependencies available to every activity group."""

    def __init__(
        self,
        *,
        stores: RuntimeStores,
        agents: Mapping[str, AgentDefinition],
        hooks_factory: HookFactory | None = None,
        listener_factory: ListenerFactory | None = None,
        message_preprocessor: MessagePreprocessor | None = None,
        run_completion_handler: RunCompletionHandler | None = None,
        sandboxes: SandboxRegistry | None = None,
        artifact_sink: ArtifactSink | None = None,
    ) -> None:
        self.stores = stores
        self.agents = agents
        self.hooks_factory = hooks_factory
        self.listener_factory = listener_factory
        self.message_preprocessor = message_preprocessor
        self.run_completion_handler = run_completion_handler
        self.sandboxes = sandboxes
        self.artifact_sink = artifact_sink

    def _require_agent(self, agent_id: str) -> AgentDefinition:
        agent = self.agents.get(agent_id)
        if agent is None:
            raise ApplicationError(f"Unknown agent: {agent_id}", non_retryable=True)
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
        thread = await self.stores.threads.get_or_create(agent.id, thread_id)
        return await self.sandboxes.for_thread(agent.sandbox, thread)

    def _hooks(self, thread: AgentThread) -> AgentThreadHooks:
        if self.hooks_factory is None:
            return AgentThreadHooks()
        return self.hooks_factory(thread)

    def _listener(self, thread: AgentThread) -> StreamListener:
        if self.listener_factory is None:
            return StreamListener()
        return self.listener_factory(thread)
