"""Dependencies shared by worker-bound Temporal activities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from temporalio.exceptions import ApplicationError

from actant.agents import AgentDefinition
from actant.llm.messages import Message
from actant.runtime.completion import RunCompletionHandler
from actant.runtime.events import AgentThreadHooks, StreamListener
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.types.threads import AgentThread

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
    ) -> None:
        self.stores = stores
        self.agents = agents
        self.hooks_factory = hooks_factory
        self.listener_factory = listener_factory
        self.message_preprocessor = message_preprocessor
        self.run_completion_handler = run_completion_handler

    def _require_agent(self, agent_id: str) -> AgentDefinition:
        agent = self.agents.get(agent_id)
        if agent is None:
            raise ApplicationError(f"Unknown agent: {agent_id}", non_retryable=True)
        return agent

    def _hooks(self, thread: AgentThread) -> AgentThreadHooks:
        if self.hooks_factory is None:
            return AgentThreadHooks()
        return self.hooks_factory(thread)

    def _listener(self, thread: AgentThread) -> StreamListener:
        if self.listener_factory is None:
            return StreamListener()
        return self.listener_factory(thread)
