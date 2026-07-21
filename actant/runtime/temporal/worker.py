"""Worker-side registration for the Temporal runtime."""

from __future__ import annotations

from collections.abc import Mapping

import temporalio.client
import temporalio.worker

from actant.agents import AgentDefinition
from actant.runtime.completion import RunCompletionHandler
from actant.runtime.events.lifecycle import AgentThreadHooks, PublishingThreadHooks
from actant.runtime.events.publisher import EventSink
from actant.runtime.events.streaming import PublishingStreamListener, StreamListener
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.temporal.activities import (
    HookFactory,
    ListenerFactory,
    MessagePreprocessor,
    TemporalRuntimeActivities,
)
from actant.runtime.temporal.types import TemporalRuntimeConfig
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.runtime.types.threads import AgentThread


def _publishing_hooks_factory(sink: EventSink) -> HookFactory:
    def create_hooks(thread: AgentThread) -> AgentThreadHooks:
        return PublishingThreadHooks(thread.id, publisher=sink)

    return create_hooks


def _publishing_listener_factory(sink: EventSink) -> ListenerFactory:
    def create_listener(thread: AgentThread) -> StreamListener:
        return PublishingStreamListener(thread.id, publisher=sink)

    return create_listener


class TemporalRuntimeWorker:
    """Poll Temporal and host Actant's workflow and activity implementations.

    The worker is a deployment role, not the client used to send messages.
    It needs the same task queue and namespace as the client plus the stores,
    agent definitions, and observer factories used by activity execution.
    """

    def __init__(
        self,
        *,
        stores: RuntimeStores,
        agents: Mapping[str, AgentDefinition],
        config: TemporalRuntimeConfig | None = None,
        hooks_factory: HookFactory | None = None,
        listener_factory: ListenerFactory | None = None,
        message_preprocessor: MessagePreprocessor | None = None,
        run_completion_handler: RunCompletionHandler | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.config = config or TemporalRuntimeConfig()
        sink = event_sink or getattr(stores, "publisher", None)
        if hooks_factory is None and sink is not None:
            hooks_factory = _publishing_hooks_factory(sink)
        if listener_factory is None and sink is not None:
            listener_factory = _publishing_listener_factory(sink)
        self._activities = TemporalRuntimeActivities(
            stores=stores,
            agents=agents,
            hooks_factory=hooks_factory,
            listener_factory=listener_factory,
            message_preprocessor=message_preprocessor,
            run_completion_handler=run_completion_handler,
        )

    async def run(self) -> None:
        client = await temporalio.client.Client.connect(
            self.config.address,
            namespace=self.config.namespace,
        )
        worker = temporalio.worker.Worker(
            client,
            task_queue=self.config.task_queue,
            workflows=[AgentThreadWorkflow],
            activities=self._activities.all,
        )
        await worker.run()
