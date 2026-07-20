"""Worker-side registration for the Temporal runtime."""

from __future__ import annotations

from collections.abc import Mapping

import temporalio.client
import temporalio.worker

from actant.agents import AgentDefinition
from actant.runtime.interfaces.stores import RuntimeStores
from actant.runtime.temporal.activities import (
    HookFactory,
    ListenerFactory,
    MessagePreprocessor,
    TemporalRuntimeActivities,
)
from actant.runtime.temporal.types import TemporalRuntimeConfig
from actant.runtime.temporal.workflow import AgentThreadWorkflow


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
    ) -> None:
        self.config = config or TemporalRuntimeConfig()
        self._activities = TemporalRuntimeActivities(
            stores=stores,
            agents=agents,
            hooks_factory=hooks_factory,
            listener_factory=listener_factory,
            message_preprocessor=message_preprocessor,
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
