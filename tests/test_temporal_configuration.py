"""Small tests for configuration values that cross into workflow input.

These checks exist because configuration on the client cannot affect a running
workflow unless it is copied into a serializable workflow payload.
"""

from typing import cast

import pytest
import temporalio.client
import temporalio.worker

from actant.agents import AgentDefinition
from actant.runtime import AgentRuntime
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.types import TemporalRuntimeConfig, ThreadInput
from actant.runtime.temporal.workflow import _history_rotation_threshold


def test_history_rotation_threshold_comes_from_thread_input() -> None:
    payload = ThreadInput(
        agent_id="agent",
        thread_id="thread",
        history_size_threshold=321,
    )

    assert _history_rotation_threshold(payload) == 321


def test_history_rotation_threshold_has_a_safe_minimum() -> None:
    payload = ThreadInput(
        agent_id="agent",
        thread_id="thread",
        history_size_threshold=0,
    )

    assert _history_rotation_threshold(payload) == 1


async def test_worker_takes_its_activity_cap_from_the_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps: list[int | None] = []

    class RecordingWorker:
        def __init__(
            self, *_: object, max_concurrent_activities: int | None, **__: object
        ) -> None:
            caps.append(max_concurrent_activities)

        async def run(self) -> None:
            return None

    async def resolve(agent_id: str, thread_id: str) -> AgentDefinition:
        raise AssertionError("no turn runs")

    monkeypatch.setattr(temporalio.worker, "Worker", RecordingWorker)
    runtime = AgentRuntime(
        client=cast(temporalio.client.Client, object()),
        stores=InMemoryRuntimeStores(),
        config=TemporalRuntimeConfig(max_concurrent_activities=12),
        resolve_agent=resolve,
    )
    await runtime.run_worker()

    assert caps == [12]
