from __future__ import annotations

import asyncio
import uuid

from temporalio.client import Client

from actant.agents import AgentDefinition
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime, TemporalRuntimeConfig, TemporalRuntimeWorker
from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools import ToolRegistry


async def wait_for_temporal(config: TemporalRuntimeConfig) -> None:
    last_error: Exception | None = None
    for _ in range(60):
        try:
            await Client.connect(config.address, namespace=config.namespace)
            return
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = exc
            await asyncio.sleep(1)
    raise RuntimeError(
        f"Temporal did not become ready at {config.address}"
    ) from last_error


async def main() -> None:
    run_id = uuid.uuid4().hex[:8]
    thread_id = f"thread_smoke_{run_id}"
    config = TemporalRuntimeConfig(task_queue=f"actant-runtime-smoke-{run_id}")
    await wait_for_temporal(config)

    stores = InMemoryRuntimeStores()
    agent = AgentDefinition(
        id="agent_1",
        name="tester",
        persona="test",
        llm=FakeLLM([FakeResponse(text="done")]),
        tools=ToolRegistry(),
    )
    agents = {agent.id: agent}

    worker = TemporalRuntimeWorker(stores=stores, agents=agents, config=config)
    worker_task = asyncio.create_task(worker.run())
    runtime = AgentRuntime(
        stores=stores, agents=agents, temporal=config
    )

    try:
        await asyncio.sleep(0.5)
        if worker_task.done():
            worker_task.result()
        await runtime.send_message(agent.id, thread_id, "hello")
        for _ in range(50):
            messages = await stores.messages.list_for_thread(agent.id, thread_id)
            transcript = [(message.role, message.content) for message in messages]
            if transcript == [("user", "hello"), ("assistant", "done")]:
                break
            await asyncio.sleep(0.2)

        messages = await stores.messages.list_for_thread(agent.id, thread_id)
        transcript = [(message.role, message.content) for message in messages]
        assert transcript == [("user", "hello"), ("assistant", "done")]
        print("Temporal runtime smoke passed")
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(main())
