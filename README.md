# Actant

Actant is a durable Python agent runtime built on
[Temporal](https://temporal.io/). Define agents and tools normally; Actant
handles parallel tools, human approval, deferred work, subagents, suspension,
and crash-safe continuation.

> Actant is pre-1.0. Public APIs may change.

## Why Actant

Agent tools become a distributed-systems problem when calls run in parallel,
wait for people, or outlive a worker. Actant handles that orchestration:

- allowed tools execute concurrently;
- deferred tools pause without holding a worker;
- the next model turn waits for the complete tool group;
- approvals and nested-agent waits surface through the same API;
- Temporal recovers execution after process or worker failure;
- projection stores keep state easy for APIs and UIs to read.

```mermaid
flowchart TB
    Turn["One agent turn emits A, B, and C"]
    A["A: execute → completed"]
    B["B: wait ··· human approves → completed"]
    C["C: execute → completed"]
    Barrier["Durable tool-group barrier"]
    Next["Next agent turn"]

    Turn --> A & B & C
    A --> Barrier
    B --> Barrier
    C --> Barrier
    Barrier --> Next
```

Read [Why Actant?](docs/why-actant.md) for the detailed guarantees and
framework comparison.

## Install

```bash
pip install actant
pip install "actant[openai]"     # optional provider
pip install "actant[anthropic]"  # optional provider
pip install "actant[gemini]"     # optional provider
```

Start a local Temporal development server:

```bash
actant server start
```

The server stays attached so its logs and lifecycle remain visible. Pass
`--detach` only when you intentionally want it to run in the background.

## Quickstart

This complete example streams tokens and then prints the persisted final
response:

```python
import asyncio
from contextlib import suppress
from uuid import uuid4

from actant import AgentDefinition
from actant.llm.messages import Message
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime import AgentRuntime, TemporalRuntimeConfig, TemporalRuntimeWorker
from actant.runtime.events import AgentThreadHooks, StreamListener
from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools import ToolRegistry

responses: asyncio.Queue[Message | Exception] = asyncio.Queue()
stores = InMemoryRuntimeStores()
config = TemporalRuntimeConfig(address="localhost:7233")


class Hooks(AgentThreadHooks):
    async def on_assistant_message(self, message: Message) -> None:
        if not message.tool_calls:
            await responses.put(message)

    async def on_error(self, error: Exception) -> None:
        await responses.put(error)


class Stream(StreamListener):
    async def on_text_delta(self, delta: str) -> None:
        print(delta, end="", flush=True)


agent = AgentDefinition(
    id="assistant",
    name="Assistant",
    persona="You are a useful assistant.",
    llm=FakeLLM(
        [
            FakeResponse(
                text="Hello from Actant.",
                text_chunks=["Hello ", "from ", "Actant."],
            )
        ]
    ),
    tools=ToolRegistry([]),
)
agents = {agent.id: agent}

runtime = AgentRuntime(stores=stores, agents=agents, temporal=config)
worker = TemporalRuntimeWorker(
    stores=stores,
    agents=agents,
    config=config,
    hooks_factory=lambda _thread: Hooks(),
    listener_factory=lambda _thread: Stream(),
)


async def main() -> None:
    worker_task = asyncio.create_task(worker.run())
    try:
        print("Streaming: ", end="", flush=True)
        await runtime.send_message(agent.id, uuid4().hex, "hello")
        response = await asyncio.wait_for(responses.get(), timeout=60)
        if isinstance(response, Exception):
            raise response
        print(f"\nFinal: {response.content}")
    finally:
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task


asyncio.run(main())

# Streaming: Hello from Actant.
# Final: Hello from Actant.
```

`send_message()` durably submits work and returns immediately. `StreamListener`
receives live deltas, hooks receive persisted lifecycle events, and the message
store provides durable reload.

The runtime has three write-side entry points:

```python
thread = runtime.thread(agent.id, uuid4())
await thread.send("Start")
await thread.resolve(tool_call_id, approved=True)
await thread.cancel()
```

The equivalent runtime-level methods remain available when an application
already carries `agent_id` and `thread_id` separately. A thread handle also
exposes `state()`, `messages()`, `waiting_tools()`, and typed live `events()`.

Use `OpenAIProvider`, `AnthropicProvider`, `GeminiProvider`, or `QwenProvider`
in place of `FakeLLM`. Actant never chooses a model ID for you.

## Demo

The included FastAPI + React viewer demonstrates streaming, approvals,
multiple-choice questions, mixed parallel tools, and nested subagents without
an API key:

```bash
just demo-sync
just demo
```

Open `http://localhost:5173`.

## Documentation

- [Core concepts](docs/concepts.md)
- [Runtime architecture](docs/architecture.md)
- [Runtime and deployment](docs/actant-runtime-guide.md)
- [Tools and admission](docs/tools-guide.md)
- [Pauses and deferred work](docs/pauses-and-resume.md)
- [Subagents](docs/subagents.md)
- [Application coordinators](docs/coordinator-guide.md)
- [Release process](docs/releasing.md)

## Development

```bash
just sync
just test
just lint
just typecheck
just package
```

The `justfile` is repository-only. Installed users receive the `actant` CLI;
run `actant server --help` for local Temporal commands.

## License

[MIT](LICENSE)
