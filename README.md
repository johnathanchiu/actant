# Actant

Actant is a small Python runtime kernel for long-lived agents with
durable inboxes, tools, admission, memory, and replay — built on top of
[Temporal](https://temporal.io/).

The package is intentionally domain-neutral. Applications provide
agents, tools, memory content, and UI.

## Provider Adapters

Install only the SDKs you need:

```bash
uv add --extra openai actant
uv add --extra anthropic actant
uv add --extra gemini actant
uv add --extra qwen actant
uv add actant
```

Supported adapters:

- `OpenAIProvider` for OpenAI Responses API completions
- `AnthropicProvider` for Anthropic Messages API completions
- `GeminiProvider` for Gemini `generate_content`
- `QwenProvider` for DashScope's OpenAI-compatible endpoint

You can route by model prefix:

```python
from actant.llm import llm_for_model

llm = llm_for_model("gpt-5.2")
```

## Memory

Memory is modeled as durable cards plus optional tools. Register the
tools for an agent namespace when you want the agent to manage its own
memory:

```python
from actant.memory import memory_tools
from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools import ToolRegistry

stores = InMemoryRuntimeStores()
tools = ToolRegistry(memory_tools(stores.memory, "agent_pm_1"))
```

Available memory tools: `list_memory_cards`, `read_memory_card`,
`search_memory`, `write_memory_card`, `append_memory_card`.

Applications can also build prompt-cache-friendly memory context
explicitly via `build_memory_context`.

## Agent Runtime

`AgentRuntime` is the client-side facade. It signals a Temporal
workflow per `(agent_id, thread_id)`:

```python
from actant import AgentDefinition
from actant.runtime import AgentRuntime, TemporalRuntimeConfig
from actant.runtime.stores import InMemoryRuntimeStores
from actant.tools import ToolRegistry

stores = InMemoryRuntimeStores()
agent = AgentDefinition(
    id="assistant",
    name="Assistant",
    persona="You are a useful assistant.",
    llm=llm,
    tools=ToolRegistry([...]),
)

runtime = AgentRuntime(
    stores=stores,
    agents={agent.id: agent},
    temporal=TemporalRuntimeConfig(address="localhost:7233"),
)

await runtime.send_message("assistant", "thread_1", "hello")
```

Behind the scenes, `send_message` issues a `signal_with_start` against
the thread workflow — the workflow is created on first contact and
signalled on subsequent calls. Idempotent.

To actually execute the workflow, run a worker process:

```python
from actant.runtime import TemporalRuntimeWorker

worker = TemporalRuntimeWorker(
    stores=stores,
    agents={agent.id: agent},
    config=TemporalRuntimeConfig(address="localhost:7233"),
    hooks_factory=my_hooks_factory,
    listener_factory=my_listener_factory,
)
await worker.run()
```

Run as many worker processes as you want — Temporal's task queue
load-balances workflows + activities across them.

## Workflow Anatomy

`AgentThreadWorkflow`:

- **Outer loop** = thread lifetime. Parks on `wait_condition(inbox)`
  until a user message arrives.
- **Inner loop** = one run. Drains the inbox, runs turns until the
  agent has nothing more to say, hits the per-run turn budget, or is
  cancelled.
- **Turn** = one `run_turn` activity invocation (one LLM call).
- **Tool fan-out** = parallel `admit_tool` activities, followed by
  `execute_tool` for allowed calls or `await_external_resolution` for
  deferred calls. Deferred calls park as Temporal async activities, not
  workflow signals.

## Cancel + Resolve

```python
# Cancel an in-flight thread
await runtime.cancel_thread("assistant", "thread_1")

# Resolve a deferred (WAIT) tool call
await runtime.resolve_tool(
    "assistant", "thread_1", tool_call_id,
    approved=True,
    answer="ok",
)

# Read live state without disturbing the workflow
state = await runtime.get_state("assistant", "thread_1")
```

## Tool Admission

Tools can decide whether a requested call can execute immediately. Most
tools don't define admission logic and run by default. A tool that
needs approval, consensus, a timer, or another external condition can
implement `can_execute` and return `allow`, `block`, or `wait`:

```python
from actant.tools import ToolDecision, ToolWaitRequest

async def can_execute(self, call, invocation, context):
    if await approval_store.approved(call.id):
        return ToolDecision.allow()
    return ToolDecision.wait(
        ToolWaitRequest(
            kind="human_review",
            prompt="waiting for human review",
            payload={"tool_call_id": call.id},
        )
    )
```

The `admit_tool` activity records WAITING calls and emits an
`on_tool_waiting` hook. Resolve via `runtime.resolve_tool`, which
completes the parked Temporal async activity after persisting the
resolved tool result.

## Runtime Stores

`actant.runtime.stores` ships projection-only stores. Coordination
(durable inbox, single-writer per thread, work scheduling) lives in
Temporal — these stores hold the readable side of runtime state.

In-memory variants for tests/local dev:
- `InMemoryRuntimeStores` — drop-in for the projection contracts.

Postgres backend:
- `actant.runtime.stores.postgres.sqlalchemy` — DeclarativeBase models +
  `ACTANT_RUNTIME_METADATA` you can plug into your own Alembic setup.

Tables: `actant_threads`, `actant_runs`, `actant_messages`,
`actant_message_parts`, `actant_tool_calls`, `actant_memory_cards`.

Applications can implement custom stores against the contracts in
`actant.runtime.interfaces.stores`.

## Subagents

Subagent invocation is represented as a normal tool. Register `TaskTool`
when an agent is allowed to delegate work:

```python
from actant.tools import InMemorySubagentRegistry, TaskTool, ToolRegistry

registry = InMemorySubagentRegistry({"researcher": researcher_invoker})
tools = ToolRegistry([TaskTool(registry)])
```

The invoker is app-owned, so a subagent can be another Actant
coordinator, a remote worker, a durable workflow, or a test double.

## Hooks

`AgentThreadHooks` exposes async callbacks fired from inside activities
for persistence, streaming, and observability:

- user/assistant messages
- turn start
- text/thinking deltas (via `StreamListener`)
- tool calls, tool results, waiting calls, resolved calls
- completion and errors

Hooks announce; they don't write. The canonical state lives in the
stores. Apps wire hooks to their pubsub/SSE/websocket layer of choice
via `PublishingThreadHooks` / `PublishingStreamListener` or custom
implementations.

## Examples

See `examples/` for runnable compositions of agents, tools, memory, and
admission. `examples/demo/` is the worked FastAPI + React demo.

For apps that need multiple agents, `task()`-style delegation, or
robust state recovery when Temporal and the store diverge, read
[`docs/coordinator-guide.md`](docs/coordinator-guide.md). The
`examples/demo/` directory ships a worked example (`DemoCoordinator`)
that uses the framework's coordinator primitives.

## Local Development

```bash
just sync                  # install deps + Temporal extra
just temporal-up-detached  # start local Temporal stack
just test                  # run tests (uses in-memory WorkflowEnvironment)
just temporal-smoke        # full docker round-trip
```
