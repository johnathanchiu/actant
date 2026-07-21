# Actant runtime guide

Actant uses one Temporal workflow for each `(agent_id, thread_id)`. Client code
signals workflows; worker code hosts the model, tools, stores, and activities.

Read [core concepts](concepts.md) first if thread, run, and turn are not yet
familiar.

## Install

Install the provider-neutral runtime plus the model SDKs your worker uses:

```bash
pip install actant
pip install "actant[openai]"  # choose only the provider extras you need
```

Actant does not select a “latest” model. Pass a model ID from application
configuration to the corresponding provider adapter.

## Start Temporal locally

The installed CLI can manage a Docker-backed Temporal server for development:

```bash
actant server start
```

It listens on `localhost:7233`, with the Temporal UI at
`http://localhost:8233`. It stays attached by default so you can see its logs
and stop it with `Ctrl-C`. Pass `--detach` to run it in the background; then use
`actant server status` to inspect it and `actant server stop` to stop it while
retaining its data. `actant server reset` stops it and deletes that data.

This command is a local convenience, not the production deployment model.
Production clients and workers should connect to an independently managed
Temporal service through `TemporalRuntimeConfig`.

For local overrides, `server start` accepts `--port`, `--ui-port`, and
`--no-ui`. Options placed before the action select a different Compose file,
project, or compatible command:

```bash
actant server start \
  --compose-file ./temporal.yml \
  --project-name my-temporal \
  --compose-command "podman compose"
```

Every override also has an environment form; run `actant server --help` and
`actant server start --help` for the complete surface.

## Define an agent

```python
from actant import AgentDefinition
from actant.llm.providers import OpenAIProvider
from actant.tools import ToolRegistry

llm = OpenAIProvider(model_id=settings.model_id)
agent = AgentDefinition(
    id="assistant",
    name="Assistant",
    persona="You are a careful assistant.",
    llm=llm,
    tools=ToolRegistry([]),
)
agents = {agent.id: agent}
```

Use an explicit application setting for `model_id`. Provider model catalogs
change independently of Actant releases.

## Create the runtime client

```python
from actant.runtime import AgentRuntime, TemporalRuntimeConfig
from actant.runtime.stores import InMemoryRuntimeStores

stores = InMemoryRuntimeStores()
config = TemporalRuntimeConfig(
    address="localhost:7233",
    namespace="default",
    task_queue="actant-runtime",
)
runtime = AgentRuntime(stores=stores, agents=agents, temporal=config)
```

In-memory stores are suitable for tests and local examples. Use the included
SQLAlchemy Postgres stores, or implement the store protocols, when projections
must survive process restarts.

## Run a worker

The runtime facade does not execute model calls by itself. A worker must poll
the same Temporal namespace and task queue:

```python
from actant.runtime import TemporalRuntimeWorker

worker = TemporalRuntimeWorker(
    stores=stores,
    agents=agents,
    config=config,
    hooks_factory=my_hooks_factory,
    listener_factory=my_listener_factory,
)
await worker.run()
```

Client and worker can live in one service for local development or separate
processes in production. Every worker that may receive an activity must be able
to resolve the referenced agent definition and access compatible projection
stores.

## Send messages

```python
from uuid import uuid4

thread_id = uuid4().hex
await runtime.send_message("assistant", thread_id, "Hello")
```

Thread IDs are strings because they cross Temporal and persistence boundaries.
Generate them from UUIDs (or an equivalently collision-resistant application
scheme) instead of using sequential labels in production.

`send_message` uses Temporal signal-with-start. The first message starts the
thread workflow; later messages signal that same workflow. Messages arriving
while a run is active remain in the workflow inbox and are drained at the next
run boundary.

The call returns after delivery to Temporal. Observe completion through hooks,
projections, or your application's event API rather than holding the request
open for the entire agent run.

## Inspect and cancel

```python
state = await runtime.get_state("assistant", thread_id)
await runtime.cancel_thread("assistant", thread_id)
```

The query returns live workflow state such as inbox size, total turns, current
run ID, and cancellation state. Projection stores provide richer readable
history. Cancellation is durable and projection cleanup is idempotent.

## Resolve deferred tools

```python
await runtime.resolve_tool_call(
    "assistant",
    thread_id,
    tool_call_id,
    approved=True,
    answer="Approved",
)
```

Resolution durably signals the owning thread workflow. See
[pauses and deferred work](pauses-and-resume.md) for the full lifecycle.

## Workflow lifecycle

For each run, `AgentThreadWorkflow`:

1. drains all currently queued inbound messages;
2. records a run through an activity;
3. executes one agent turn through an activity;
4. admits all emitted tool calls in parallel;
5. executes allowed calls and awaits deferred calls concurrently;
6. finalizes the tool-result group in transcript order;
7. repeats until completion, exhaustion, failure, or cancellation;
8. parks until another message arrives.

At a run boundary, sufficiently long workflow histories use Temporal
continue-as-new. Queued inbox messages are carried into the new execution.

## Hooks and streaming

`AgentThreadHooks` reports persisted lifecycle events. `StreamListener` reports
low-latency model deltas. Supply factories because each thread receives its own
hook/listener instance.

Good hook responsibilities include publishing SSE/websocket events, updating
product status, and emitting audit telemetry. Do not persist duplicate runtime
messages from hooks: the runtime stores are already the transcript writer.

Use `RunCompletionHandler` for correctness-bearing work that must retry after a
run projection commits, such as resolving the parent of a completed subagent.
Pass it to `TemporalRuntimeWorker`; unlike hooks, handler failure keeps the
finalization activity incomplete and eligible for retry. Handlers must be
idempotent.

## Production checklist

- Use durable projection stores shared by all workers.
- Keep client and worker Temporal configuration identical.
- Make tool side effects idempotent where retries or operator actions matter.
- Choose external-resolution timeouts from product requirements.
- Rebuild or persist application-owned subthread registries.
- Reconnect UIs from projections, then resume live event consumption.
- Test cancellation and stale deferred resolution, not only happy paths.
- Pin provider SDK ranges and configure model IDs outside library code.

For a complete application composition, see the [demo server](../examples/demo/server/)
and [coordinator guide](coordinator-guide.md).
