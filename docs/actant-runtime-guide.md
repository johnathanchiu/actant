# Actant runtime guide

Actant uses one Temporal workflow for each `(agent_id, thread_id)`. Client code
signals workflows; worker code hosts the model, tools, stores, and activities.

Read [core concepts](concepts.md) first if thread, run, and turn are not yet
familiar.

## Sandboxes and artifacts

Tools that run code get a per-thread sandbox from the worker (see the tools
guide). Wire the backends and the artifact sink on the worker:

```python
from pathlib import Path
from actant.runtime import AgentRuntime
from actant.sandbox.local import LocalSandboxProvider
from actant.sandbox.registry import SandboxRegistry

runtime = AgentRuntime(
    client=client,
    stores=stores,
    resolve_agent=resolve_agent,
    sandboxes=SandboxRegistry(
        {"local": LocalSandboxProvider(Path("/srv/agent-workspaces"))}, stores.threads
    ),
    artifact_sink=sink,
)
```

Sandbox providers are explicit, caller-owned dependencies. `actant[modal]` adds
`ModalSandboxProvider`. Tool execution location does not change the Temporal runtime.

Run `alembic upgrade actant@head` to apply Actant's schema.

## Install

Install the provider-neutral runtime plus the model SDKs your worker uses:

```bash
pip install actant
pip install "actant[openai]"  # choose only the provider extras you need
```

Actant does not select a “latest” model. Pass a model ID from application
configuration to the corresponding provider adapter.

### Bounded model calls and final turns

`OpenAIProvider(idle_s=60, reasoning_idle_s=180, turn_s=240, attempts=3)` bounds a
whole completion, including retries, backoff, and rate-limiter waits, to `turn_s`
seconds. Opening the stream, and any silence while a message or tool call is
streaming, has an `idle_s` deadline. A reasoning model emits no events while it
thinks, so silence outside an open output item has the longer `reasoning_idle_s`
deadline. Only transient failures retry within the budget: timeouts, connection
errors, 408/409/429/5xx, `server_error` or `rate_limit_exceeded` failures, a stream
that closes before its terminal event, and tool-argument whitespace loops. An
`incomplete` response (max output tokens, content filter) or any other failure
raises at once. The SDK's own retries are disabled, and each attempt takes its own
rate-limiter reservation. Only a completed attempt produces the canonical assistant
message and usage record; before a retry the listener receives `on_stream_reset()`
(published as `stream_reset`), and consumers drop the deltas they have shown.

Set `AgentDefinition.final_tools=("edit", "finish")` to constrain the last turn
of each run. The full tool schema is retained, with a native OpenAI
`allowed_tools` restriction and a short final-turn note. The
names must be registered and allowed, and the client must declare
`supports_allowed_tools = True`; otherwise the definition raises `ValueError`.
Only OpenAI supports it today.
See [OpenAI tool choice](https://developers.openai.com/api/docs/guides/function-calling#tool-choice).

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
Temporal service through an injected Temporal `Client`. `TemporalRuntimeConfig` controls queue and lifecycle policy.

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


async def resolve_agent(agent_id: str, thread_id: str) -> AgentDefinition:
    return agents[agent_id]
```

Use an explicit application setting for `model_id`. Provider model catalogs
change independently of Actant releases.

## Create the runtime

```python
from actant.runtime import AgentRuntime, TemporalRuntimeConfig
from actant.runtime.stores import InMemoryRuntimeStores

stores = InMemoryRuntimeStores()
config = TemporalRuntimeConfig(
    address="localhost:7233",
    namespace="default",
    task_queue="actant-runtime",
)
from temporalio.client import Client

client = await Client.connect(config.address, namespace=config.namespace)
runtime = AgentRuntime(client=client, stores=stores, config=config)
```

In-memory stores are suitable for tests and local examples. Use the included
SQLAlchemy Postgres stores, or implement the store protocols, when projections
must survive process restarts.

## Migrating the schema

Actant owns the five tables its SQLAlchemy stores read and write, so it ships
their migrations: an Alembic branch labelled `actant`, whose revision files
live in the installed package. Upgrading Actant brings the DDL with it, rather
than leaving the application to notice that a dependency changed shape.

An application keeps its own revisions on its own branch and embeds Actant's
as a second `version_locations` entry. Alembic resolves that while building
its `ScriptDirectory` -- before `env.py` is imported -- so set it on the config
before invoking a command:

```python
import os

from alembic import command
from alembic.config import Config
from actant.migrations import versions_path

config = Config("alembic.ini")
config.set_main_option(
    "version_locations", os.pathsep.join([local_versions, str(versions_path())])
)
config.set_main_option("path_separator", "os")

command.upgrade(config, "heads")
```

Note `heads`, plural. With two branches there is more than one head, and
`head` raises rather than picking one.

`path_separator = "os"` is what Alembic 1.16 and later call the option;
before that it was `version_path_separator`. Joining on `os.pathsep` rather
than a space matters for the same reason -- a space-separated value splits
apart if either path contains a space.

New application revisions need `--head` to say which branch they extend:

```
alembic revision --autogenerate --head app@head -m "a name"
```

Without it, Alembic does not guess and does not misplace the file -- it
aborts with "Multiple heads are present". `--version-path` is optional once
`--head` is given, since Alembic defaults it to the directory holding the
resolved head.

Leave Actant's tables out of the application's own `target_metadata`, and
exclude them from its autogenerate with `include_object`. A table present in
the database but absent from the metadata reads as "drop this table", and a
table in both branches gets migrated twice.

**Adopting this on a database that already has the tables** -- created by
`create_all`, or by the application's own migrations before Actant shipped
these -- means stamping the branch once rather than running it:

```python
command.stamp(config, "actant@head")
```

That records it as applied without re-issuing the DDL. Later Actant revisions
apply normally.

## Run a worker

Call `run_worker()` on the same `AgentRuntime` type to execute work. An API process
can use it only for commands; an execution process supplies an async resolver:

```python
from actant.runtime import AgentRuntime

runtime = AgentRuntime(
    client=client,
    stores=stores,
    resolve_agent=resolve_agent,
    config=config,
    event_sink=my_event_sink,
    event_source=my_event_source,
)
await runtime.run_worker()
```

Submission and execution can live in one service for local development or separate
processes in production. Every worker that may receive an activity must be able
to resolve the referenced agent definition and access compatible projection
stores.

## Send messages

```python
from uuid import uuid4

thread = runtime.thread("assistant", uuid4())
await thread.send("Hello")
```

Thread IDs are strings because they cross Temporal and persistence boundaries.
The handle accepts a UUID and normalizes it at the boundary. Generate IDs from
UUIDs (or an equivalently collision-resistant application scheme) instead of
using sequential labels in production.

`send` uses Temporal signal-with-start. The first message starts the
thread workflow; later messages signal that same workflow. Messages arriving
while a run is active remain in the workflow inbox and are drained at the next
run boundary. Set `TemporalRuntimeConfig(interleave_inbox=True)` to hand them to
the model on the run's next turn instead: they land after that turn's tool
results, so the model never sees a user message between a tool call and its
result.

The call returns after delivery to Temporal. Observe completion through events,
projections, or your application's event API rather than holding the request
open for the entire agent run.

## Inspect and cancel

```python
state = await thread.state()
messages = await thread.messages()
waiting = await thread.waiting_tools()
await thread.cancel()
```

State reads persisted projections, including total turns, current run, and cancellation.
Inbox size is unknown because a completed workflow may have left Temporal retention. Projection stores provide richer readable
history. Cancellation is durable and projection cleanup is idempotent.

## Resolve deferred tools

```python
await thread.resolve(tool_call_id, approved=True, answer="Approved")
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
8. processes queued messages, or closes when its inbox is empty.

A later message starts a new execution with the same logical thread ID and persisted history.

At a run boundary, sufficiently long workflow histories use Temporal
continue-as-new (`history_size_threshold`). Queued inbox messages are carried
into the new execution. This rotation concerns Temporal's event history only;
the model's request is built from the stores and never changes because of it.
Reducing what the model sees is [context compaction](#context-compaction).

## Hooks and streaming

For application-facing live observation, consume typed thread events:

```python
async for event in thread.events():
    if event.type == "text_delta" and event.text:
        print(event.text, end="", flush=True)
    elif event.type == "tool_waiting":
        print(f"Approval needed: {event.tool_call_id}")
```

`AgentRuntime` reads from an explicit `event_source` and writes both lifecycle and
model-stream events to one explicit `event_sink`. Neither dependency is inferred from
stores. Applications adapt envelopes to their UI and owner/parent routing contracts.

Each event carries immutable activity identity. Use its `run_id`, `turn_id` and
`turn_index` rather than a process-local current-turn cache. Tool events include the
structured `result` for image and artifact adapters. `StreamListener` remains the LLM
provider callback contract; applications no longer construct hook/listener factories.

Events are observational. After reconnect, reload persisted messages and waiting tools
before resuming the stream. Do not duplicate transcript writes in an event adapter.

Use `RunCompletionHandler` for correctness-bearing work that must retry after a
run projection commits, such as resolving the parent of a completed subagent.
Pass it to `AgentRuntime`; unlike live events, handler failure keeps the
finalization activity incomplete and eligible for retry. Handlers must be
idempotent.

## Turn gate

Event sinks observe; they cannot stop a run. Tool admission decides one tool call at
a time, and a denied call still leaves the agent taking turns. To stop a run
before it spends a model call -- an organization is out of credit, a budget or
rate limit is reached -- pass a `TurnGate` to `AgentRuntime`:

```python
from actant.runtime import AgentRuntime, TurnStart


async def check_credit(turn: TurnStart) -> str | None:
    if await billing.balance_for(turn.agent_id, turn.thread_id) <= 0:
        return "credit balance exhausted"
    return None


runtime = AgentRuntime(
    client=client, stores=stores, resolve_agent=resolve_agent, turn_gate=check_credit
)
```

The gate runs in the turn activity before every model call, after the run's
inbound messages are persisted. `TurnStart` carries `agent_id`, `thread_id`,
`run_id`, `turn_id`, and `turn_index`. Returning `None` lets the turn proceed.
Returning a reason ends the run without calling the model: the run finalizes
`exhausted` with the reason as `stop_reason`, and `RunCompletion.stop_reason`
and `on_complete(reason=...)` receive it. The next message starts a new run,
which the gate sees again. The gate is worker configuration and never enters
workflow payloads. An exception it raises fails the turn, and the run finalizes
`failed`.

## Context compaction

Actant never truncates or rewrites a thread's context. The one reducer is a
summarizing compaction, and it is off unless a thread opts in:

```python
from actant.runtime import AgentRuntime, CompactionConfig, TemporalRuntimeConfig

config = TemporalRuntimeConfig(
    context_compaction=CompactionConfig(
        context_window_tokens=1_000_000,
        max_images_per_request=50,  # Azure OpenAI rejects a request with more
        threshold=0.9,
        keep=["brief", "tool:read_checklist"],
    )
)
runtime = AgentRuntime(client=client, stores=stores, config=config, resolve_agent=resolve_agent)
await runtime.thread(agent_id, thread_id).send(brief, tag="brief")
```

Before each model call, the turn activity measures the request it is about to
send: the last turn's reported input and output tokens plus an estimate of what
was added since, and an exact count of its images. When it would pass
`threshold` of `context_window_tokens`, or carry more than
`max_images_per_request` images, nothing is sent or stored. The workflow runs
`compact_context`: one call on the agent's own model, with no tools, over the
context up to the last model reply, answering the built-in prompt (the goal,
decisions and why, what was verified, every open item, next steps, key facts,
and an index of notable images by id). That call carries no more than the last
request did, so it is under the image limit, and its output is capped to the
window's remaining margin. If it fails, the run fails; nothing is dropped.
`compaction_instructions` on `AgentRuntime` appends to the prompt.

The summary is stored as a compaction row in the transcript: a message with
`kind="compaction"` whose content is one `CompactionBlock` (the summary, the
message ids of the kept messages, the reason, and tokens and images before and
after). A `context_compacted` event carries the same. The turn then runs again,
and its new messages are stored after that row. From then on the model sees:

1. the system prompt;
2. the summary, as text;
3. the kept messages, in transcript order: for each tag in `keep`, the latest
   message with that tag, and the turn that was still open (an assistant tool
   call with its results, never split);
4. every message after the compaction row.

Tool results are tagged `tool:<tool name>`; an app tags what it sends with
`send(..., tag=...)`. A kept tool result whose call the summary replaced is
sent as a user message labelled as retained, its content verbatim: a tool
result without its call is rejected by providers, and replaying the old call
out of place (with its siblings and reasoning items) is not safe either.

`messages.list_for_model` reads exactly those rows: the latest compaction row,
then the rows at or after it plus the kept ones; `list_for_thread` remains the full transcript,
compaction rows included. Images before the row are not sent again;
`RecallImageTool(stores.messages)` gives the agent `recall_image(id)` to attach
one again from the asset store by the id the summary lists. Limits left `None`
fall back to attributes of the same names on the model client (`OpenAIProvider`
declares 50 images on an `AsyncAzureOpenAI` client and 1,500 otherwise).

## Production checklist

- Use durable projection stores shared by all workers.
- Embed Actant's migration branch, and upgrade to `heads` on deploy.
- Keep client and worker Temporal configuration identical.
- Make tool side effects idempotent where retries or operator actions matter.
- Choose external-resolution timeouts from product requirements.
- Rebuild or persist application-owned subthread registries.
- Reconnect UIs from projections, then resume live event consumption.
- Test cancellation and stale deferred resolution, not only happy paths.
- Pin provider SDK ranges and configure model IDs outside library code.

For a complete application composition, see the [demo server](../examples/demo/server/)
and [coordinator guide](coordinator-guide.md).
