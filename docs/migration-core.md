# Unified runtime migration (0.17.0)

Actant 0.17.0 changed the Python runtime API and sandbox image protocol without a
DB migration. The steps below describe that released migration. The subsequent event
cleanup is documented separately below and requires coordinated consumer updates.

## One runtime

`AgentRuntime` replaces both `TemporalRuntimeClient` and `TemporalRuntimeWorker`.
There are no compatibility aliases. Pass an existing Temporal `Client` to the public
`client` parameter. The caller owns that connection, including TLS, credentials,
interceptors, and namespace. `config` controls task queue and lifecycle policy.

```python
from temporalio.client import Client
from actant.runtime import AgentRuntime, TemporalRuntimeConfig

config = TemporalRuntimeConfig(task_queue="agents")
client = await Client.connect("localhost:7233")
runtime = AgentRuntime(client=client, stores=stores, config=config)
await runtime.thread("assistant", thread_id).send("hello")
```

A process that executes work supplies an asynchronous resolver and calls `run_worker()`.
The same instance can also submit messages. There is no separate executor or worker facade.

For graceful process shutdown, set `config.graceful_shutdown_timeout_seconds` and call
`await runtime.shutdown()` while `run_worker()` is running. Both return after active
activities finish or acknowledge cancellation after the grace period. This stops polling;
it does not drain entire workflows or close caller-owned resources.

```python
from actant import AgentDefinition


async def resolve_agent(agent_id: str, thread_id: str) -> AgentDefinition:
    context = await application.load_context(thread_id)
    return application.build_agent(agent_id, context)


runtime = AgentRuntime(
    client=client,
    stores=stores,
    config=config,
    resolve_agent=resolve_agent,
    event_sink=events,
    turn_gate=check_credit,
    run_completion_handler=on_complete,
)
await runtime.run_worker()
```

The resolver may be called on any worker and more than once per run. Reuse provider clients
and service runners in application-owned resources; do not create unclosed clients on each
resolution. Resolution must work from persisted context rather than a process-local map of
previously started threads. Returning a different agent ID is an error.

`AgentDefinition.max_turns_per_thread` determines a new run's budget at execution time.
`TemporalRuntimeConfig.max_turns_per_run`, when supplied, is an additional ceiling;
its default is `None`. A gateway needs no definitions. Existing thread command methods,
`ThreadHandle`, DB models, and public DB exports remain available.

Injected dependencies are caller-owned. Runtime worker shutdown does not close shared
connections, sandbox registries, or service runners. `run_worker()` validates its resolver
before polling and rejects simultaneous polling on the same instance.

## Explicit sandbox dependencies

Replace `sandbox_providers=...` with a `SandboxRegistry` passed as `sandboxes`.
There is no implicitly created local provider.

```python
from pathlib import Path
from actant.sandbox import LocalSandboxProvider
from actant.sandbox.registry import SandboxRegistry

sandboxes = SandboxRegistry(
    {"local": LocalSandboxProvider(Path("/srv/workspaces"))},
    stores.threads,
)
```

Local and Modal remain tool execution locations. Temporal remains the only agent executor.
The sandbox claim, per-thread single-flight behavior, and HTTP deadlines are unchanged.

## Durable image references

Uploaded service images now return `AssetSource(storage_key=...)`. The key is a complete
`s3://bucket/prefix/object` reference so the bucket is not lost at persistence. Application
attachments can retain their existing opaque keys. `image_block(image)` serializes:

```python
{"type": "asset", "storage_key": "s3://bucket/images/t/hash.png", "mime": "image/png"}
```

Replace `SandboxSpec.image_url_ttl_s=None` with `upload_images=False`. Otherwise uploads are
enabled when the provider has image storage. Remove `public_endpoint_url` from
`ModalSandboxProvider` and `ImageBucket`; signing configuration belongs to the resolver.
`ImageUploadConfig` no longer accepts signing endpoint or TTL. Upload failure still returns
inline bytes and a storage diagnostic. Storage retention remains application-owned.

Implement `AssetResolver.resolve(AssetReference, AssetContext)` or use the optional S3 adapter:

```python
import boto3
from botocore.config import Config
from actant.storage.s3 import S3AssetResolver
from actant.storage.sigv4 import SigningKeys

s3 = boto3.client(
    "s3",
    endpoint_url=public_storage_endpoint,
    config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2}),
)
assets = S3AssetResolver(
    s3,
    endpoint_url=public_storage_endpoint,
    region=region,
    keys=SigningKeys(access_key_id, secret_access_key),
    bucket="bucket",
    prefix="images/",
)
runtime = AgentRuntime(client=client, stores=stores, resolve_agent=resolve_agent, assets=assets)
```

Install `actant[s3]` to obtain boto3, used only to check that an object exists. URLs are
signed at the start of a fixed window (`window_s`, 6 h) and live `window_s + buffer_s`, so every
process sends the same URL for an object within a window. Nothing is stored. Use static
credentials: with a session token the URL changes whenever the token rotates. Per-user authorization belongs
in the application's resolver before delegation to the bucket/prefix-scoped S3 adapter.

Model preparation never mutates persisted history. It preserves message metadata and handles:

| Stored content | Behavior |
|---|---|
| Asset reference | Resolve to URL or inline image |
| Inline image | Preserve bytes |
| Live legacy URL | Strip internal metadata before provider conversion |
| Expired URL with explicit storage key | Resolve using that key |
| Expired URL without key | Visible unavailable-image note |
| Missing object | Visible unavailable-image note |
| Permission or transport failure | Fail explicitly; never substitute a missing-image note |

No resolver for an asset reference is a configuration error. URLs must cover the model
activity's 600-second budget; the S3 adapter also keeps a 120-second refresh margin.
The gateway's display path and direct sandbox-runner consumers must resolve references too.
Do not rewrite old transcripts or infer object keys from arbitrary URLs.

## Events and completion

Default runtime events carry `agent_id`, `run_id`, and, for turn/tool events, `turn_id` in
`data`; identities come from the activity payload. Existing event names and content fields
remain. An application SSE adapter can keep its external wire contract unchanged.

Live event publication is observational. Event sink errors are logged and isolated from
execution; cancellation propagates. Keep sinks fast: blocking publication still consumes
activity time. Temporal-level tool failures drain sibling activities, then fail the run
and repair incomplete tool results without repeating uncertain side effects.

Durable product effects belong in `run_completion_handler`, not a live event handler.
Completion failures retry after finalization. Consumers must deduplicate by `run_id`;
Actant does not promise exactly-once external delivery. Subagent notification should read
persisted parent links, as the demo does, rather than depend on the spawning process.

`InMemorySessionStore` and the unused `SessionStore` interface are removed. Use runtime
stores. `message_to_parts` and `parts_to_messages` remain, along with existing Postgres models.

Custom `ToolCallStore` adapters must make `update_status` an atomic conditional write: return
`True` when updating a nonterminal call, `False` when the call is already completed, blocked,
or failed, and preserve that terminal status and result. Missing IDs still raise `KeyError`.
The built-in stores implement this without a schema change. This fences late writes from
workers that survive a Temporal timeout; it does not undo external tool side effects.

## Downstream checklist

- Ume gateway: remove private nested-client assignment and empty `agents` mapping.
- Ume workers: use `AgentRuntime.run_worker()`; replace activity re-decoration with resolver.
- Ume media: consolidate preprocessing into an asset resolver and update image display.
- Ume events: adapt scoped events while preserving UI and billing contracts.
- Spaceform: handle `AssetSource` in direct runner responses; resolve for models or local files.
- Roomform: update hosted adapters only where affected; leave core geometry pipeline unchanged.
- Pin consumers to the exact Actant revision and run cross-repository tests.

## Event cleanup after 0.17.0

- Remove `hooks_factory` and `listener_factory`; supply one `event_sink` adapter.
- Pass `event_source` explicitly when using `ThreadHandle.events()`. Stores no longer
  implicitly supply either event dependency.
- `AgentThreadHooks`, the publishing hook/listener classes, and
  `actant.runtime.coordinator` are removed. Provider `StreamListener` remains.
- Read turn identity from each event instead of maintaining shared current-turn state.
- Tool events carry a structured `result`, including content blocks and metadata.
- Keep PDF/non-image preprocessing, execution gates, and durable completion callbacks.
  None is replaced by best-effort event publication.

See the [coordinator guide](coordinator-guide.md) for event and completion contracts.
No schema migration or activity name/input/result change is introduced by this cleanup.

## Cutover and rollback

For upgrades from versions before 0.17.0, active Temporal histories are not promised replay
compatibility: the start-run activity result changed. Pause submissions and drain old
executions, including children and approval waits, on old workers before switching.
A worker's graceful shutdown period alone is not proof that all workflows drained.
Deploy compatible image readers and workers together, smoke-test, then reopen submissions.

For the later event cleanup, migrate consumers before updating dependency pins: factories
and coordinator imports have no compatibility aliases. Validate UI events and completion
contracts against the pinned candidate before deployment.

Historical DB rows remain. Rollback must retain readers that understand durable asset
references. Schema compatibility alone does not establish workflow or image compatibility.
