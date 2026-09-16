# Unified runtime migration (unreleased)

This branch changes Actant's Python API and sandbox image protocol. It is not a release.
Existing consumers should keep their published pins until their migration branches pass
integration tests against this exact candidate commit. No database migration is required.

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

s3 = boto3.client(
    "s3",
    endpoint_url=public_storage_endpoint,
    config=Config(
        connect_timeout=5,
        read_timeout=15,
        retries={"max_attempts": 2},
        signature_version="s3v4",
    ),
)
assets = S3AssetResolver(s3, bucket="bucket", prefix="images/", url_ttl_s=3600)
runtime = AgentRuntime(client=client, stores=stores, resolve_agent=resolve_agent, assets=assets)
```

Install `actant[s3]` to obtain boto3. The adapter uses the SDK's signing implementation,
checks object existence before signing, and caches URLs while their remaining lifetime
covers the model activity plus a refresh margin. Credentials must remain valid for that
lifetime; the application owns temporary-credential renewal. Per-user authorization belongs
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

Observer errors are logged and isolated from execution. Existing hook/listener factories
remain observational integration points, wrapped to isolate exceptions. They must stay fast;
blocking callbacks can still consume an activity's time budget. Cancellation propagates. Temporal-level tool failures drain sibling activities, then fail the
run and repair incomplete tool results without repeating uncertain side effects.

Durable product effects belong in `run_completion_handler`, not a live event handler.
Completion failures retry after finalization. Consumers must deduplicate by `run_id`;
Actant does not promise exactly-once external delivery. Subagent notification should read
persisted parent links, as the demo does, rather than depend on the spawning process.

`InMemorySessionStore` and the unused `SessionStore` interface are removed. Use runtime
stores. `message_to_parts` and `parts_to_messages` remain, along with existing Postgres models.

## Downstream checklist

- Ume gateway: remove private nested-client assignment and empty `agents` mapping.
- Ume workers: use `AgentRuntime.run_worker()`; replace activity re-decoration with resolver.
- Ume media: consolidate preprocessing into an asset resolver and update image display.
- Ume events: adapt scoped events while preserving UI and billing contracts.
- Spaceform: handle `AssetSource` in direct runner responses; resolve for models or local files.
- Roomform: update hosted adapters only where affected; leave core geometry pipeline unchanged.
- Pin all candidate consumers to the exact Actant revision and run cross-repository tests.

## Cutover and rollback

The new start-run activity result carries the resolved turn budget. Existing Temporal
histories are not promised replay compatibility. Do not put new workers onto old active
executions. After separate release authorization, upgrade readers first, pause submissions,
drain old work (including approval waits), then switch workers and writers. Cancellation of
pending work is an explicit product decision, not an automatic migration step.

Historical DB rows remain. A rollback must retain readers that understand newly written
asset references; reverting binaries solely because the schema is unchanged is insufficient.
This draft PR does not authorize merge, release, deployment, or production data changes.
