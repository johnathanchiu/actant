# Changelog

Notable user-facing changes to Actant are recorded here. Internal refactors,
tests, and documentation-only edits may be omitted unless they materially
affect users.

## Unreleased

- Images a service returns can reach the model as presigned URLs instead of bytes
  relayed through the worker. A host whose files reach a bucket (`disk_sync` on Modal,
  or `LocalSandboxProvider(images=ImageBucket(...))`) uploads each image at once under
  `actant-images/<thread>/` and presigns it for `SandboxSpec.image_url_ttl_s` (default 6 h;
  `None` sends bytes). `public_endpoint_url` (on `ModalSandboxProvider` and
  `ImageBucket`) names the host URLs are signed for, when the model provider reaches the
  bucket another way (a tunnel). A failed upload keeps the bytes and sets
  `StorageStatus.image_error`. `actant.tools.image_block(image)` builds the content block.
  Gemini accepts URL image sources. On replay the adapters replace an expired URL image
  with a text note.
- **Breaking:** `Image.data_b64` is now `Image.source`, an `InlineSource(data_b64)` or
  `UrlSource(url, expires_at)` (`ImageSourceKind`). `HostConfig.images`
  (`ImageUploadConfig`). `CallResponse.storage` is also present from a host that uploads
  images.

## 0.12.0

- `SandboxSpec.seed` (`disk_sync`): a key prefix that starts a new thread. When the
  thread's prefix is empty, the sandbox pulls the seed while it is copied into the
  thread's prefix; startup waits for both, then writes a marker object beside the
  prefix. A thread with files ignores its seed; one without the marker (a copy cut off)
  fails startup.
- A `disk_sync` sandbox starts sooner: the services' modules import and the mtime
  listing runs while the restore pulls. Importing a service module must not read the
  restored files.
- **Breaking:** `RestoreConfig.seed` (`SeedConfig`) in the entry config;
  `host.main(config, services=None)` takes already loaded services
  (`host.load_services`).

## 0.11.0

**Breaking:** toolsets are now services. The sandbox host serves plain classes whose
public methods are callable remotely, by a model's tools or by orchestration code
alike. No compatibility aliases; host and client must run the same actant version.

- The generic machinery (runners, `call_host`) lives in `actant.sandbox.service` and
  is exported from `actant.sandbox`; it does not import `actant.llm`, `actant.runtime`
  or `actant.tools`. `actant.tools` keeps only the model adapter: `tools(cls, runner)`
  and `tool_schemas(cls)`.
- Runners return a `CallResponse` (`text`, `images`, `error`, `storage`) instead of a
  `ToolResult`: `Runner.call(method, args, *, key=None, sandbox=None)`, so code calls
  a service without a `CallContext`. `call_host` returns a `CallResponse` too. Tools
  still see a `ToolResult`, with storage status on `metadata["storage"]`.
- Wire and launch messages: `CallRequest.service` and `HostConfig.services`.

| 0.10 | 0.11 |
| --- | --- |
| `SandboxSpec.toolsets` | `SandboxSpec.services` |
| `SandboxSpec.toolset_port` | `SandboxSpec.service_port` |
| `HostConfig.toolsets` | `HostConfig.services` |
| `CallRequest.toolset` (JSON `"toolset"`) | `CallRequest.service` (JSON `"service"`) |
| `actant.tools.toolset` | `actant.sandbox.service` (runners) + `actant.tools.service` (adapter) |
| `actant.tools.LocalRunner`, `RemoteRunner`, `SandboxRunner`, `Runner`, `call_host` | `actant.sandbox.…` (same names) |
| `actant.tools.toolset_schema(cls)` | `actant.tools.tool_schemas(cls)` |
| `host.public_methods(cls)` | `host.service_methods(cls)` |
| `ToolsetTool`, `ToolsetInvocation` | `ServiceTool`, `ServiceInvocation` |
| `Runner.call(method, args, ctx) -> ToolResult` | `Runner.call(method, args, *, key, sandbox) -> CallResponse` |
| `call_host(...) -> ToolResult` | `call_host(...) -> CallResponse` |
| `RemoteRunner(endpoint, toolset, key)`, `SandboxRunner(toolset)` | `RemoteRunner(endpoint, service, key)`, `SandboxRunner(service)` |

## 0.10.0

**Breaking:** `python -m actant.sandbox.entry` takes one `EntryConfig` JSON document
(see below); a sandbox image must carry the same actant version as the worker.

- Storage sync never fails or stalls a run. The toolset host kills a push after
  `SandboxSpec.sync_timeout_s` (default 300) and keeps pushing; it also pushes every
  `sync_interval_s` (default 60) while completed calls are unpushed. The final push
  on shutdown is bounded the same way.
- Push status is visible: call responses carry `storage`, surfaced as
  `ToolResult.metadata["storage"]` (`last_attempt_at`, `last_success_at`,
  `last_error`, `consecutive_failures`, `pending`).
- `disk_sync` restore has a timeout (fails startup) and gives restored files their
  objects' mtimes, so pushes no longer re-upload the whole workspace after a restore.
- `ModalSandbox.sync` and `close` are bounded even when Modal's API hangs, and
  `close` never raises (failures are logged).
- `SandboxSpec` rejects a non-positive `sync_interval_s` or `sync_timeout_s`.
- Typed wire and launch messages in `actant.sandbox.protocol`: host bodies are
  `CallRequest`/`CallResponse` (JSON unchanged; a malformed call body is now HTTP
  400), `StorageStatus` is exported from `actant.sandbox`, routes and headers are
  `Route`/`Header`, and `actant.tools.MetadataKey` names the runtime's metadata keys.
- **Entrypoint command line:** `python -m actant.sandbox.entry` now takes one
  `EntryConfig` JSON document instead of `--restore`/`--` host flags, and
  `host.launch_args`, `host.CALL_PATH` and `host.SHUTDOWN_PATH` are gone. The
  image's actant must match the worker's.

## 0.9.0

**Breaking:** the `Sandbox` protocol gains `sync()` and `endpoint(refresh=)`; a
custom backend must implement both. The `modal` extra requires `modal>=1.5.5`.

- Toolsets (`actant.tools`): a plain class whose public methods are tools, with
  schemas from the signatures. `tools(cls, runner)` runs them through
  `LocalRunner` (in-process), `RemoteRunner`/`call_host` (a host endpoint) or
  `SandboxRunner` (the thread's sandbox). Results encode identically everywhere.
- The toolset host (`actant.sandbox.host`, launched by `python -m actant.sandbox.entry`)
  serves the named `SandboxSpec.toolsets` inside a sandbox over authenticated,
  kept-alive HTTP; `Sandbox.close` shuts it down gracefully (instances closed,
  storage pushed).
- `SandboxSpec` gains `gpu`, `network`, `secrets`, `scrub_env`, `storage`
  (`Storage.MOUNT` or `Storage.DISK_SYNC`), `toolsets` and `toolset_port`;
  `Backend`, `Storage` and `Endpoint` are exported from `actant.sandbox`.
- Modal backend honours the new spec fields; `disk_sync` restores the bucket prefix
  onto local disk with s5cmd (`with_s5cmd(image)`) and pushes it back. The toolset
  host is reached through a cached Modal connect token, so no port is public.

## 0.8.1

- Review fixes: a sandbox is reattached by its persisted id on every call, so one the
  backend reclaimed is reopened instead of served stale; closing a sandbox never raises
  and the id is forgotten first; `execute_tool` heartbeats from before the sandbox opens;
  `ls` cannot leave the sandbox root; a missing artifact sink is a non-terminal failure;
  approval-gated function tools get their `CallContext` on resolve.
- `send_message(..., parent_thread_id=)` / `ThreadInput.parent_thread_id` record a
  sub-thread's parent on the thread row.
- Delegation is one level deep: `CallContext.parent_thread_id` is set for a subagent's
  calls and `TaskTool` refuses to spawn from one.
- Modal backend on Modal 1.5: the legacy `Sandbox.open`/`mkdir` file API was removed by
  Modal, so reads and writes go through `Sandbox.filesystem`; the `modal` extra requires
  `modal>=1.5`. `close` waits for termination so `attach` on a closed sandbox raises.
- `RunStore.list_for_thread` lists a thread's runs, newest first.
- Local backend: the interpreter's directory is first on `PATH`.

## 0.8.0

**Breaking:** `Tool.build` now takes the call: `build(params, ctx: CallContext)`.
Every tool receives who is calling, from which thread and run, and, when it
asked for one, the thread's sandbox. `build_for_call` is gone; `TaskTool`
reads its parent thread from the context. A workflow started before the
deploy is unaffected: no history payload changed shape, and the new
`RunTurnInput`/`TurnResult`/`FinalizeRunInput` fields are defaulted.

### Sandboxes

A tool that runs code declares it (`needs_sandbox = True`, or a `Sandbox`
parameter on a function tool) and the runtime opens one sandbox per thread,
lazily, from the backend named by `AgentDefinition.sandbox`. `local` is a
directory and a subprocess; `modal` (extra `actant[modal]`) is a
network-blocked `modal.Sandbox` over a bucket prefix mounted from the
product's own object storage. The sandbox id is persisted on the thread, so
any worker reattaches. `SandboxRegistry`, `SandboxProvider`, `LocalSandbox`.

### The exit point

`FinishTool`: `finish(summary, paths=[])` ends the run and names the
deliverables. Any terminal tool result may list `metadata["deliverables"]`;
the runtime reads them from the sandbox, stores them through the worker's
`ArtifactSink`, and reports them as `metadata["artifacts"]` and
`RunCompletion.artifacts`.

`AgentDefinition.completion="terminal"`: a task agent's run completes only
on a terminal result; a text-only turn gets one persisted reminder, a second
ends the run as exhausted with `stop_reason="stopped without finishing"`
(`AgentRun.stop_reason`, `RunCompletion.stop_reason`). `"reply"` keeps chat behaviour.

### Runtime

- `execute_tool` heartbeats every 30 s and carries a 2-minute heartbeat
  timeout, so a worker that dies mid-tool is noticed in minutes.
- Migration `0002_sandbox_and_run_reason`: `actant_threads.sandbox_id`,
  `actant_runs.stop_reason`.
- `TemporalRuntimeWorker(sandbox_providers=..., artifact_sink=...)`.

## 0.7.0

**Breaking, and requires draining in-flight workflows before deploy.**

Two payloads that Temporal serializes into workflow history changed shape:
`AdmitDecision`'s string values, and `ThreadInput` (which lost
`exit_when_idle`). A workflow started before the deploy replays the old
values against the new code, and the admission mismatch **fails quietly** --
the call matches no branch, nothing runs, nothing waits, and the transcript
gets `{"error": "No result"}`. Drain first; do not rely on it erroring.

### Subagents are ordinary tools

Delegation no longer parks the parent. `TaskTool` starts the subagent and
returns its thread id, so a parent can start several and supervise them with
the new `check_subagent` / `message_subagent` / `stop_subagent`, over a
`SubagentSupervisor` the host implements.

- `SubagentSpawner.spawn` returns the sub-thread id and no longer takes
  `parent_tool_call_id`.
- `ToolDecision.spawn`, `ToolSpawnRequest` and the `SPAWN` admission kind are
  removed. `SubThreadLink.parent_tool_call_id` and
  `SubThreadRegistry.find_by_parent_tool_call` go with them.
- Tools may define `build_for_call`, discovered structurally like
  `can_execute` and `on_resolve`, for the few that are about the call rather
  than its arguments.

### Threads end when their work is done

`ThreadInput.exit_when_idle` is removed; it is simply what threads do. A
closed thread resumes on its next message with the same id and nothing lost.

- `get_state` reads the stores rather than querying the workflow, and raises
  for a thread that does not exist rather than creating one.
- `inbox_size` is now `int | None`, and is `None` when read from the stores.
- `cancel_thread` tolerates an already-finished thread but raises
  `ThreadNotFoundError` for one that never existed.

### The admission gate says what it decides

`ALLOW` -> `EXECUTE`, `BLOCK` -> `DENY`, `WAIT` -> `AWAIT_HUMAN`, with
`ToolDecision.allow`/`block`/`wait` renamed to match. `BLOCK` read as
"suspend" and was the only one that never suspended.

`ToolCallStatus.WAITING`/`BLOCKED` are a different enum, are persisted, and
are deliberately unchanged.

## 0.5.0 - 2026-08-31

### Added

- Actant now ships the Alembic migrations for the tables it owns, as a branch
  labelled `actant` under `actant/migrations/versions`, with
  `actant.migrations.versions_path()` returning their installed location.
  Previously the package defined the models and left every application to
  notice on its own that upgrading a dependency had changed its schema --
  `create_all` only creates, so an existing database silently kept the old
  shape until something failed on a missing column. Upgrading the package now
  brings its DDL with it.

### Upgrading

Applications embed the branch as a second `version_locations` entry. Alembic
resolves that while building its `ScriptDirectory`, before `env.py` is
imported, so the path has to be set on the config before invoking a command
rather than computed in `env.py`:

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

`heads` rather than `head`: with two branches there is more than one, and
`head` raises rather than choosing. New application revisions likewise need
`--head <label>@head` to say which branch they extend, or Alembic aborts with
"Multiple heads are present".

**On a database that already has these tables** -- created by `create_all` or
by the application's own migrations -- stamp the branch instead of running it,
once:

```python
command.stamp(config, "actant@head")
```

That records it as applied without re-issuing the DDL. Later Actant revisions
then apply normally. An application that had been migrating these tables on
its own branch should also stop doing so: drop them from its `target_metadata`
and exclude them from autogenerate, or the two branches will both try.

## 0.4.0 - 2026-08-31

### Added

- Added provider-reported token usage to `Message` as `input_tokens` /
  `output_tokens`, with a `total_tokens` property. The OpenAI and Anthropic
  providers already read these off the response for rate limiting; they now
  survive to the caller and are persisted on the message header row, so the
  transcript is itself the usage record. `None` means the provider reported
  nothing, and is deliberately distinct from a reported `0` — a caller
  billing on these must treat unknown as unknown, not as free. The Gemini
  and Qwen providers do not report usage yet and leave both fields `None`.

### Upgrading

The `actant_messages` table gains two nullable columns. `create_all` handles
a fresh database; an existing one needs:

```sql
ALTER TABLE actant_messages ADD COLUMN input_tokens INTEGER;
ALTER TABLE actant_messages ADD COLUMN output_tokens INTEGER;
```

Rows written before the upgrade keep `NULL` usage, which reads back as "not
reported". Nothing else changes: the new fields are optional everywhere, and
a message without usage serializes exactly as it did in 0.3.2.

## 0.3.2 - 2026-07-21

### Changed

- Added validated approval prompt templates such as
  `@tool(approval="Publish {title}?")`, keeping callable approval policies as
  an advanced escape hatch.

## 0.3.1 - 2026-07-21

### Changed

- Updated the main quickstart to use thread-scoped typed events and automatic
  worker publishing instead of custom hook and stream-listener classes.
- Made detached local-server startup wait until Temporal's `default` namespace
  is ready, preventing immediate clients from racing container initialization.

## 0.3.0 - 2026-07-21

### Added

- Added a thread-scoped runtime handle with commands, projection reads, and
  typed live events.
- Added `@tool` and `FunctionTool` for annotated sync and async functions,
  including generated JSON schemas and validated arguments.
- Added concise approval, admission, and deferred-resolution callbacks on
  function tools while preserving class-based tools for advanced behavior.
- Added import-cycle regression coverage and automatic worker event
  publishing through explicit event sink/source protocols.

### Changed

- Replaced deferred async-activity completion with durable workflow signals
  and conditions. Human waits now require no application polling or waiting
  activity, and only the workflow may advance an agent run.
- Renamed the public resolution command to `AgentRuntime.resolve_tool_call`.
- Simplified tool-call projections by removing Temporal activity routing IDs.
- Clarified the workflow structure as thread lifecycle → agent run → agent
  turn → tool-group barrier, and preserved thread turn counts across
  `continue_as_new` history rotation.
- Added typed not-found and not-waiting errors for invalid tool resolutions.

## 0.2.0 - 2026-07-20

### Added

- Installed `actant server` CLI for starting, inspecting, logging, stopping,
  and resetting a Docker-backed local Temporal development server.
- Live demo flow for independently approved parallel tools, durable
  cancellation, and continuation on the same thread.

### Changed

- Consolidated repository Temporal recipes behind `just server <command>`;
  detached execution is now the `--detach` flag on `server start`.
- Simplified the README and expanded the architecture, runtime, tool,
  subagent, and release documentation.
- Deferred approval rendering now follows the tool's declared wait kind
  instead of requiring a special tool name.

### Security

- Added regression coverage requiring OpenAI Responses API requests to use
  `store: false`.

## 0.1.0 - 2026-07-20

### Added

- Temporal-native, long-lived agent threads with durable inboxes.
- Explicit `ALLOW`, `BLOCK`, and `WAIT` tool admission.
- Parallel tool execution with a deterministic group barrier.
- Durable external resolution for approvals and other deferred work.
- Nested subagent delegation with parent-facing completion propagation.
- In-memory and SQLAlchemy/Postgres projection stores.
- Optional OpenAI, Anthropic, Gemini, and Qwen provider adapters.
- Deterministic FastAPI and React demo with streaming and approval flows.

[0.3.2]: https://github.com/johnathanchiu/actant/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/johnathanchiu/actant/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/johnathanchiu/actant/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/johnathanchiu/actant/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/johnathanchiu/actant/releases/tag/v0.1.0
