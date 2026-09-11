# Changelog

Notable user-facing changes to Actant are recorded here. Internal refactors,
tests, and documentation-only edits may be omitted unless they materially
affect users.

## 0.8.1

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
