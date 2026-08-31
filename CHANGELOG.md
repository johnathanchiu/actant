# Changelog

Notable user-facing changes to Actant are recorded here. Internal refactors,
tests, and documentation-only edits may be omitted unless they materially
affect users.

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
