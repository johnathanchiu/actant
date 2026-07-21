# Changelog

Notable user-facing changes to Actant are recorded here. Internal refactors,
tests, and documentation-only edits may be omitted unless they materially
affect users.

## Unreleased

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

[Unreleased]: https://github.com/johnathanchiu/actant/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/johnathanchiu/actant/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/johnathanchiu/actant/releases/tag/v0.1.0
