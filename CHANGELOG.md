# Changelog

Notable user-facing changes to Actant are recorded here. Internal refactors,
tests, and documentation-only edits may be omitted unless they materially
affect users.

## Unreleased

No user-facing changes yet.

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

[Unreleased]: https://github.com/johnathanchiu/actant/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/johnathanchiu/actant/releases/tag/v0.1.0
