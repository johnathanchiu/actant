# Deferred cleanup

Recorded 2026-09-16. Planning only; do not pursue during the current release work.

## Collapse the runtime client layers

- [ ] Inventory all direct consumers of `TemporalRuntimeClient`, including Ume and examples.
- [ ] Keep `AgentRuntime` as the public API and move the Temporal client implementation into it. Actant is Temporal-only; remove the redundant forwarding class rather than adding an alias or compatibility shim.
- [ ] Keep `ThreadHandle` as convenient thread-scoped access to that single client.
- [ ] Update direct consumers, imports, documentation, and tests explicitly. Remove constructor options that have no role on the client side after checking their consumers.
- [ ] Preserve workflow IDs, signals, activity contracts, database schemas, transcript semantics, cancellation, and sandbox behavior. This is not an agent-loop or orchestration redesign.
- [ ] Verify the full Actant suite, lint, type checks, and Ume's local Temporal scene-loop integration test. Check that existing histories still replay before shipping.
- [ ] Submit a separate PR. Do not bundle this with Roomform/Spaceform pipeline consolidation or deploy it as part of tonight's fixes.

## Audit overlapping session interfaces separately

- [ ] Check actual consumers of `SessionStore` and `InMemorySessionStore` versus runtime `MessageStore` before deciding whether to consolidate them.
- [ ] Preserve the message/parts serialization helpers used by Postgres; do not mistake an overlapping public interface for unused serialization code.

## Explicit non-goals

- Keep sandbox hosting and execution in Actant. Local/remote tool execution through the same interface is a useful capability, not a subsystem to remove merely because it is large.
- Do not split packages, rewrite the runtime, or remove provider behavior based on line counts.

## Related consumer work, owned outside Actant

- Roomform and Spaceform should own their complete pipelines. Ume should invoke typed package entry points and supply resources and progress, usage, and artifact callbacks.
- Define checkpoint/resume and cancellation contracts; callbacks alone do not provide durable execution.
- Consolidate CLI and hosted execution and add parity tests as separate follow-up work. Shared scheduling constants are not full runner consolidation.
- Export canonical conversation/tool history to package artifacts for debugging and explicit resume.
