# Actant Demo UI

Single-thread chat surface for the demo server. Vite + React + Tailwind v4,
no auth, no routing, no persistence beyond the server's in-memory stores.

Consumes the demo server's raw SSE event stream (`text_delta`,
`tool_call_start`, `assistant_message`, `tool_result`, …) and folds them into
console entries (user / assistant / event). The hook in
`src/chat/useAgentConsole.ts` is the reference implementation for SDK
consumers — it covers streaming text, streaming tool-call args, final
`assistant_message` reconciliation, and tool-result rendering.

## Prerequisites

The demo server is running on `http://localhost:8181` (see
`examples/demo/server/README.md`).

## Run

```bash
cd examples/demo/ui
bun install   # or: npm install / pnpm install
bun run dev   # http://localhost:5173
```

Set `VITE_ACTANT_API_BASE` to point at a different server origin.

## What's out of scope

The demo is intentionally narrow: one local development app focused on
threaded chat, tool calls, deferred user input, and nested sub-agent
transcripts.
