# Actant Demo UI

Threaded chat surface for the demo server. Vite + React + Tailwind v4,
with no authentication. Runtime projections are persisted by the demo server.

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

## Verification

```bash
bun run test
bun run build
ACTANT_DEMO_UI_URL=http://localhost:5173 bun run test:e2e
```

The Playwright test drives the real UI through streaming, approval resolution,
multiple-choice resolution, nested subagent delegation, and history reload.

## What's out of scope

The demo is intentionally narrow: one local development app focused on
threaded chat, tool calls, deferred user input, and nested sub-agent
transcripts.

When the server is running without an API key, try these deterministic prompts:

- `Show me an approval`
- `Ask me a question`
- `Delegate this to a subagent`
