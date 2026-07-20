# Actant Demo Server

Minimal FastAPI + SSE server demonstrating the Actant runtime. Wraps
`AgentRuntime` and an in-process `TemporalRuntimeWorker` behind a small
HTTP/SSE API.

## Prerequisites

1. A local Temporal cluster (from the repo root:
   `actant server start --detach --port 27233 --ui-port 28233`).
2. Optionally, an LLM API key in the environment:

   ```bash
   export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY / GEMINI_API_KEY
   export ACTANT_MODEL=...        # model id supported by that provider
   ```

Without a key, the server uses a deterministic local model that exercises
streaming, deferred approvals, multiple-choice questions, and nested subagents.

## Run

```bash
cd examples/demo/server
uv sync
uv run python -m uvicorn app.main:app --port 8181 --reload
```

Open `http://localhost:8181/docs` for the OpenAPI page.

## API

| Method | Path | Body | Purpose |
|---|---|---|---|
| `GET` | `/api/agent` | — | Single agent metadata (name, model, tools). |
| `POST` | `/api/threads/{thread_id}/messages` | `{"content": str}` | Append a user message; runs idempotently per `(agent, thread)`. |
| `GET` | `/api/threads/{thread_id}/events` | — | SSE stream of raw runtime hook events. |
| `GET` | `/api/threads/{thread_id}/state` | — | Current thread state snapshot. |
| `DELETE` | `/api/threads/{thread_id}` | — | Cancel an in-flight thread. |
| `POST` | `/api/threads/{thread_id}/tool_calls/{tool_call_id}/resolve` | `{"approved": bool, "answer": str, "payload": object}` | Resolve a deferred (WAIT) tool call. |

## SSE event types

The stream emits the raw hook events from `PublishingThreadHooks`. Each frame
has `event: <type>` and `data: <json>` where the JSON is
`{"type", "thread_id", "data"}`. Event types:

- `assistant_message` — finalised assistant message (content + tool_calls).
- `turn_start` — a new turn begins.
- `tool_call` — assistant requested a tool.
- `tool_result` — tool returned a result.
- `tool_waiting` — tool admission deferred; resolve via the POST above.
- `tool_resolved` — a previously waiting tool completed.
- `complete` — run finished (success or otherwise).
- `error` — runtime error.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` | — | Pick which provider to use; first one set wins. |
| `ACTANT_PROVIDER` | auto | Force `fake`, `anthropic`, `openai`, or `gemini`. |
| `ACTANT_MODEL` | required for real providers | Model id supported by the selected provider. |
| `ACTANT_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowlist for the demo UI dev server. |
