# Actant Examples

Examples are runnable compositions of Actant primitives. They show the
app-owned pieces a real project provides: agents, tools, stores, admission
checks, wait resolution, UI, and provider wiring.

## Demo App

`examples/demo/` contains a FastAPI + React demo with a main agent, nested
subagents, deferred user input, approval prompts, and live SSE streaming.

From the repository root:

```bash
just demo-sync
just demo
```
