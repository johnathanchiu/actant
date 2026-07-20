# Actant documentation

Actant is a Temporal-native runtime for persistent agents. Start with the
concepts before copying the production wiring: the important distinction is
that Actant models a durable thread, not a single model invocation.

## Start here

1. [Core concepts](concepts.md) — the vocabulary and ownership boundaries.
2. [Runtime architecture](architecture.md) — the exact workflow algorithm,
   activity contracts, state ownership, and recommended code-reading order.
3. [Runtime guide](actant-runtime-guide.md) — connect a client and run a worker.
4. [Tools guide](tools-guide.md) — tool schemas, execution, and admission.
5. [Pauses and deferred work](pauses-and-resume.md) — approval and durable wait
   semantics.
6. [Subagents](subagents.md) — synchronous and durable delegation.
7. [Coordinator guide](coordinator-guide.md) — application policy for
   multi-agent products.
8. [Release checklist](releasing.md) — validate, version, and publish the wheel
   through GitHub's trusted PyPI workflow.

## Learn from a running system

The [demo](../examples/demo/) combines FastAPI, Postgres, Temporal, and a React
viewer. Its deterministic model exercises streaming, approvals,
multiple-choice input, and nested delegation without an API key.

```bash
just demo-sync
just demo
```

Open `http://localhost:5173` after the services start.

## Documentation boundaries

- These guides describe Actant's public runtime concepts and extension points.
- The Python modules remain the source of truth for exact signatures while the
  package is pre-1.0.
- Product concerns such as authentication, tenancy, HTTP APIs, artifact
  storage, and UI state deliberately remain outside the kernel.
