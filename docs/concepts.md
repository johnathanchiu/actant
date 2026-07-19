# Core concepts

Actant treats an agent as a long-lived, addressable service. A request does not
have to finish inside one HTTP connection or one Python process. It can call
tools, wait for a person or another agent, resume later, and accept another
message on the same thread.

## The object model

### Agent definition

An `AgentDefinition` is immutable runtime configuration: identity, persona,
LLM client, tool registry, tool allowlist, context policy, memory namespace,
and turn budget. It describes how an agent behaves; it is not the agent's
conversation state.

Applications keep the definition registered anywhere a Temporal worker may
execute its activities. The client process and worker process may be separate.

### Thread

A thread is the durable address `(agent_id, thread_id)`. One
`AgentThreadWorkflow` owns that address, serializes its work, and remains alive
between runs. New messages are signals to the same workflow.

The thread is the right unit for cancellation, live state queries, conversation
history, and parent/child relationships.

### Run

A run begins when an idle thread drains one or more queued messages. It ends
when the model returns without tool calls, reaches its turn budget, fails, or
is cancelled. A later message starts another run on the same thread.

### Turn

A turn is one model call. It can produce text, thinking summaries, and a group
of tool calls. If tools return results, Actant appends those results to the
transcript and begins the next turn.

### Tool call

A tool call has a persisted identity and lifecycle. Before execution, admission
classifies it as:

- `ALLOW`: execute now.
- `BLOCK`: record a failed result and do not execute.
- `WAIT`: persist the wait request and suspend until an external resolution.

Calls emitted by the same turn are admitted in parallel. Allowed and waiting
calls are then processed concurrently, and their results are finalized as one
tool group before the next model turn.

### Projection stores

Temporal is the coordination source of truth. Runtime stores are readable
projections of threads, runs, messages, tool calls, and memory. They make APIs,
UIs, audits, and recovery practical without replaying a workflow history for
every read.

The runtime writes conversation projections. Hooks announce events; they should
not write duplicate messages.

## What Temporal does

Temporal provides durable workflow state, signals, activity scheduling,
timeouts, cancellation, retry boundaries, and replay. Actant maps those
primitives onto an agent lifecycle:

```text
send_message
    -> signal-with-start thread workflow
    -> start run activity
    -> model turn activity
    -> admit tool activities
    -> execute or durably await resolution
    -> finalize tool results
    -> next turn or park thread
```

The workflow contains only serializable orchestration state. Model calls,
database writes, tools, hooks, and network access run in activities.

## What Actant does

Actant supplies:

- the thread workflow and activity lifecycle;
- provider-neutral model and message interfaces;
- durable inbox, turn, tool, and cancellation semantics;
- explicit tool admission and external resolution;
- projection store contracts and included implementations;
- streaming and lifecycle hooks;
- memory cards and memory tools;
- subagent delegation primitives.

## What the application does

The host application still owns:

- authentication, authorization, and tenant isolation;
- agent selection and model IDs;
- prompts, tools, and domain policy;
- HTTP/SSE/websocket APIs and event transport;
- artifact and blob storage;
- subagent harvest and cancellation policy;
- migrations and deployment topology.

That boundary is intentional. Actant governs durable agent execution without
requiring an application to adopt a web framework, UI protocol, or product data
model.

## Durable state versus live events

Persisted messages and tool calls are the reload path. Hooks and stream
listeners are the low-latency path. A UI normally consumes both:

1. Load projections to reconstruct the transcript.
2. Subscribe to events for new deltas and lifecycle changes.
3. On reconnect, load projections again rather than assuming every event was
   received.

This distinction is why the demo can show streaming nested agents and still
reconstruct the same transcript after a page refresh.
