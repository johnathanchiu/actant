# Runtime architecture

This guide is the code-reading map for Actant. It explains not only which
components exist, but where execution crosses a durability boundary and which
component owns each decision.

If you only need the vocabulary, begin with [core concepts](concepts.md). If
you are trying to change or debug the runtime, start here.

## The shortest accurate model

Actant runs one long-lived Temporal workflow for each `(agent_id, thread_id)`.
The workflow is the durable coordinator. Activities perform all work that can
touch the outside world.

```text
AgentRuntime (client API)
    |
    | signal-with-start / complete activity / cancel / query
    v
AgentThreadWorkflow (durable decisions)
    |
    | schedules and awaits
    v
TemporalRuntimeActivities (side effects)
    |
    +-- model provider
    +-- tools
    +-- projection stores
    +-- lifecycle events and stream listeners
```

The central invariant is:

> Tool calls emitted by one assistant turn progress independently, but the
> next model turn cannot start until every call in that group has reached a
> terminal result and the group has been finalized into the transcript.

## Read the implementation in this order

1. `actant/runtime/temporal/workflow.py`
   - `AgentThreadWorkflow.run`: lifetime of a thread.
   - `AgentThreadWorkflow._do_run`: model-turn loop for one run.
   - `AgentThreadWorkflow._execute_tool_group`: the group barrier.
2. `actant/runtime/temporal/activities.py`
   - External work and projection writes scheduled by the workflow.
3. `actant/runtime/temporal/client.py` and `worker.py`
   - Client operations and worker registration.
4. `actant/runtime/temporal/types.py`
   - Serializable payloads crossing workflow/activity boundaries.
5. `actant/runtime/interfaces/stores.py`
   - Projection-store contracts.
6. `actant/runtime/hooks.py`
   - Optional live event and model-stream observers.

The public entry point is `actant/runtime/runtime.py`. It deliberately contains
almost no orchestration: `AgentRuntime` delegates commands to its Temporal
client.

## Thread, run, turn, and group

```text
thread (one long-lived workflow)
└── run (work caused by one drain of the inbox)
    ├── turn 1 (one model invocation)
    │   └── tool group
    │       ├── tool call A
    │       └── tool call B
    ├── turn 2
    └── ...
```

- A **thread** remains addressable between user messages.
- A **run** starts after the workflow drains its inbox and ends on completion,
  exhaustion, failure, or cancellation.
- A **turn** is one model call.
- A **tool group** contains every tool call emitted by the same assistant turn.

All calls in a group share a `group_id`. Every persisted call also carries the
`run_id` and `turn_id` that produced it.

## The workflow algorithm

The following pseudocode mirrors `AgentThreadWorkflow` intentionally. Keep the
documentation and method order aligned when changing the algorithm.

```python
while thread_is_active:
    wait_until_inbox_has_messages()
    messages = drain_inbox()
    run_id = new_id()
    start_run(run_id)

    while turns_remain:
        turn = run_turn(messages_on_first_iteration_only)
        if turn.has_no_tool_calls:
            finish_run(COMPLETED)
            break

        terminal_tool = execute_tool_group(turn.tool_calls)
        if terminal_tool:
            finish_run(COMPLETED)
            break

    park_until_the_next_inbound_message()
```

`run_turn` is an activity because it loads projections, calls the model,
streams provider output, and persists the resulting assistant message. The
workflow sees only its durable `TurnResult`.

### Tool-group algorithm

For every tool call in one turn:

```text
1. Schedule admit_tool for every call.
2. Await all admission outcomes in completion order.
3. For each outcome:
   ALLOW -> schedule execute_tool
   WAIT  -> schedule await_external_resolution
   BLOCK -> no second activity; admission already persisted a terminal result
4. Await every scheduled execution/wait handle in completion order.
5. Run finalize_tool_group once.
6. Return control to the model-turn loop.
```

Admission and execution use `workflow.as_completed`. Completion order does not
control transcript order: `finalize_tool_group` materializes results in a
deterministic order after the whole group has resolved.

For a mixed group, the timeline can be:

```text
time --->

allowed call:   admit -- execute ---------------- completed
deferred call:  admit -- WAIT ........ approve -- completed
group barrier:  ================================== open
next model turn:                                  start
```

The allowed call does not wait for the deferred call before executing. The
agent does wait before starting another model turn.

## Deferred activity completion

`await_external_resolution` uses Temporal asynchronous activity completion:

1. The activity records its Temporal activity identity on the tool-call
   projection.
2. It asks Temporal to leave the activity logically incomplete.
3. The activity invocation returns control to the worker; no Python task or
   worker slot must remain occupied while a person or subagent responds.
4. `AgentRuntime.resolve_deferred_tool_call` validates the persisted waiting call and
   completes that Temporal activity by identity.
5. Temporal records the result and wakes the workflow.
6. The group barrier closes only when all sibling handles have also completed.

The activity identity is routing information, not a second scheduler. Temporal
history remains authoritative for whether workflow execution may continue.

## Activity contracts

Temporal gives an activity a durable scheduled/completed boundary. It does not
make several external side effects one database transaction. Each activity
therefore needs an explicit idempotency expectation.

| Activity | Responsibility | External effects | Idempotency key/expectation |
|---|---|---|---|
| `start_run` | Open a projected run | Thread/run writes | `run_id`; safe to observe an existing run |
| `run_turn` | Produce one assistant turn | Message reads/writes, model call, live events | `turn_id` identifies the logical turn; model calls are not inherently idempotent |
| `admit_tool` | Classify one call | Tool construction/policy, tool-call write, event | `tool_call_id`; terminal failure conversion at activity boundary |
| `execute_tool` | Run one allowed call | Arbitrary tool side effect, tool-call write, event | Tool implementations own external idempotency; automatic retries are disabled |
| `await_external_resolution` | Park a deferred call | Store Temporal identity, async completion | `tool_call_id`; external resolution reconciles stale handles |
| `finalize_tool_group` | Append tool results | Message writes, resolved events | `group_id`; message stores must prevent duplicate materialization |
| `finalize_run` | Close run projection | Run/thread writes, completion event | `run_id`; terminal writes are repeatable |
| `apply_thread_cancellation` | Repair open projected state | Run/thread/tool/message writes | Thread identity; explicitly idempotent |

Activity code converts expected tool/admission failures into typed outcomes so
one failed tool does not fail the entire workflow task. Infrastructure failures
that escape an activity still follow its configured Temporal retry policy.

## Tool-call states and continuation

Names are defined in `actant/tools/calls.py`; the important semantic
distinction is terminal versus non-terminal.

| State | Meaning | Group may treat this call as resolved? |
|---|---|---:|
| `REQUESTED` | Recorded but not classified | No |
| `RUNNING` | Allowed and executing | No |
| `WAITING` | Awaiting an external result | No |
| `COMPLETED` | Successful result persisted | Yes |
| `BLOCKED` | Admission rejected; error result persisted | Yes |
| `FAILED` | Execution/resolution error persisted | Yes |

Failure is terminal, not invisible. The transcript must receive one tool-result
message for every tool call the model emitted; otherwise the next provider call
would see an invalid assistant/tool-message sequence.

## Three kinds of state

### Temporal workflow state

Authoritative for execution: inbox contents, current run, cancellation, the
scheduled activity handles, and whether the group barrier has closed. Workflow
code must stay deterministic because Temporal reconstructs it by replay.

### Projection-store state

Authoritative for product reads: threads, runs, messages, tool-call status,
wait requests, and activity routing identity. APIs and viewers query these
stores instead of replaying workflow history.

Projection state describes execution; it must not independently decide that a
workflow may continue.

### Live events

Hooks and stream listeners are low-latency notifications. Consumers must assume
events can be missed or duplicated and reload projections on reconnect.

Live events must not become the only mechanism for a durable coordination
decision. In particular, parent/subagent completion should ultimately be owned
by durable orchestration rather than by an SSE-publishing hook.

## Hooks versus stream listeners

`AgentThreadHooks` reports lifecycle facts such as persisted assistant
messages, waiting tools, tool results, and run completion.

`StreamListener` reports provider output before the final assistant message is
assembled: text deltas, thinking deltas, and partial tool arguments.

```text
provider stream -- StreamListener --> responsive UI
       |
       v
assembled message -- persist -- AgentThreadHooks --> lifecycle event
```

Good consumers include SSE/websocket publication, telemetry, audit feeds, and
non-critical notifications. Hooks should not duplicate canonical message
writes. A reconnecting consumer loads stores first and then resumes live event
consumption.

## Client and worker

`AgentRuntime` is a client facade. It can live in an API process and sends
commands to Temporal:

- `send_message`
- `resolve_deferred_tool_call`
- `cancel_thread`
- `get_state`

`TemporalRuntimeWorker` hosts the registered workflow and activity
implementations. It must have the same agent definitions, stores, factories,
Temporal namespace, and task queue expected by the client configuration.

They may run in one process for development, but they represent separate
deployment roles.

## Subagents

A deferred `TaskTool` makes a parent tool call wait while another thread runs.
The parent/child relationship includes:

- parent thread ID;
- parent tool-call ID;
- child thread ID;
- child agent identity.

Child lifecycle and streaming events may be dual-published to the root thread
for UI display. Event routing is not the same thing as durable parent
resolution. This boundary is an active hardening area: production-grade nested
execution must survive worker/process restart without relying solely on an
in-memory subthread registry or completion hook.

See [subagents](subagents.md) for the public patterns and
[coordinator guide](coordinator-guide.md) for application wiring.

## Change checklist

When changing orchestration code, answer these questions in the pull request:

1. Which component is authoritative for the new state?
2. Can the operation be replayed or retried safely?
3. Can a worker disappear after the external effect but before completion is
   recorded?
4. Does every emitted tool call still receive exactly one transcript result?
5. Can one waiting sibling prevent another sibling from starting?
6. Can the next model turn begin before every group member is terminal?
7. Does cancellation reconcile Temporal and projection state?
8. Can the UI reconstruct the same state without receiving live events?
