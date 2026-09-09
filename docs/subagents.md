# Subagents

Actant models delegation as a tool call named `task`. This is deliberately
ordinary: the parent receives a governed, persisted tool result whether the
child is a local function, another durable Actant thread, or a remote service.

## Two modes

### Synchronous invocation

Pass an `invoker` to `TaskTool` when delegated work finishes inside one tool
execution. This is useful for test doubles, deterministic transforms, and
short-lived in-process specialists.

```python
from actant.tools import InMemorySubagentRegistry, TaskTool, ToolRegistry

registry = InMemorySubagentRegistry({"researcher": researcher_invoker})
tools = ToolRegistry([TaskTool(invoker=registry)])
```

The invoker returns a `ToolResult` directly. There is no child thread to
observe or resume.

### Durable delegation

Pass a `spawner` when the child should own a thread. `TaskTool` starts the
child and returns its thread id as the tool result. The parent is not parked
and can start others, or carry on with something else.

```json
{"subagent": "researcher", "thread_id": "thr_...", "sub_thread_id": "thr_...", "status": "running"}
```

The parent does not poll for the ending. A finished sub-thread messages its
parent, which wakes it whether it is parked or already closed — the same
inbox a person's message arrives on. Use `check_subagent`, `message_subagent`
and `stop_subagent` (`actant.tools.supervision_tools`) to look at a running
child, say something else to it, or abandon it.

Delegation used to park the parent on a `WAIT` until the child finished,
resolved by the application through `resolve_tool_call`. That overloaded
`WAIT` — which otherwise always means a person has to answer — onto a machine
finishing its work, and a parent could supervise exactly one child.

```python
task_tool = TaskTool(
    spawner=coordinator,
    subagent_choices=["researcher", "summarizer"],
    subagent_descriptions={
        "researcher": "Collect and compare evidence.",
        "summarizer": "Turn supplied material into a concise brief.",
    },
)
```

The parent thread ID is normally taken from the live tool call. An application
that builds one tool instance per thread may set `parent_thread_id` explicitly.

## Parent and child linkage

A durable coordinator records:

- child thread ID;
- parent thread ID;
- child agent ID and display name.

The parent's task call no longer needs recording as a link: it completes
immediately, and its stored result already names the child thread.

Register this link before sending the child's first message. Publishing hooks
can then dual-publish child events onto the parent's channel with enough
metadata for a viewer to place them under the correct task call.

`SubThreadRegistry` is an in-memory event-routing index. Reconstruct it from
thread projections after restart. It is not the durable completion mechanism.

## Completion and harvesting

Child completion is not automatically equivalent to “return the final text.”
The coordinator owns harvest semantics. It might return:

- the last assistant message;
- structured findings;
- artifact references;
- a success/failure envelope;
- a product-specific result assembled from several stores.

Register a `RunCompletionHandler` on `TemporalRuntimeWorker`, or use the
thread hooks. Either runs after the child's projections are committed. The
handler harvests persisted child output and **sends the parent a message**
saying the child is done.

That message is what resumes the parent. It arrives on the same inbox a
person's message would, so a completing subagent and a user speaking are the
same kind of event, and a parent that has already closed is restarted by it.

Deliver it idempotently: completion handlers retry, and signals are not
deduplicated.

## Nested delegation

A child can have its own `TaskTool`, producing a tree of threads. Each link
always refers to the immediate parent and spawning tool call. Event consumers
can reconstruct arbitrary depth recursively instead of encoding special cases
for “main” and “researcher.”

The included demo exercises main → researcher → summarizer delegation and
reconstructs both levels after a browser reload.

## Policies the application must choose

- Which agents a parent may invoke.
- Whether child definitions are global or built per thread.
- What context crosses the boundary.
- How results and artifacts are harvested.
- Whether cancellation cascades.
- How link state is recovered after a process restart.
- Limits on depth, fan-out, cost, and concurrent delegation.

See the [coordinator guide](coordinator-guide.md) for the complete wiring
pattern.
