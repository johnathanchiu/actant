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

Pass a `spawner` when the child should own a thread. `TaskTool` spawns the
child, returns `WAIT` for the parent's task call, and the parent parks. When the
child completes, the application resolves that task call with the harvested
child output.

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
- parent task tool-call ID;
- child agent ID and display name.

Register this link before sending the child's first message. Publishing hooks
can then dual-publish child events onto the parent's channel with enough
metadata for a viewer to place them under the correct task call.

`SubThreadRegistry` is an in-memory implementation. Production applications
that must survive coordinator restarts should persist links or reconstruct them
from thread projections.

## Completion and harvesting

Child completion is not automatically equivalent to “return the final text.”
The coordinator owns harvest semantics. It might return:

- the last assistant message;
- structured findings;
- artifact references;
- a success/failure envelope;
- a product-specific result assembled from several stores.

That result resolves the parent's parked task call. `TaskTool.on_resolve`
converts the JSON envelope into a normal `ToolResult`, after which the parent
continues its turn loop.

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
