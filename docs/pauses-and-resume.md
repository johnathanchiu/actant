# Pauses and deferred work

`WAIT` is the central control-flow primitive for work that cannot safely finish
now. Typical cases include human approval, a user question, an external job,
or a delegated subagent.

## Lifecycle

1. The model requests a tool.
2. The tool's `can_execute` returns `ToolDecision.wait(...)`.
3. Actant persists the call as waiting and emits `on_tool_waiting`.
4. The workflow schedules `await_external_resolution`.
5. That activity switches to Temporal async completion, so no Python worker is
   held while the call is parked.
6. The application later calls `runtime.resolve_tool(...)`.
7. Actant persists the resolution and completes the activity by its stable ID.
8. The workflow wakes, finalizes the tool-result message, and continues the
   normal turn loop.

The workflow is durable during the pause. A web request, event-stream
connection, or worker process does not need to stay alive.

## Declaring a wait

```python
from actant.tools import ToolDecision, ToolWaitRequest

async def can_execute(self, call, invocation, context):
    return ToolDecision.wait(
        ToolWaitRequest(
            kind="publish_approval",
            prompt="Approve publishing this report?",
            payload={"channel": call.args.get("channel")},
        )
    )
```

`kind` is an application-facing discriminator. `prompt` is suitable for a UI,
and `payload` carries structured rendering or policy context. Do not put secrets
in a wait request that will be exposed to clients.

## Resolving a wait

```python
await runtime.resolve_tool(
    "assistant",
    thread_id,
    tool_call_id,
    approved=True,
    answer="Approved by the report owner",
    payload={"review_id": "review_123"},
)
```

Resolution re-enters the existing workflow; it must not call the model directly
from the HTTP endpoint. Tools may implement `on_resolve` to translate the
external `ToolResolution` into the `ToolResult` the model receives.

## Timeouts and stale resolutions

`TemporalRuntimeConfig.external_resolution_timeout_seconds` bounds how long an
external resolution may remain pending. Choose it from product semantics, not
HTTP timeout conventions. Human approvals may reasonably wait for days.

If Temporal no longer has the parked activity, resolution raises
`ToolResolutionStaleError` after reconciling the projection. Multi-agent apps
should funnel resolution through `actant.runtime.coordinator.resolve_deferred`
and surface a refreshable conflict to the caller.

## Cancellation

Cancelling a thread cancels its workflow and reconciles active runs and open
tool calls in the projections. Application-level side effects may already have
occurred, so destructive or non-idempotent tools should still use careful
admission and idempotency keys.

Parent-to-child cancellation is application policy. Actant does not assume
that cancelling a parent should terminate independently useful delegated work.

## Designing user interfaces

A deferred-work UI should:

- render by `wait_request.kind`;
- submit a resolution against the owning thread and tool-call ID;
- disable repeated submission while resolving;
- handle stale-resolution errors by reloading projections;
- rebuild waiting panels from stored open tool calls after reconnect.

The demo viewer implements approval buttons, multiple-choice answers, and
subagent waits using this same lifecycle.
