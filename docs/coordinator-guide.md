# Application-owned coordination

Actant runs durable agent threads. Applications choose agent definitions, parent/child
relationships, user authorization, and what a completed child means to its parent.
A single-agent application can use `AgentRuntime` directly.

## Persist identity before starting work

Save the child's agent identity, definition inputs, and parent linkage before submitting
its first message. A resolver must reconstruct that definition on a worker that did not
spawn it. Process-local registries cannot be the source of truth.

`TaskTool` returns a child thread ID immediately; the parent can continue. The supervision
tools check, message, or stop that child. Durable completion notifies the parent separately.

## One live event adapter

Pass an `EventSink` to `AgentRuntime(event_sink=...)`. The runtime publishes both lifecycle
and model-stream events through `publish(channel, event)`. Applications can transform
those events into their UI contract or route them to parent and owner channels.

```python
class ApplicationEvents:
    async def publish(self, channel, event):
        await broker.publish(channel, event)
        # Load durable parent/owner links here before routing another copy.


runtime = AgentRuntime(
    client=client,
    stores=stores,
    resolve_agent=resolve_agent,
    event_sink=ApplicationEvents(),
    event_source=broker,
    run_completion_handler=on_complete,
)
```

`event_source` is the optional reader used by `ThreadHandle.events()`. It is explicit,
as is the sink; neither is inferred from a store's `publisher` attribute. A gateway can
subscribe to the application's broker without running a worker.

Each runtime event contains `type`, `thread_id`, and `data`. Activity-scoped data includes
`agent_id`, `run_id`, and, when available, `turn_id`, `turn_uid`, and `turn_index`.
Use these identities instead of tracking a mutable "current turn" in the event adapter.
A delayed event from one activity must not acquire the next activity's identity.

Tool events include `result`, a structured tool-result payload (including partial output on failure), alongside
the display-oriented `output`. Adapters can recover structured output, content blocks,
and artifact metadata without reparsing Python string representations. Authorize and
resolve image references before exposing them to a UI.

Live publication is observational: ordinary sink errors are logged, cancellation
propagates, and events may be lost or repeated. Do not use it as the only delivery path
for durable product actions. Keep adapters lightweight; an awaited slow sink still uses
activity time. Model stream resets tell clients to discard an abandoned attempt's deltas.

## Durable completion

`run_completion_handler` runs after finalization, in a retryable activity. It receives
persisted run/thread identity and outcome. Read parent links from storage and deduplicate
product effects by `run_id`; external delivery is not exactly once.

Applications own any credit gate, usage charging, notifications, or parent follow-up policy.
Actant provides execution gates and usage events without implementing billing policy.
Keep those concerns separate from best-effort UI publication.

## Demo

The [demo coordinator](../examples/demo/server/app/coordinator.py) constructs agents,
spawns children, and notifies parents from durable completion. Its
[event adapter](../examples/demo/server/app/events.py) derives ancestor routing from
persisted thread rows and caches immutable links. A fresh adapter can route a grandchild
straight to the root without a startup registry reconstruction pass.

The former `actant.runtime.coordinator` module and hook/listener factories are removed.
Applications own routing in their sink; there is no replacement coordinator framework.
