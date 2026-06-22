# Actant Runtime Guide

Actant is the agent runtime kernel. Product services compose it with their own
coordinator, API, auth, artifacts, UI events, and lifecycle.

This document is for engineers and agents building product runtimes on top of
Actant.

## Mental Model

`AgentRuntime` wires three workers over shared runtime stores:

- `AgentOrchestrator` consumes wake signals and inbox messages, then creates
  turn jobs and tool jobs.
- `TurnWorker` performs one LLM turn and persists assistant/tool-call state.
- `LocalToolWorker` claims one tool job, executes the tool, persists the result,
  and emits a wake signal.

Apps own the long-running service loop. Actant owns the queue/job semantics.

The product service should own:

- thread/session metadata outside Actant's internal thread rows
- user auth and tenancy checks
- artifact/blob storage
- SSE/websocket/event publishing
- cancellation API
- driver task lifecycle
- hooks that bridge runtime events into the product

## Production Driver Pattern

Use a small pool of app-owned driver tasks that repeatedly call
`runtime.run_one()`.

```python
import asyncio

from actant.runtime import AgentRuntime
from actant.runtime.types.orchestration import StepStatus

DRIVER_CONCURRENCY = 4
IDLE_SLEEP_SECONDS = 0.05


class ProductCoordinator:
    def __init__(self, runtime: AgentRuntime) -> None:
        self.runtime = runtime
        self._drivers: set[asyncio.Task[None]] = set()
        self._shutdown = False

    def ensure_drivers_started(self) -> None:
        running = {task for task in self._drivers if not task.done()}
        self._drivers = running
        for _ in range(DRIVER_CONCURRENCY - len(running)):
            self._drivers.add(asyncio.create_task(self.drive()))

    async def drive(self) -> None:
        while not self._shutdown:
            step = await self.runtime.run_one()
            if step.status == StepStatus.IDLE:
                await asyncio.sleep(IDLE_SLEEP_SECONDS)

    async def shutdown(self) -> None:
        self._shutdown = True
        for task in self._drivers:
            task.cancel()
        await asyncio.gather(*self._drivers, return_exceptions=True)
        self._drivers.clear()
```

`run_until_idle()` is useful for tests, examples, and local demos. Deployed
services should prefer a driver pool over one task per thread. With durable SQL
stores, multiple drivers can safely claim disjoint work.

## Agent Registration Rule

Register the `AgentDefinition` before enqueuing a message or wake that drivers
can claim.

```python
self._agents[agent.id] = agent
self._thread_contexts[thread_id] = context
self._turn_states.setdefault(thread_id, TurnState())

await self.runtime.orchestrator.send_message(
    agent_id=agent.id,
    thread_id=thread_id,
    payload={"content": user_message},
)
self.ensure_drivers_started()
```

This ordering matters. A driver can claim the wake immediately after it is
created, so the runtime must already be able to resolve the agent, hooks,
listener, and per-thread context.

Use a mutable `agents` mapping when your app creates per-thread agents:

```python
agents: dict[str, AgentDefinition] = {}
runtime = AgentRuntime(stores=stores, agents=agents)
agents["assistant:thread_1"] = build_agent(...)
```

## Hooks And Persistence

Runtime stores persist conversation state, tool calls, jobs, and wake signals.
Hooks should publish events or update app-owned metadata. Do not double-write
messages from hooks.

Good hook responsibilities:

- publish text deltas to SSE/websocket clients
- update product thread status
- persist generated artifacts in product storage
- emit audit events
- bridge waiting-tool prompts to a UI

Avoid:

- writing duplicate assistant/user messages
- mutating runtime tables outside store APIs
- doing expensive blocking work inside hooks

## Concurrency Guarantees

`run_one()` claims at most one unit of work. A product service gets concurrency
by running multiple driver tasks.

With Postgres-backed stores, job claims use row-level locking and skip locked
rows, so concurrent drivers claim different jobs. This is what allows a single
agent turn with multiple tool calls to execute those tool jobs in parallel.

Use a modest driver count. Four drivers is a reasonable default for a
single-user conversational product with 1-3 parallel tool calls per turn. Larger
values should be paired with provider and tool-level rate limits.

## Start Run Flow

A product `start_run()` usually does this:

1. Create or load the product thread.
2. Resolve the agent name and immutable thread context.
3. Build/register the `AgentDefinition`.
4. Mark product thread status active.
5. Enqueue the message with `orchestrator.send_message(...)`.
6. Ensure runtime drivers are running.
7. Return the product thread id immediately.

Keep product ids and Actant ids explicit. A common pattern is:

```python
agent_id = f"{agent_name}:{thread_id}"
```

## Deferred Tools

Waiting tools are resolved by the product service, not by Actant automatically.
A typical flow:

1. Tool admission returns `ToolDecision.wait(...)`.
2. Product API records or displays the wait request.
3. User or service resolves it.
4. Product updates the stored tool call status/result.
5. Product enqueues `WakeSignal(..., reason=WakeReason.TOOL_UPDATED)`.
6. Product ensures drivers are running.

The continuation path is still queue-driven. Do not call the LLM directly from
the resolve endpoint.

## Cancellation

With a global driver pool, cancellation is queue/state based. Do not assume a
one-task-per-thread model.

Product cancellation should:

- cancel queued turn jobs for the thread
- cancel queued tool jobs for the thread
- cancel active internal runs for the thread
- update the runtime thread status if present
- update product thread status
- publish a cancellation event

A tool already running in a driver may finish. The important invariant is that
queued or subsequent work for that thread does not continue after cancellation.

## Shutdown

Shut down in this order:

1. Stop and await runtime driver tasks.
2. Close event publishers/subscribers.
3. Close Redis/queue clients.
4. Close DB pools.

Drivers should be cancelled before their stores and publishers are disposed.

## Minimal Product Skeleton

```python
class ProductCoordinator:
    def __init__(self, stores, publisher) -> None:
        self.agents: dict[str, AgentDefinition] = {}
        self.contexts: dict[str, ProductContext] = {}
        self.runtime = AgentRuntime(
            stores=stores,
            agents=self.agents,
            hooks_factory=self.hooks_for,
            listener_factory=self.listener_for,
        )

    async def start_run(self, thread_id: str, message: str) -> None:
        agent = self.build_agent(thread_id)
        self.agents[agent.id] = agent
        self.contexts[thread_id] = ProductContext(...)

        await self.runtime.orchestrator.send_message(
            agent_id=agent.id,
            thread_id=thread_id,
            payload={"content": message},
        )
        self.ensure_drivers_started()

    def hooks_for(self, thread):
        return ProductHooks(thread_id=thread.id, publisher=self.publisher)
```

Keep product concerns outside Actant. Keep runtime state changes behind Actant
stores. The product coordinator is the seam between the two.
