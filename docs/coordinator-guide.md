# Building a coordinator on actant

This guide is for apps that go beyond a single-agent chatbot — apps
that have multiple agents, delegate work via `task()`, or need
robust state recovery when Temporal and the actant store diverge.

If you're building a single-agent chatbot, skip this guide and use
[`AgentRuntime`](./actant-runtime-guide.md) directly. You don't need
a coordinator.

## The problem actant doesn't solve for you

Actant is a kernel: durable inbox, tool admission, hooks,
replay. It deliberately doesn't ship policy for:

- **Per-thread vs global agent definitions** — does each thread get
  its own `AgentDefinition` (different model, workspace, tools)?
  Or do all threads share one? Your call.
- **Sub-thread lifecycle** — when a parent's `task()` tool delegates
  to a sub-agent, who registers the relationship? Where does the
  sub-agent's terminal event trigger the parent's deferred-tool
  resolution? Your code.
- **Harvest semantics** — when the sub-agent finishes, what gets
  passed back to the parent? Just the final text? An artifact bundle?
  A structured envelope? Your decision.
- **State reconciliation** — if Temporal's view diverges from the
  store's (volume reset, activity timeout, workflow termination),
  who reconciles?
- **Cancellation policy** — when a parent thread is cancelled, do
  its sub-threads cancel too? Continue? Get reaped?

A production coordinator is one set of answers to these questions.
Yours will be different depending on whether your app needs workspace
directories, artifact gateways, owner-scoped event channels, or a much
simpler thread model.

## What actant DOES provide

[`actant.runtime.coordinator`](../actant/runtime/coordinator.py)
ships four primitives:

1. **`SubThreadLink`** — dataclass capturing a single parent ↔ sub
   relationship.
2. **`SubThreadRegistry`** — in-memory map of active links.
3. **`publishing_hooks_factory`** / **`publishing_listener_factory`** —
   build `AgentRuntime` factories that auto-dual-publish sub-thread
   events to the parent's SSE channel.
4. **`resolve_deferred_tool_call`** — thin wrapper over
   `AgentRuntime.resolve_deferred_tool_call` that surfaces
   `ToolResolutionStaleError` when Temporal has lost the activity
   so apps don't keep WAITING state alive forever.

Plus a related framework fix:

- **`TaskTool` now takes `parent_thread_id` per-call** (reads
  `call.thread_id` if no construction-time value). A single
  TaskTool instance can be shared across threads — you don't need
  per-thread agent construction just for delegation.

## The canonical pattern

```python
from actant.agents import AgentDefinition
from actant.runtime import AgentRuntime, RunCompletion, TemporalRuntimeWorker
from actant.runtime.coordinator import (
    SubThreadLink,
    SubThreadRegistry,
    publishing_hooks_factory,
    publishing_listener_factory,
    resolve_deferred_tool_call,
)
from actant.runtime.exceptions import ToolResolutionStaleError
from actant.runtime.stores.in_memory import InMemoryEventPublisher
from actant.tools.task import TaskTool


class MyCoordinator:
    def __init__(self, stores, llm):
        self.stores = stores
        self.publisher = InMemoryEventPublisher()
        self.registry = SubThreadRegistry()

        # Build agents — your policy decides per-thread vs global.
        self.main_agent = AgentDefinition(
            id="main",
            name="Main",
            persona="...",
            llm=llm,
            tools=ToolRegistry(
                [
                    # TaskTool reads call.thread_id at invocation time, so
                    # ONE instance works for many threads.
                    TaskTool(
                        spawner=self,  # implements SubagentSpawner
                        subagent_choices=["researcher"],
                    ),
                    # ... your other tools ...
                ]
            ),
        )
        self.researcher_agent = AgentDefinition(
            id="researcher",
            name="Researcher",
            persona="...",
            llm=llm,
            tools=ToolRegistry([...]),
        )

        # Wire AgentRuntime with the registry-aware factories.
        self.runtime = AgentRuntime(
            stores=stores,
            agents={
                self.main_agent.id: self.main_agent,
                self.researcher_agent.id: self.researcher_agent,
            },
            hooks_factory=publishing_hooks_factory(self.publisher, registry=self.registry),
            listener_factory=publishing_listener_factory(self.publisher, registry=self.registry),
        )

    # TaskTool's SubagentSpawner Protocol —
    # ``can_execute`` calls this with the parent's thread/tool_call.
    async def spawn(
        self,
        *,
        name,
        message,
        context,
        parent_thread_id,
        parent_tool_call_id,
    ):
        sub_thread_id = f"sub_{uuid.uuid4().hex[:10]}"
        # Register the link BEFORE send_message so the hook factory
        # sees the relationship synchronously.
        self.registry.register(
            SubThreadLink(
                sub_thread_id=sub_thread_id,
                parent_thread_id=parent_thread_id,
                parent_tool_call_id=parent_tool_call_id,
                sub_agent_id=self.researcher_agent.id,
                subagent_name=name,
            )
        )
        await self.runtime.send_message(
            self.researcher_agent.id,
            sub_thread_id,
            message,
        )

    # Passed to TemporalRuntimeWorker(run_completion_handler=...).
    # This runs inside the retryable finalize_run activity, after the
    # child's thread/run/message projections have committed.
    async def handle_run_completion(self, completion: RunCompletion):
        child = await self.stores.threads.get(
            completion.agent_id,
            completion.thread_id,
        )
        if child.parent_tool_call_id is None or child.parent_thread_id is None:
            return
        parent_call = await self.stores.tool_calls.get(child.parent_tool_call_id)
        messages = await self.stores.messages.list_for_thread(
            completion.agent_id,
            completion.thread_id,
        )
        final_text = next(
            (
                m.content
                for m in reversed(messages)
                if m.role == "assistant" and isinstance(m.content, str)
            ),
            completion.outcome,
        )
        envelope = {"text": final_text, "subagent": completion.agent_id}
        try:
            await resolve_deferred_tool_call(
                self.runtime,
                agent_id=parent_call.agent_id,
                thread_id=child.parent_thread_id,
                tool_call_id=parent_call.id,
                approved=completion.succeeded,
                answer=json.dumps(envelope),
            )
        except ToolResolutionStaleError:
            # Parent's activity is gone (workflow cancelled / Temporal
            # reset). The store is already marked FAILED — nothing
            # more to do.
            pass

    # Worker wiring is separate from AgentRuntime's client role.
    # worker = TemporalRuntimeWorker(
    #     stores=self.stores,
    #     agents={...},
    #     run_completion_handler=self.handle_run_completion,
    # )

    # Single entry point for user-driven resolves too — funneling
    # both paths through the same method gives one place to handle
    # state divergence.
    async def resolve_user_input(
        self,
        *,
        agent_id,
        thread_id,
        tool_call_id,
        approved=None,
        answer="",
        payload=None,
    ):
        try:
            await resolve_deferred_tool_call(
                self.runtime,
                agent_id=agent_id,
                thread_id=thread_id,
                tool_call_id=tool_call_id,
                approved=approved,
                answer=answer,
                payload=payload,
            )
        except ToolResolutionStaleError as exc:
            # Surface a clean error upward. The store is reconciled;
            # the UI should pull fresh thread state.
            raise RuntimeError(
                f"This deferred tool call is no longer pending: {exc.reason}"
            ) from exc
```

## What you DON'T have to do

- Use hooks to continue a parent. Hooks only publish observations;
  `RunCompletionHandler` owns retryable completion integration.
- Maintain a side-channel map of "is this thread a sub-thread"
  (the registry IS that map; the factories consult it).
- Handle "activity not found" errors from Temporal yourself (the
  runtime reconciles + raises a typed error).
- Build separate code paths for user-driven vs sub-thread-driven
  resolves (one `resolve_deferred_tool_call` entry, two callers).

## What you still own

- **Per-thread agent construction strategy.** Some apps rebuild an
  agent for every thread; the demo registers one agent globally and
  uses `call.thread_id` per-invocation via TaskTool's fallback. Both
  are valid — pick what fits your app.
- **Harvest semantics.** What does "sub-agent done" mean for your
  parent? A production app might extract artifacts or structured
  outputs. The demo just passes the final text. Your call.
- **Cancellation policy.** When a parent thread cancels, do
  in-flight sub-threads cancel too? Production apps often implement
  cascade-cancel; the demo doesn't bother.
- **Registry reconstruction.** `SubThreadRegistry` is an in-memory live-event
  routing index. Rebuild active links from thread and tool-call projections on
  startup. Parent continuation remains safe because the completion handler
  derives linkage from those projections rather than registry memory.

## Reference implementation

The `actant` repo's `examples/demo/` directory contains a complete worked
example (`DemoCoordinator`) that uses these primitives. Read it
alongside this guide — it's intentionally minimal but production-
shaped.

## When you DON'T need a coordinator

Counter-example. If you're building this:

```python
runtime = AgentRuntime(
    stores=InMemoryRuntimeStores(),
    agents={"bot": my_agent},
)
thread_id = uuid.uuid4().hex
await runtime.send_message("bot", thread_id, "hello")
```

…and you don't have a `task()` tool, and you don't run a worker
across multiple processes, you don't need ANY of this. Use the
runtime directly. The coordinator pattern is for apps with state
that needs to be coordinated; if there's no coordination to do,
adding a coordinator is pure overhead.
