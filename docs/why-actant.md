# Why Actant?

Actant makes durable agent orchestration the default instead of an
application-level engineering project.

Defining an agent and registering a tool is the easy part. Production agents
also have to coordinate concurrent calls, wait for people or external systems,
survive worker loss, resume the correct work, preserve model-valid result
ordering, and apply the same rules through nested agents. Those concerns are
easy to underestimate because they appear only after the first successful
demo.

Actant provides one opinionated runtime contract for all of them.

## What the application does not assemble

An Actant application does not need to build:

- a graph node for every tool or approval boundary;
- a custom pause/resume state machine;
- a polling loop for approval records;
- a process that remains alive while a person responds;
- serialization logic for reconstructing an interrupted agent run;
- fan-out/fan-in coordination for a model's parallel tool calls;
- a second approval mechanism for subagents;
- recovery code that decides which turn to repeat after worker loss;
- duplicate-submission reconciliation for deferred results;
- UI events as a hidden source of orchestration truth.

The application still owns its agents, tools, prompts, authorization, domain
policy, UI, and external side-effect safety. Actant owns the durable agent
control flow between them.

## The central guarantee

One model response may emit a group of tool calls. Actant gives every call a
durable identity and progresses the calls independently, while treating the
group as a barrier:

1. Admit all calls concurrently.
2. Execute every `ALLOW` call without waiting for its siblings.
3. Record every `BLOCK` call as a terminal result.
4. Park every `WAIT` call using Temporal asynchronous activity completion.
5. Accept external resolutions by the original tool-call identity.
6. Finalize one deterministic tool-result group after every sibling is
   terminal.
7. Only then begin the next agent turn.

This is especially important for a mixed group:

```text
time ----------------------------------------------------------------->

lookup:       admit -> execute -> completed
publish:      admit -> WAIT ................. approval -> completed
subagent:     admit -> WAIT ..... child run ............ -> completed
group:        [================ durable barrier =====================]
next turn:                                                       start
```

The lookup is not serialized behind the approval. The agent does not continue
with only the lookup result. No worker is occupied merely because the human or
subagent has not answered yet.

## Why Temporal matters

Actant is not a database polling loop disguised as an agent runtime. Temporal
is authoritative for coordination: workflow history records scheduled and
completed activities, inbox signals, cancellation, and whether the durable
barrier has closed.

Projection stores serve a different purpose. They keep threads, runs,
messages, wait requests, and tool-call states easy for APIs and UIs to read.
Hooks and stream listeners provide responsive live delivery, but correctness
does not depend on a browser connection or on every event being received.

When an external approval arrives, Actant persists the terminal result and
completes the already-waiting Temporal activity by identity. It does not invoke
the model from the approval endpoint, and it does not rerun the original
deferred activity body. Temporal wakes the workflow when that durable handle is
ready.

## Subagents are not a special case

A subagent is invoked through the same tool system. If the child waits for an
approval, that state can be projected to the root UI. When the child completes,
durable completion resolves the parent's waiting tool call. The parent remains
behind its original tool-group barrier until every sibling—including the
subagent call—is terminal.

That symmetry is valuable: tools, human decisions, external jobs, and delegated
agents all use the same continuation semantics.

## How adjacent frameworks differ

The ecosystem changes quickly, and several frameworks now implement meaningful
parts of this design. Actant's claim is not that these capabilities are
impossible elsewhere. Its claim is that applications should not have to
assemble them.

| Framework | What it provides | What remains different |
|---|---|---|
| [Pydantic AI](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) | Parallel tools, deferred requests/results, approvals, external tools, and several durable-execution integrations, including [Temporal](https://pydantic.dev/docs/ai/integrations/durable_execution/temporal/) | Its out-of-process deferred flow is expressed as ending a run, carrying message history and deferred requests/results, and beginning a follow-up run. Actant owns that durable continuation as its normal runtime contract. |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/interrupts) | Durable checkpointers, parallel graph branches, multiple interrupts, subgraphs, and resume commands | Applications explicitly construct the graph and resumption flow. Interrupted nodes restart from their beginning, so code before an interrupt must be designed for re-execution. |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/human_in_the_loop/) | Parallel tool execution, serializable run state, partial approvals, and nested approval propagation | Durable distribution is supplied through a separate [Temporal, Restate, or Dapr integration](https://openai.github.io/openai-agents-python/running_agents/). Actant fixes the Temporal-backed lifecycle and projection model as one package contract. |
| [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/integrations/durable-extension) | Durable graph workflows, fan-out/fan-in, request ports, sub-workflows, and recovery through Durable Task | It is a broader executor/workflow system. Actant is narrowly organized around the semantics of agent turns and their tool-call groups. |

The practical distinction is the developer-facing abstraction. Other systems
may expose graph topology, interrupt values, serialized run snapshots, deferred
result objects, external events, or durable promises. Actant exposes an agent
thread whose turns and tools already obey one durable orchestration protocol.

## Guarantees and boundaries

Actant is precise about what it can and cannot guarantee:

- **Durable coordination:** a workflow can recover after process or worker
  loss without reconstructing control flow from UI state.
- **Concurrent progress:** one waiting tool does not prevent allowed siblings
  from executing.
- **Barriered continuation:** the next model invocation cannot observe a
  partially resolved tool group.
- **Stable resolution routing:** external responses target the original
  thread and tool-call identity.
- **Deterministic transcript materialization:** completion timing does not
  scramble the order presented back to the model.
- **Readable projections:** applications can rebuild a UI without replaying
  Temporal history.
- **At-least-once infrastructure reality:** Temporal durability does not make
  arbitrary external side effects magically idempotent. Tools such as payment,
  email, or provisioning operations still need application-level idempotency
  keys where duplicates would be harmful.

The last boundary matters. Actant removes orchestration boilerplate; it does
not pretend that every external system participates in one transaction.

## When Actant is a good fit

Use Actant when an agent must remain correct across long waits, parallel tool
calls, process restarts, human decisions, external jobs, or nested delegation.
It is particularly useful when API and worker processes are separate and the
UI must display durable state without owning it.

For a short-lived agent that performs a few local tools and returns inside one
request, a smaller in-process loop may be sufficient. Actant earns its keep
when agent execution becomes a real distributed workflow.

Continue with [core concepts](concepts.md), then read the exact
[runtime architecture](architecture.md).
