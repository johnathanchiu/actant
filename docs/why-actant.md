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
4. Suspend every `WAIT` call on a durable Temporal workflow condition.
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

When an external approval arrives, Actant signals the thread workflow by the
original tool-call identity. Temporal records the signal and wakes the workflow,
which schedules a short activity to persist the terminal result. The approval
endpoint does not invoke the model, and no activity remains alive while the
workflow is waiting.

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
| [Pydantic AI](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) | Parallel tools, inline deferred handlers, out-of-process deferred requests/results, approvals, external tools, and several durable-execution integrations, including [Temporal](https://pydantic.dev/docs/ai/integrations/durable_execution/temporal/) | Its documented out-of-process UI flow ends a run and starts a follow-up run with message history and deferred results. Its Temporal integration durably offloads model and tool I/O, but applications still define the surrounding workflow. Actant supplies a long-lived thread workflow, resolution signals, projections, and tool-group barrier as one runtime contract. |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/interrupts) | Durable checkpointers, high-level prebuilt agents, parallel graph branches, multiple interrupts, subgraphs, resume commands, and time-travel debugging | LangGraph is more general and configurable. Its graph/checkpoint/interrupt model is a feature when topology is product logic. Actant instead fixes one narrower agent-thread protocol, and interrupted LangGraph nodes restart from their beginning, so work before an interrupt must be safe to repeat. |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/human_in_the_loop/) | Parallel tool execution, serializable run state, partial approvals, streaming resume, and nested approval propagation | Durable distribution is an optional integration layer rather than the base run contract. Actant fixes the Temporal-backed lifecycle, tool-group barrier, and queryable projections as one package contract. |
| [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/integrations/durable-extension) | Durable graph workflows, fan-out/fan-in, request ports, sub-workflows, and recovery through Durable Task | It is a broader executor/workflow system. Actant is narrowly organized around the semantics of agent turns and their tool-call groups. |

These are not claims of exclusive capability. LangGraph is a better fit when
explicit graph topology, checkpoint inspection, or time travel is the product;
Pydantic AI and the OpenAI Agents SDK have broader model/tool ecosystems. The
practical distinction is the developer-facing default. Other systems expose
graph topology, interrupt values, serialized run snapshots, deferred result
objects, or integration hooks. Actant exposes an agent thread whose turns and
tools already obey one Temporal-backed orchestration protocol.

One concrete default differs from LangChain's prebuilt HITL middleware. Its
[documented execution lifecycle](https://docs.langchain.com/oss/python/langchain/human-in-the-loop#execution-lifecycle)
interrupts after the model response and before the batch's tool calls execute
when any call needs review. Actant admits calls independently: allowed siblings
can finish while another sibling waits, while the next model turn remains behind
the complete-group barrier. Raw LangGraph can model the same behavior with
custom parallel branches; Actant supplies it without making that graph topology
application code.

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
