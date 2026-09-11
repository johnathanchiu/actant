# Actant Tools Guide

Tools are the app-owned capabilities an agent can call during a turn. Actant
provides the protocol, execution lifecycle, admission hooks, and result shape.
The product owns the actual behavior.

## Function tools

For most tools, decorate an annotated sync or async function:

```python
from typing import Annotated

from pydantic import Field

from actant import tool


@tool
async def weather(
    city: Annotated[str, Field(description="City to check")],
    days: int = 1,
) -> dict[str, object]:
    """Get a weather forecast."""
    return {"city": city, "days": days, "forecast": "sunny"}
```

Actant uses the function name, docstring, annotations, defaults, and Pydantic
field metadata to build the model-facing JSON schema. Arguments are validated
before execution. Async functions are awaited; sync functions run in a worker
thread so they do not block the activity event loop.

Native return values are wrapped in `ToolResult.ok(...)`. Return an explicit
`ToolResult` when you need an error, metadata, or content blocks.

```python
from actant.tools import ToolResult


@tool
async def load_report(report_id: str) -> ToolResult:
    """Load a report."""
    if not report_id:
        return ToolResult.fail("report_id is required")
    return ToolResult.ok({"report_id": report_id}, source="warehouse")
```

Register tools on the agent:

```python
agent = AgentDefinition(
    id="assistant",
    name="Assistant",
    persona="...",
    llm=llm,
    tools=ToolRegistry([weather, load_report]),
    tool_allowlist={"weather", "load_report"},
)
```

## Approvals

Add an approval prompt using the function's parameter names:

```python
@tool(approval="Publish {title}?")
async def publish(title: str) -> dict[str, str]:
    """Publish an update."""
    return {"published": title}
```

The call enters the normal durable AWAIT_HUMAN state. The function has not executed
at that point. Resolve it through the thread handle:

```python
await thread.resolve(tool_call_id, approved=True)
```

Approval executes the function only when `approved=True`. Rejection produces a
failed tool result, releases the tool-group barrier, and lets the agent handle
that result normally. Template fields are checked against the function
signature when the tool is defined. Use doubled braces (`{{` and `}}`) for
literal braces.

For policy that cannot be expressed as a template, `approval` may instead be a
sync or async callback receiving the validated argument dictionary.

For custom admission, pass a callback returning `ToolDecision`:

```python
from actant.tools import ToolDecision


async def admit_publish(args):
    if args["title"] == "draft":
        return ToolDecision.deny("Drafts cannot be published")
    return ToolDecision.execute()


@tool(admission=admit_publish)
async def publish(title: str) -> dict[str, str]:
    """Publish an update."""
    return {"published": title}
```

Custom callbacks may also return `ToolDecision.await_human(...)`. Add a `resolve=`
callback when the external answer itself should produce the tool result. Use a
class-based tool when admission needs the full tool-call or turn context.

## Advanced declarative tools

Implement the underlying protocol directly when a tool needs custom invocation
state, full admission context, or specialized resolution behavior:

```python
from actant.core import JSONObject
from actant.tools import BaseDeclarativeTool, BaseToolInvocation, CallContext
from actant.tools import ToolResult, make_tool_schema


class EchoInvocation(BaseToolInvocation[JSONObject, object]):
    async def execute(self) -> ToolResult:
        message = self.params.get("message")
        if not isinstance(message, str):
            return ToolResult.fail("message is required")
        return ToolResult.ok({"echo": message})


class EchoTool(BaseDeclarativeTool):
    def __init__(self) -> None:
        super().__init__(
            "echo",
            make_tool_schema(
                "echo",
                "Echo a message.",
                parameters={"message": {"type": "string"}},
                required=["message"],
            ),
        )

    async def build(self, params: JSONObject, ctx: CallContext) -> EchoInvocation:
        return EchoInvocation(params)
```

## ToolResult

Return success with `ToolResult.ok(...)`:

```python
return ToolResult.ok({"rows": rows})
```

Return failure with `ToolResult.fail(...)`:

```python
return ToolResult.fail("file not found")
```

Use `metadata` for product-side details that should be persisted with the tool
result but are not the main output:

```python
return ToolResult.ok(
    {"summary": "created report"},
    artifact_id="artifact_123",
    mime_type="text/html",
)
```

Three metadata keys mean something to the runtime:

- `terminal=True` ends the run after this tool group, without another model
  turn. `FinishTool` sets it; a verifier tool can too.
- `deliverables=[...]` on a terminal result names workspace paths the runtime
  reads from the thread's sandbox and stores through the worker's
  `ArtifactSink`. The stored refs come back as `metadata["artifacts"]` and on
  `RunCompletion.artifacts`. A path that cannot be read fails the tool and
  drops `terminal`, so the model can correct it.
- `artifacts` is written by the runtime, never by a tool.

Use `content_blocks` when the tool result needs multimodal provider input or
rich persisted blocks:

```python
return ToolResult(
    output={"image_size_bytes": len(data)},
    content_blocks=[
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encoded_png,
            },
        }
    ],
)
```

Products may interpret metadata and content blocks to emit artifacts, render UI
previews, or feed future agent turns.

## Sandboxed tools

A tool that runs code should not run it on the worker. Declare a `Sandbox`
parameter and the runtime opens one sandbox per thread, from the backend the
agent definition names, and hands it in. The parameter never appears in the
schema.

```python
from actant.sandbox import Sandbox, SandboxSpec
from actant.tools import tool


@tool
async def run_script(path: str, sandbox: Sandbox) -> dict:
    """Run a Python file in the workspace."""
    result = await sandbox.exec(["python", path], timeout=120)
    return {"returncode": result.returncode, "stdout": result.stdout[-4000:]}


agent = AgentDefinition(
    ...,
    sandbox=SandboxSpec(backend="local", mount="/srv/agent-workspaces"),
    completion="terminal",
)
```

`Sandbox` is a filesystem plus `exec`: `read`, `write` (whole file), `ls`,
`exec(argv, cwd=, timeout=, env=)`, `close`. Paths are relative to the
thread's root. A class-based tool sets `needs_sandbox = True` and reads
`ctx.sandbox` in `build`. A `CallContext` parameter alone gives a tool its
agent, thread, run and call ids without a sandbox.

Backends: `local` (a directory under `mount`, subprocesses; always
registered) and `modal` (`actant[modal]`: a network-blocked `modal.Sandbox`
over a bucket prefix mounted from your object storage, so the files live
with you, not on Modal). Register providers on the worker:

```python
TemporalRuntimeWorker(
    ...,
    sandbox_providers={"modal": ModalSandboxProvider("my-app", bucket="agents", secret_name="r2")},
    artifact_sink=my_sink,
)
```

The sandbox id is stored on the thread, so another worker reattaches rather
than opening a second one over the same files. An exec's `timeout` must stay
under the ten-minute tool activity. Mounted buckets write whole files only
(no append, no seek), which is what `write` promises anyway.

## Finishing a task

A chat agent is done when it answers. A task agent has to say so. Give it
`FinishTool` and `completion="terminal"`: the run completes only on a
terminal result; a turn with no tool calls gets one reminder, a second ends
the run as exhausted with the stop reason stored on the run. `finish(summary,
paths=[...])` names the deliverables, which the runtime stores as artifacts.

## Advanced admission: Allow, Block, Wait

Most tools do not need admission logic. If a tool must ask for approval or wait
for an external condition, implement `can_execute`.

```python
from actant.tools import ToolDecision, ToolWaitRequest


class PublishTool(BaseDeclarativeTool):
    async def can_execute(self, call, invocation, context):
        if content_policy.blocks(call.args):
            return ToolDecision.deny("content policy blocked this action")

        if not await approval_store.is_approved(call.id):
            return ToolDecision.await_human(
                ToolWaitRequest(
                    kind="publish_approval",
                    prompt="Approve publishing this update?",
                    payload={"tool_call_id": call.id, "args": call.args},
                )
            )

        return ToolDecision.execute()
```

Admission decisions:

- `allow`: execute immediately
- `block`: mark the call blocked with a reason
- `wait`: mark the call waiting and let the product resolve it later

The product resolves waiting calls through its own API/service flow and then
wakes the runtime.

## Product-Owned Side Effects

Tools can touch external systems, but keep side effects explicit and auditable.

Recommended patterns:

- validate input inside `execute`
- return structured output, not only strings
- write large files to product storage and return artifact references
- include enough metadata for UI and later inspection
- make destructive tools use admission
- make long-running remote work resumable where possible

Avoid:

- hiding critical state only in model text
- returning huge payloads that should be artifacts
- doing app authorization inside the model prompt instead of the product API
- writing duplicate conversation messages from inside tools

## Deferred resolution

If a tool waits, the product resolves it through the runtime facade:

```python
thread = runtime.thread(agent_id, thread_id)
await thread.resolve(tool_call_id, approved=True, answer="Approved")
```

Do not update tool-call rows yourself or continue the model inline from the
approval endpoint. `resolve_tool_call` signals the workflow, which persists the
resolution and releases the durable tool-group barrier, allowing the existing
workflow to resume normally.
See [pauses and deferred work](pauses-and-resume.md) for details.

## Testing Tools

Unit-test a function tool directly:

```python
invocation = await weather.build({"city": "Paris", "days": 2}, ctx)
result = await invocation.execute()
assert result.output["city"] == "Paris"
```

Integration-test through `AgentRuntime` when you need to verify:

- schema exposure
- tool-call persistence
- admission behavior
- waiting/resolution
- continuation after tool results

Use fake LLM providers for deterministic tool calls and outputs.
