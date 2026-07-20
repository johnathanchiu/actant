# Actant Tools Guide

Tools are the app-owned capabilities an agent can call during a turn. Actant
provides the protocol, execution lifecycle, admission hooks, and result shape.
The product owns the actual behavior.

## Basic Tool Shape

A tool has:

- `name`
- JSON schema exposed to the model
- `build(params)` returning a `ToolInvocation`
- `execute()` returning a `ToolResult`

Use `BaseDeclarativeTool`, `BaseToolInvocation`, and `make_tool_schema` for the
common case:

```python
from actant.core import JSONObject
from actant.tools import BaseDeclarativeTool, BaseToolInvocation
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
                name="echo",
                description="Echo a message.",
                parameters={
                    "message": {
                        "type": "string",
                        "description": "Message to echo.",
                    },
                },
                required=["message"],
            ),
        )

    async def build(self, params: JSONObject) -> EchoInvocation:
        return EchoInvocation(params)
```

Register tools on the agent:

```python
agent = AgentDefinition(
    id="assistant",
    name="Assistant",
    persona="...",
    llm=llm,
    tools=ToolRegistry([EchoTool()]),
    tool_allowlist={"echo"},
)
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

## Admission: Allow, Block, Wait

Most tools do not need admission logic. If a tool must ask for approval or wait
for an external condition, implement `can_execute`.

```python
from actant.tools import ToolDecision, ToolWaitRequest


class PublishTool(BaseDeclarativeTool):
    async def can_execute(self, call, invocation, context):
        if content_policy.blocks(call.args):
            return ToolDecision.block("content policy blocked this action")

        if not await approval_store.is_approved(call.id):
            return ToolDecision.wait(
                ToolWaitRequest(
                    kind="publish_approval",
                    prompt="Approve publishing this update?",
                    payload={"tool_call_id": call.id, "args": call.args},
                )
            )

        return ToolDecision.allow()
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

## Deferred Resolution

If a tool waits, the product resolves it through the runtime facade:

```python
await runtime.resolve_deferred_tool_call(
    agent_id,
    thread_id,
    tool_call_id,
    approved=True,
    answer="Approved",
)
```

Do not update tool-call rows yourself or continue the model inline from the
approval endpoint. `resolve_deferred_tool_call` persists the resolution and completes the
parked Temporal activity, allowing the existing workflow to resume normally.
See [pauses and deferred work](pauses-and-resume.md) for details.

## Testing Tools

Unit-test the invocation directly:

```python
result = await EchoInvocation({"message": "hi"}).execute()
assert result.output == {"echo": "hi"}
```

Integration-test through `AgentRuntime` when you need to verify:

- schema exposure
- tool-call persistence
- admission behavior
- waiting/resolution
- continuation after tool results

Use fake LLM providers for deterministic tool calls and outputs.
