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
`exec(argv, cwd=, timeout=, env=)`, `sync`, `close`. Paths are relative to the
thread's root. A class-based tool sets `needs_sandbox = True` and reads
`ctx.sandbox` in `build`. A `CallContext` parameter alone gives a tool its
agent, thread, run and call ids without a sandbox. `ctx.parent_thread_id` is
set when the calling thread is a subagent; `TaskTool` refuses to spawn from
one, so delegation is one level deep.

Backends: `local` (a directory under `mount`, subprocesses; always
registered) and `modal` (`actant[modal]`: a `modal.Sandbox` whose files live in
your object storage, either mounted or restored to local disk and pushed back
with `storage="disk_sync"`). Register providers on the worker:

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

`scrub_env` names variables (service keys from `secrets`) that `exec` removes
from the agent's own commands. It is best effort, not a security boundary.

## Services and tools

A service is a class served next to the sandbox's files; `tools()` exposes a
service to a model. Orchestration code calls services directly through a runner.

Write a service when tools share state or work on many files. Its public
methods are callable; as tools, the schema comes from each signature without
`self`, the description from the docstring. `open` (an optional classmethod)
and `close` are lifecycle, not callable. A plain `def` runs in a
worker thread. An `async def` runs on the host's event loop: blocking or
CPU-heavy work inside one stalls every other call until it awaits, so write
such a method as a plain `def` (or call `asyncio.to_thread` yourself).

```python
from dataclasses import dataclass, field

from actant.sandbox import LocalRunner, SandboxRunner
from actant.tools import tools


@dataclass
class Page:
    text: str
    images: list[str | bytes] = field(default_factory=list)


class Notebook:
    @classmethod
    async def open(cls, title: str) -> "Notebook":
        return cls(title)

    def __init__(self, title: str) -> None:
        self.title = title
        self.lines: list[str] = []

    async def append(self, line: str) -> str:
        """Add a line to the notebook."""
        self.lines.append(line)
        return f"{len(self.lines)} lines"

    async def sketch(self, path: str) -> Page:
        """Return a saved drawing."""
        return Page("the sketch", [path])


# In-process: the same tools, run on an instance you hold.
local_tools = tools(Notebook, LocalRunner(Notebook("scratch")))

# In the thread's sandbox: the spec names the class, the host serves it there.
agent = AgentDefinition(
    ...,
    tools=ToolRegistry(tools(Notebook, SandboxRunner("notebook", init={"title": "scratch"}))),
    sandbox=SandboxSpec(
        backend="modal",
        services={
            "notebook": "myproduct.notebook:Notebook",
            # Calls the product makes itself, kept off the model's tool list.
            "pipeline": "myproduct.notebook:Pipeline",
        },
    ),
)
```

A method returns a `str`, an object with `.text` and `.images` (image file
paths or bytes, sent as image content blocks), or any JSON value. An exception
becomes a failed result with the traceback tail. Arguments are validated
against the signature, so a parameter typed as a pydantic model arrives as
that model. `RemoteRunner(endpoint, service, key, init)` calls a host you
reach yourself, and `call_host(endpoint, service, method, args, key=...)`
makes one call (with `await sandbox.endpoint()`). Every runner returns the same
`CallResponse` (`text`, `images`, `error`, `storage`); `tools()` turns it into a
`ToolResult`. From orchestration code, call a runner directly:
`await SandboxRunner("pipeline").call("stage", {}, key=thread_id, sandbox=sandbox)`.

A sandbox serves several named services from one host (`python -m
actant.sandbox.entry '{"host": {"services": {"name": "pkg.mod:Class"}}}'`, an
`actant.sandbox.protocol.EntryConfig`). Give the model its
tools on one class and put the product's own calls on another, instead of
filtering methods. The host keeps one instance per service and thread and runs
calls concurrently. Services do not share instances: two classes over the same
state build it from the same `init` (the sandbox's files, or an object their
`open` looks up). Inside it, tools that
start scripts should pass `env=actant.sandbox.host.script_env()` so the
scripts do not inherit `scrub_env`. On Modal the host is the sandbox
entrypoint, readiness is a TCP probe on `service_port`, `disk_sync` storage is
restored before it listens and pushed after calls and periodically, and requests go through a
Modal connect token (cached, re-minted on a 401), so no port is public.
Connections are kept alive and pooled, so a call costs one round trip.
`Sandbox.close` asks the host to shut down: it calls each instance's `close`,
pushes `disk_sync` storage once more, and exits (SIGTERM, SIGINT and SIGHUP do
the same). Modal's `terminate`, `timeout` and `idle_timeout` kill the container
outright, so `close` does not run on those paths. The image needs actant and
the services' package installed.

Storage pushes never fail or stall a run. The host pushes after calls and every
`SandboxSpec.sync_interval_s` (default 60) while completed calls are unpushed,
one push at a time; a push running longer than `sync_timeout_s` (default 300)
is killed and the next still runs. A failure is logged and reported: every call
through a host that pushes carries `ToolResult.metadata[MetadataKey.STORAGE]`, the
JSON of an `actant.sandbox.StorageStatus` (read it with
`StorageStatus.model_validate`):

| Field | Meaning |
| --- | --- |
| `last_attempt_at` | Unix time the latest push started, or `None` |
| `last_success_at` | Unix time the latest successful push started, or `None` |
| `last_error` | Short reason the latest push failed; `None` once one succeeds |
| `consecutive_failures` | Failed pushes since the last success |
| `pending` | Completed calls not yet covered by a successful push |
| `image_error` | Why an image in this response went as bytes instead of a URL, or `None` |

Warn when `consecutive_failures` is non-zero. The final push on shutdown is
bounded the same way, so shutdown finishes even when storage is unreachable;
`Sandbox.sync` and `close` never wait without bound. A restore that fails or
exceeds 30 minutes fails startup. Restored files take their objects' mtimes,
so a push uploads only files changed since.

Images a service returns reach the model as presigned URLs when the host can
reach a bucket: `disk_sync` on Modal, or a `LocalSandboxProvider(images=ImageBucket(...))`.
The host uploads each image at once, content-addressed under `actant-images/<thread>/`
(`image_prefix` on `ModalSandboxProvider`, `prefix` on `ImageBucket`), apart from the
thread's files so no push touches it. It presigns the object for
`SandboxSpec.image_url_ttl_s` (default 6 h, at most 7 days) and returns `Image.source` as
a `UrlSource(url, expires_at)` instead of an `InlineSource(data_b64)`. The model provider
fetches the URL, so each request stays small however many images a run re-sends.
`image_url_ttl_s=None` always sends bytes.

URLs are signed for `public_endpoint_url`, which the model provider must reach, so it is
required: `ImageBucket` cannot be built without it, and a `disk_sync` spec with services
fails `ModalSandboxProvider.open` without it (unless `image_url_ttl_s=None`). For R2 or S3
it is the bucket endpoint itself; for a MinIO it is a public tunnel. URLs signed with
temporary credentials stop working when those do.

A failed upload or presign never fails the call: that image goes inline and
`StorageStatus.image_error` says why. Upload plus presign of one image is bounded by
`SandboxSpec.image_upload_timeout_s` (default 10 s), so an unreachable bucket costs at
most that per image before the bytes go instead; after one failure, images of the same
response not yet started skip the upload.

Nothing deletes uploaded images. Expire them with a lifecycle rule on the prefix, longer
than `image_url_ttl_s` (a re-uploaded image resets its age):

```bash
mc ilm rule add --expire-days 8 --prefix actant-images/ local/my-bucket    # MinIO
aws s3api put-bucket-lifecycle-configuration --bucket my-bucket --lifecycle-configuration \
  '{"Rules": [{"ID": "actant-images", "Status": "Enabled",
    "Filter": {"Prefix": "actant-images/"}, "Expiration": {"Days": 8}}]}'  # S3, R2
```

Local setup: run the local backend against a MinIO behind a tunnel, and give the tunnel's
public URL as `public_endpoint_url`. Uploads go straight to MinIO; the tunnel forwards the
public `Host` header, which the URL signs.

```python
# cloudflared tunnel --url http://127.0.0.1:9000   (or: ngrok http 9000)
LocalSandboxProvider(
    Path("/srv/agent-workspaces"),
    images=ImageBucket(
        "my-bucket",
        public_endpoint_url="https://<name>.trycloudflare.com",
        endpoint_url="http://127.0.0.1:9000",
    ),
)
```

The host process reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_REGION`
(`us-east-1` for MinIO) from its environment and needs s5cmd on `PATH`.

Code that calls a runner itself turns an image into a content block with
`actant.tools.image_block(image)` (a URL source when present, else base64). A stored
message keeps its URLs; on replay the LLM adapters replace an image whose URL has
expired (or will within two minutes) with a text note, since a provider rejects the
whole request when a fetch fails.

`SandboxSpec.seed` (a bucket key prefix ending in `/`) starts a new thread from a
template. When the thread's prefix is empty, the sandbox pulls the seed while
s5cmd copies it into the thread's prefix, and startup waits for both, so no push
races the copy; a failed copy fails startup. A finished copy writes a marker
object beside the prefix (`<thread>.actant-seeded`, never pulled or pushed). A
thread whose prefix has files restores those and ignores the seed, unless the
marker is missing: then the copy was cut off, and startup fails instead of
serving a partial workspace (delete the prefix to seed it again).

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
