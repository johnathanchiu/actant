"""Sandboxed tools, ``finish`` with deliverables, and the terminal completion policy,
driven through the real Temporal workflow against in-memory stores."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from actant.agents import AgentDefinition
from actant.core import new_id
from actant.llm.messages import ToolCall, ToolCallFunction
from actant.llm.providers.fake import FakeLLM, FakeResponse
from actant.runtime.completion import RunCompletion
from actant.runtime.stores import InMemoryRuntimeStores
from actant.runtime.temporal.activities import TemporalRuntimeActivities
from actant.runtime.temporal.types import InboundMessage, ThreadInput
from actant.runtime.temporal.workflow import AgentThreadWorkflow
from actant.sandbox import ArtifactRef, LocalSandboxProvider, Sandbox, SandboxSpec
from actant.sandbox.registry import SandboxRegistry
from actant.tools import FinishTool, ToolRegistry, tool
from actant.tools.base import Tool
from actant.tools.calls import ToolCallStatus

_AGENT = "sandboxed_agent"
_THREAD = "thread_1"


@dataclass
class _Sink:
    saved: list[tuple[str, str, bytes, str]] = field(default_factory=list)

    async def save(self, thread_id: str, name: str, data: bytes, mime: str) -> ArtifactRef:
        self.saved.append((thread_id, name, data, mime))
        return ArtifactRef(name, f"mem://{thread_id}/{name}", mime, len(data))


@tool
async def write_file(path: str, text: str, sandbox: Sandbox) -> str:
    """Write a file in the workspace."""
    await sandbox.write(path, text.encode())
    return f"wrote {path}"


def _call(name: str, args: str = "{}") -> ToolCall:
    return ToolCall(id=new_id("tc"), function=ToolCallFunction(name=name, arguments=args))


def _agent(
    llm: FakeLLM, tools: list[Tool], *, mount: Path, completion: str = "reply"
) -> AgentDefinition:
    return AgentDefinition(
        id=_AGENT,
        name="test",
        persona="test persona",
        llm=llm,
        tools=ToolRegistry(tools),
        tool_allowlist={t.name for t in tools},
        sandbox=SandboxSpec(backend="local", mount=str(mount)),
        completion=completion,  # type: ignore[arg-type]
    )


@dataclass
class _Setup:
    stores: InMemoryRuntimeStores
    sink: _Sink
    completions: list[RunCompletion]
    client: object
    task_queue: str

    async def start(self, message: str, max_turns: int = 6):  # type: ignore[no-untyped-def]
        return await cast("object", self.client).start_workflow(  # type: ignore[attr-defined]
            AgentThreadWorkflow.run,
            ThreadInput(_AGENT, _THREAD, max_turns_per_run=max_turns),
            id=f"thread-{uuid.uuid4().hex}",
            task_queue=self.task_queue,
            start_signal="inbound",
            start_signal_args=[InboundMessage(content=message)],
        )

    async def finished(self) -> RunCompletion:
        for _ in range(200):
            if self.completions:
                return self.completions[-1]
            await asyncio.sleep(0.05)
        raise TimeoutError("the run never completed")


async def _run(agent: AgentDefinition, body: Callable[[_Setup], Awaitable[None]]) -> None:
    stores = InMemoryRuntimeStores()
    sink = _Sink()
    completions: list[RunCompletion] = []

    async def on_complete(completion: RunCompletion) -> None:
        completions.append(completion)

    activities = TemporalRuntimeActivities(
        stores=stores,
        agents={agent.id: agent},
        run_completion_handler=on_complete,
        sandboxes=SandboxRegistry({"local": LocalSandboxProvider()}, stores.threads),
        artifact_sink=sink,
    )
    task_queue = f"test-actant-{uuid.uuid4().hex[:8]}"
    async with await WorkflowEnvironment.start_local() as env:
        async with Worker(
            env.client,
            task_queue=task_queue,
            workflows=[AgentThreadWorkflow],
            activities=activities.all,
        ):
            await body(_Setup(stores, sink, completions, env.client, task_queue))


@pytest.mark.asyncio
async def test_a_sandboxed_tool_writes_into_the_threads_sandbox(tmp_path: Path) -> None:
    write = _call("write_file", '{"path": "out/a.txt", "text": "hello"}')
    agent = _agent(
        FakeLLM([FakeResponse(tool_calls=[write]), FakeResponse(text="done")]),
        [write_file],
        mount=tmp_path,
    )

    async def body(s: _Setup) -> None:
        handle = await s.start("go")
        completion = await s.finished()
        assert completion.succeeded
        thread = await s.stores.threads.get(_AGENT, _THREAD)
        assert thread.sandbox_id == str(tmp_path / _THREAD)
        assert (tmp_path / _THREAD / "out" / "a.txt").read_text() == "hello"
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(agent, body)


@pytest.mark.asyncio
async def test_finish_stores_the_deliverables_and_ends_the_run(tmp_path: Path) -> None:
    write = _call("write_file", '{"path": "result.json", "text": "{}"}')
    finish = _call("finish", '{"summary": "traced", "paths": ["result.json"]}')
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[write]),
                FakeResponse(tool_calls=[finish]),
                FakeResponse(text="never called"),
            ]
        ),
        [write_file, FinishTool()],
        mount=tmp_path,
        completion="terminal",
    )

    async def body(s: _Setup) -> None:
        handle = await s.start("go")
        completion = await s.finished()
        assert completion.succeeded and completion.stop_reason is None
        assert [(name, data) for _, name, data, _ in s.sink.saved] == [("result.json", b"{}")]
        assert completion.artifacts == (
            {
                "name": "result.json",
                "uri": f"mem://{_THREAD}/result.json",
                "mime": "application/json",
                "size": 2,
            },
        )
        record = await s.stores.tool_calls.get(finish.id)
        assert record.status == ToolCallStatus.COMPLETED
        metadata = cast(dict[str, object], cast(dict[str, object], record.result)["metadata"])
        assert metadata["terminal"] is True and metadata["artifacts"] == list(completion.artifacts)
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(agent, body)


@pytest.mark.asyncio
async def test_a_missing_deliverable_fails_the_tool_without_ending_the_run(tmp_path: Path) -> None:
    finish = _call("finish", '{"summary": "done", "paths": ["nope.json"]}')
    write = _call("write_file", '{"path": "nope.json", "text": "1"}')
    finish_again = _call("finish", '{"summary": "done", "paths": ["nope.json"]}')
    agent = _agent(
        FakeLLM(
            [
                FakeResponse(tool_calls=[finish]),
                FakeResponse(tool_calls=[write]),
                FakeResponse(tool_calls=[finish_again]),
            ]
        ),
        [write_file, FinishTool()],
        mount=tmp_path,
        completion="terminal",
    )

    async def body(s: _Setup) -> None:
        handle = await s.start("go")
        completion = await s.finished()
        first = await s.stores.tool_calls.get(finish.id)
        assert first.status == ToolCallStatus.FAILED
        assert "could not be read" in str(cast(dict[str, object], first.result)["error"])
        assert completion.succeeded and len(completion.artifacts) == 1
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(agent, body)


@pytest.mark.asyncio
async def test_a_task_agent_that_stops_talking_is_reminded_then_exhausted(tmp_path: Path) -> None:
    agent = _agent(
        FakeLLM([FakeResponse(text="I think I am done."), FakeResponse(text="Really done.")]),
        [FinishTool()],
        mount=tmp_path,
        completion="terminal",
    )

    async def body(s: _Setup) -> None:
        handle = await s.start("go")
        completion = await s.finished()
        assert completion.outcome == "exhausted"
        assert completion.stop_reason == "stopped without finishing"
        messages = await s.stores.messages.list_for_thread(_AGENT, _THREAD)
        roles = [m.role for m in messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert "finish_required" in str(messages[2].content)
        run = await s.stores.runs.get(completion.run_id)
        assert run.stop_reason == "stopped without finishing"
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(agent, body)


@pytest.mark.asyncio
async def test_a_task_agent_reminded_once_can_still_finish(tmp_path: Path) -> None:
    finish = _call("finish", '{"summary": "ok"}')
    agent = _agent(
        FakeLLM([FakeResponse(text="thinking out loud"), FakeResponse(tool_calls=[finish])]),
        [FinishTool()],
        mount=tmp_path,
        completion="terminal",
    )

    async def body(s: _Setup) -> None:
        handle = await s.start("go")
        completion = await s.finished()
        assert completion.succeeded and completion.artifacts == ()
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(agent, body)


@pytest.mark.asyncio
async def test_a_chat_agent_still_ends_on_a_reply(tmp_path: Path) -> None:
    agent = _agent(FakeLLM([FakeResponse(text="hi")]), [], mount=tmp_path)

    async def body(s: _Setup) -> None:
        handle = await s.start("hello")
        completion = await s.finished()
        assert completion.succeeded
        await asyncio.wait_for(handle.result(), timeout=5.0)

    await _run(agent, body)
