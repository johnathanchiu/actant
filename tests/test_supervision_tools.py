"""Supervising a subagent that is already running.

Delegation stopped being a blocking call, so a parent now holds a handle
rather than a suspended tool call. These are what it does with the handle,
and the point of them is that a parent can hold several at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from actant.core import JSONObject
from actant.tools.supervise import supervision_tools


@dataclass
class _Supervisor:
    """A host's supervisor, as far as these tools use it."""

    states: dict[str, JSONObject] = field(default_factory=dict)
    sent: list[tuple[str, str]] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)

    async def status(self, thread_id: str) -> JSONObject:
        return self.states.get(thread_id, {"status": "unknown"})

    async def send(self, thread_id: str, message: str) -> None:
        self.sent.append((thread_id, message))

    async def stop(self, thread_id: str) -> None:
        self.stopped.append(thread_id)


def _by_name(supervisor: _Supervisor) -> dict[str, object]:
    return {tool.name: tool for tool in supervision_tools(supervisor)}


async def _run(tool: object, **args: object) -> JSONObject:
    invocation = await tool.build(args)  # type: ignore[attr-defined]
    result = await invocation.execute()
    assert result.error is None, result.error
    return result.output  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_check_reports_what_the_host_says() -> None:
    supervisor = _Supervisor(states={"sub_1": {"status": "running", "turns": 3}})
    tools = _by_name(supervisor)

    assert await _run(tools["check_subagent"], thread_id="sub_1") == {
        "status": "running",
        "turns": 3,
    }


@pytest.mark.asyncio
async def test_a_parent_can_hold_several_at_once() -> None:
    """The whole point of a handle: supervising more than one child."""
    supervisor = _Supervisor(
        states={"sub_1": {"status": "running"}, "sub_2": {"status": "completed"}}
    )
    check = _by_name(supervisor)["check_subagent"]

    assert (await _run(check, thread_id="sub_1"))["status"] == "running"
    assert (await _run(check, thread_id="sub_2"))["status"] == "completed"


@pytest.mark.asyncio
async def test_message_reaches_the_named_subagent() -> None:
    supervisor = _Supervisor()
    tools = _by_name(supervisor)

    await _run(tools["message_subagent"], thread_id="sub_1", message="also add a handle")

    assert supervisor.sent == [("sub_1", "also add a handle")]


@pytest.mark.asyncio
async def test_stop_abandons_work_nobody_wants() -> None:
    supervisor = _Supervisor()
    tools = _by_name(supervisor)

    await _run(tools["stop_subagent"], thread_id="sub_1")

    assert supervisor.stopped == ["sub_1"]


@pytest.mark.asyncio
async def test_a_failing_host_surfaces_rather_than_reporting_success() -> None:
    """A supervisor that raises must not be mistaken for a healthy child.

    The exception propagates out of the invocation; the activity boundary
    (``_execute_tool``) is what turns it into a failed tool result. What
    matters here is that it is not swallowed into a plausible-looking
    status the model would then trust.
    """

    class _Broken(_Supervisor):
        async def status(self, thread_id: str) -> JSONObject:
            raise RuntimeError("temporal is down")

    tool = _by_name(_Broken())["check_subagent"]
    invocation = await tool.build({"thread_id": "sub_1"})  # type: ignore[attr-defined]

    with pytest.raises(RuntimeError, match="temporal is down"):
        await invocation.execute()
