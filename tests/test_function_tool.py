"""Tests for the function-backed tool surface."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Annotated, cast

import pytest
from pydantic import Field

from actant.core import JSONObject
from actant.tools import (
    FunctionTool,
    ToolDecision,
    ToolResolution,
    ToolResult,
    ToolWaitRequest,
    tool,
)
from actant.tools.admission import ToolCallView, ToolDecisionKind, TurnContextView


@dataclass
class _Call:
    args: JSONObject
    id: str = "tool-call"
    group_id: str = "group"
    agent_id: str = "agent"
    thread_id: str = "thread"
    turn_id: str = "turn"
    turn_index: int = 0
    name: str = "weather"


def _call(args: JSONObject) -> ToolCallView:
    return cast(ToolCallView, _Call(args))


@pytest.mark.asyncio
async def test_tool_builds_schema_validates_and_wraps_native_result() -> None:
    @tool
    async def weather(
        city: Annotated[str, Field(description="City to check")],
        days: int = 1,
    ) -> dict[str, object]:
        """Get a weather forecast."""
        return {"city": city, "days": days}

    assert weather.name == "weather"
    function_schema = cast(dict[str, object], weather.schema["function"])
    assert function_schema["description"] == "Get a weather forecast."
    parameters = cast(dict[str, object], function_schema["parameters"])
    assert parameters["required"] == ["city"]
    assert parameters["additionalProperties"] is False

    invocation = await weather.build({"city": "Paris", "days": "2"})
    result = await invocation.execute()
    assert result.output == {"city": "Paris", "days": 2}

    with pytest.raises(ValueError, match="Invalid arguments for weather"):
        await weather.build({"unknown": True})


@pytest.mark.asyncio
async def test_sync_tool_runs_off_the_event_loop_thread() -> None:
    caller_thread = threading.get_ident()

    @tool
    def thread_id() -> int:
        """Return the execution thread."""
        return threading.get_ident()

    result = await (await thread_id.build({})).execute()
    assert result.output != caller_thread


@pytest.mark.asyncio
async def test_explicit_tool_result_is_preserved() -> None:
    @tool
    async def fail_cleanly(reason: str) -> ToolResult:
        """Return a structured failure."""
        return ToolResult.fail(reason, retryable=False)

    result = await (await fail_cleanly.build({"reason": "nope"})).execute()
    assert result.error == "nope"
    assert result.metadata == {"retryable": False}


@pytest.mark.asyncio
async def test_approval_waits_then_executes_only_when_approved() -> None:
    executions: list[str] = []

    @tool(approval=lambda args: f"Check weather for {args['city']}?")
    async def weather(city: str) -> str:
        """Get weather."""
        executions.append(city)
        return f"sunny in {city}"

    call = _call({"city": "Paris"})
    decision = await weather.can_execute(
        call,
        await weather.build(call.args),
        cast(TurnContextView, object()),
    )
    assert decision.kind is ToolDecisionKind.WAIT
    assert decision.wait_request is not None
    assert decision.wait_request.kind == "approval"
    assert decision.wait_request.prompt == "Check weather for Paris?"
    assert executions == []

    denied = await weather.on_resolve(_call({"city": "Paris"}), ToolResolution(approved=False))
    assert denied.error == "Tool call was not approved"
    assert executions == []

    approved = await weather.on_resolve(_call({"city": "Paris"}), ToolResolution(approved=True))
    assert approved.output == "sunny in Paris"
    assert executions == ["Paris"]


@pytest.mark.asyncio
async def test_custom_admission_and_resolution_callbacks() -> None:
    async def admit(args: dict[str, object]) -> ToolDecision:
        return ToolDecision.wait(
            ToolWaitRequest(kind="question", prompt=f"Units for {args['city']}?")
        )

    @tool(
        admission=admit,
        resolve=lambda args, resolution: {
            "city": args["city"],
            "units": resolution.answer,
        },
    )
    async def weather(city: str) -> str:
        """Get weather."""
        return city

    call = _call({"city": "Paris"})
    decision = await weather.can_execute(
        call,
        await weather.build(call.args),
        cast(TurnContextView, object()),
    )
    assert decision.kind is ToolDecisionKind.WAIT
    result = await weather.on_resolve(_call({"city": "Paris"}), ToolResolution(answer="celsius"))
    assert result.output == {"city": "Paris", "units": "celsius"}


def test_tool_rejects_ambiguous_or_untyped_definitions() -> None:
    async def candidate(value: str) -> str:
        return value

    with pytest.raises(TypeError, match="either `approval` or `admission`"):
        FunctionTool(
            candidate,
            approval="Approve?",
            admission=lambda _args: ToolDecision.allow(),
        )

    with pytest.raises(TypeError, match="requires a type annotation"):

        @tool
        def untyped(value):
            return value
