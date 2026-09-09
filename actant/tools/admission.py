"""Tool execution admission primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from actant.core import JSONObject
from actant.llm.messages import Message
from actant.tools.base import ToolInvocation, ToolResult


class ToolDecisionKind(StrEnum):
    """Who produces this tool call's result.

    ``EXECUTE`` -- the tool does. ``DENY`` -- admission does, and the reason
    is the result. ``AWAIT_HUMAN`` -- a person does, either by approving so
    the tool then runs, or by answering, where their answer is the result.

    Only ``AWAIT_HUMAN`` suspends anything, which is why the old names were
    confusing: ``BLOCK`` read as "suspend" and was the one that never did.
    """

    EXECUTE = "execute"
    DENY = "deny"
    AWAIT_HUMAN = "await_human"


@dataclass(frozen=True)
class ToolWaitRequest:
    kind: str
    prompt: str
    payload: JSONObject = field(default_factory=dict)

    def to_dict(self) -> JSONObject:
        return {
            "kind": self.kind,
            "prompt": self.prompt,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class ToolResolution:
    approved: bool | None = None
    answer: str = ""
    payload: JSONObject = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JSONObject) -> "ToolResolution":
        approved = data.get("approved")
        answer = data.get("answer", "")
        payload = data.get("payload")
        return cls(
            approved=approved if isinstance(approved, bool) else None,
            answer=answer if isinstance(answer, str) else "",
            payload=payload if isinstance(payload, dict) else {},
        )

    def to_dict(self) -> JSONObject:
        return {
            "approved": self.approved,
            "answer": self.answer,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class ToolDecision:
    kind: ToolDecisionKind
    reason: str = ""
    wait_request: ToolWaitRequest | None = None

    @classmethod
    def execute(cls) -> "ToolDecision":
        return cls(kind=ToolDecisionKind.EXECUTE)

    @classmethod
    def deny(cls, reason: str) -> "ToolDecision":
        return cls(kind=ToolDecisionKind.DENY, reason=reason)

    @classmethod
    def await_human(cls, request: ToolWaitRequest) -> "ToolDecision":
        return cls(
            kind=ToolDecisionKind.AWAIT_HUMAN,
            reason=request.prompt,
            wait_request=request,
        )


class ToolCallView(Protocol):
    id: str
    group_id: str
    agent_id: str
    thread_id: str
    turn_id: str
    turn_index: int
    name: str
    args: JSONObject


class ContextAgentView(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def persona(self) -> str: ...


class TurnContextView(Protocol):
    @property
    def agent(self) -> ContextAgentView: ...

    @property
    def system_prompt(self) -> str: ...

    @property
    def messages(self) -> list[Message]: ...

    @property
    def thread_id(self) -> str: ...

    @property
    def turn_id(self) -> str: ...

    @property
    def turn_index(self) -> int: ...


class ToolCanExecute(Protocol):
    async def can_execute(
        self,
        call: ToolCallView,
        invocation: ToolInvocation,
        context: TurnContextView,
    ) -> ToolDecision: ...


class ToolResolve(Protocol):
    async def on_resolve(
        self,
        call: ToolCallView,
        resolution: ToolResolution,
    ) -> ToolResult: ...
