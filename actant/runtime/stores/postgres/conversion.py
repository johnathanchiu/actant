"""Pure row/domain conversion for the SQLAlchemy projection stores."""

from __future__ import annotations

import json
from typing import cast

from actant.core import JSONObject
from actant.llm.messages import Message, Role
from actant.runtime.session import parts_to_messages
from actant.runtime.stores.postgres.models import (
    ActantMessageModel,
    ActantMessagePartModel,
    ActantRunModel,
    ActantThreadModel,
    ActantToolCallModel,
)
from actant.runtime.types.session import MessagePart, PartKind, WaitStatus
from actant.runtime.types.threads import AgentRun, AgentThread, RunStatus, ThreadStatus
from actant.tools.calls import ToolCallRecord, ToolCallStatus


def thread_from_row(row: ActantThreadModel) -> AgentThread:
    return AgentThread(
        id=row.thread_id,
        agent_id=row.agent_id,
        status=ThreadStatus(row.status),
        turn_count=row.turn_count,
        active_run_id=row.active_run_id,
        parent_thread_id=row.parent_thread_id,
        parent_turn_id=row.parent_turn_id,
        parent_tool_call_id=row.parent_tool_call_id,
    )


def run_from_row(row: ActantRunModel) -> AgentRun:
    return AgentRun(
        id=row.run_id,
        agent_id=row.agent_id,
        thread_id=row.thread_id,
        status=RunStatus(row.status),
        turn_count=row.turn_count,
        max_turns=row.max_turns,
    )


def message_from_header(row: ActantMessageModel) -> Message:
    """Reassemble one provider-neutral message from a header and its parts."""
    role = row.role
    parts = [message_part_from_row(part) for part in row.parts]

    if role == "tool":
        for part in parts:
            if part.kind is PartKind.TOOL_RESULT:
                content: str | list[dict[str, object]]
                if part.content_blocks:
                    content = part.content_blocks
                elif part.result is not None:
                    content = json.dumps(part.result)
                else:
                    content = part.content or ""
                return Message(
                    role="tool",
                    content=content,
                    tool_call_id=part.tool_call_id,
                    name=part.tool_name,
                )
        return Message(role=cast(Role, role), content="")

    messages = parts_to_messages(parts)
    if messages:
        return messages[0]
    return Message(role=cast(Role, role), content="")


def message_part_row(
    message_id: str, part_index: int, part: MessagePart
) -> ActantMessagePartModel:
    return ActantMessagePartModel(
        message_id=message_id,
        part_index=part_index,
        kind=part.kind.value,
        content=part.content,
        content_blocks=part.content_blocks,
        signature=part.signature,
        reasoning_items=part.reasoning_items,
        tool_call_id=part.tool_call_id,
        tool_name=part.tool_name,
        args=part.args,
        result=part.result,
        wait_status=part.wait_status.value if part.wait_status is not None else None,
    )


def message_part_from_row(row: ActantMessagePartModel) -> MessagePart:
    return MessagePart(
        kind=PartKind(row.kind),
        content=row.content,
        content_blocks=row.content_blocks,
        signature=row.signature,
        reasoning_items=row.reasoning_items,
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        args=cast(JSONObject, row.args) if row.args is not None else None,
        result=row.result,
        wait_status=WaitStatus(row.wait_status) if row.wait_status is not None else None,
    )


def tool_result_part_row(
    message_id: str, tool_call_id: str, name: str, result: object
) -> ActantMessagePartModel:
    """Build a tool-result part row from a tool's provider-neutral result."""
    blocks: list[dict[str, object]] | None = None
    if isinstance(result, dict):
        candidate = result.get("content_blocks")
        if isinstance(candidate, list):
            normalized = [block for block in candidate if isinstance(block, dict)]
            blocks = normalized or None
    return ActantMessagePartModel(
        message_id=message_id,
        part_index=0,
        kind=PartKind.TOOL_RESULT.value,
        content_blocks=blocks,
        result=result if isinstance(result, dict) else {"value": result},
        tool_call_id=tool_call_id,
        tool_name=name,
    )


def tool_result_content(result: object) -> str | list[dict[str, object]]:
    if isinstance(result, dict):
        candidate = result.get("content_blocks")
        if isinstance(candidate, list):
            normalized = [block for block in candidate if isinstance(block, dict)]
            if normalized:
                return normalized
    return json.dumps(result)


def tool_call_from_row(row: ActantToolCallModel) -> ToolCallRecord:
    return ToolCallRecord(
        id=row.tool_call_id,
        group_id=row.group_id,
        run_id=row.run_id,
        agent_id=row.agent_id,
        thread_id=row.thread_id,
        turn_id=row.turn_id,
        turn_index=row.turn_index,
        name=row.name,
        args=cast(JSONObject, row.args),
        status=ToolCallStatus(row.status),
        prompt=row.prompt,
        wait_request=cast(JSONObject | None, row.wait_request),
        result=row.result,
    )
