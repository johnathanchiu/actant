"""HTTP routes for the demo. Thin wrappers over DemoCoordinator.

No business logic here. State changes funnel through the coordinator
so all resolves (user-driven AND sub-thread-completion driven) go
through one reconciled path.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from actant.runtime.exceptions import ToolResolutionStaleError

from app.agents import AGENT_ID, RESEARCHER_AGENT_ID
from app.coordinator import DemoCoordinator


router = APIRouter(prefix="/api")


class SendMessageBody(BaseModel):
    content: str


class ResolveToolBody(BaseModel):
    approved: bool | None = None
    answer: str = ""
    payload: dict[str, Any] | None = None


def get_coordinator(request: Request) -> DemoCoordinator:
    coord: DemoCoordinator | None = getattr(request.app.state, "coordinator", None)
    if coord is None:
        raise HTTPException(status_code=503, detail="Demo coordinator not ready")
    return coord


@router.get("/agent")
async def get_agent(request: Request) -> dict[str, Any]:
    coord = get_coordinator(request)
    # Find the main agent in the registered set.
    main = coord.runtime.agents[AGENT_ID]
    return {
        "id": main.id,
        "name": main.name,
        "persona": main.persona,
        "model": coord.model_id,
        "tools": [name for name in _tool_names(main)],
    }


def _tool_names(agent) -> list[str]:
    # The agent's tools registry doesn't expose `.all()`; reach for the
    # private dict for the demo (it's stable in the actant version we
    # pin).
    return list(getattr(agent.tools, "_tools", {}).keys())


@router.get("/threads")
async def list_threads(request: Request) -> list[dict[str, Any]]:
    coord = get_coordinator(request)
    threads = await coord.stores.threads.list_for_agent(AGENT_ID)
    result: list[dict[str, Any]] = []
    for t in threads:
        messages = await coord.stores.messages.list_for_thread(AGENT_ID, t.id)
        preview = _preview_for_messages(messages)
        result.append(
            {
                "id": t.id,
                "agent_id": t.agent_id,
                "status": t.status.value if hasattr(t.status, "value") else str(t.status),
                "turn_count": t.turn_count,
                "message_count": len(messages),
                "preview": preview,
            }
        )
    return result


@router.get("/threads/{thread_id}/messages")
async def get_messages(thread_id: str, request: Request) -> list[dict[str, Any]]:
    coord = get_coordinator(request)
    # The /messages endpoint returns messages for whichever agent owns
    # the thread (main vs researcher). For sub-thread requests, look
    # under researcher.
    for agent_id in (AGENT_ID, RESEARCHER_AGENT_ID):
        messages = await coord.stores.messages.list_for_thread(agent_id, thread_id)
        if messages:
            return [m.to_dict() for m in messages]
    return []


@router.get("/threads/{thread_id}/waiting_tool_calls")
async def get_waiting_tool_calls(thread_id: str, request: Request) -> list[dict[str, Any]]:
    coord = get_coordinator(request)
    # Derive agent_id the same way resolve_user_deferred does — a
    # sub-thread's waiting tool calls belong to the sub-agent (e.g.
    # researcher's ask_user), not the main agent. Without this, the
    # FE refresh path loses sub-agent deferred panels.
    link = coord.registry.get(thread_id)
    agent_id = link.sub_agent_id if link is not None else AGENT_ID
    records = await coord.stores.tool_calls.get_open_for_thread(agent_id, thread_id)
    return [
        {
            "tool_call_id": r.id,
            "name": r.name,
            "args": r.args,
            "prompt": r.prompt,
            "wait_request": r.wait_request,
        }
        for r in records
        if r.status.value == "waiting"
    ]


@router.get("/threads/{thread_id}/sub_threads")
async def list_sub_threads(thread_id: str, request: Request) -> list[dict[str, Any]]:
    coord = get_coordinator(request)
    sub_threads = await coord.stores.threads.list_for_agent(RESEARCHER_AGENT_ID)
    return [
        {
            "sub_thread_id": t.id,
            "parent_thread_id": t.parent_thread_id,
            "parent_tool_call_id": t.parent_tool_call_id,
        }
        for t in sub_threads
        if t.parent_thread_id == thread_id and t.parent_tool_call_id
    ]


@router.post("/threads/{thread_id}/messages", status_code=202)
async def send_message(
    thread_id: str, body: SendMessageBody, request: Request
) -> dict[str, str]:
    coord = get_coordinator(request)
    run_id = await coord.runtime.send_message(AGENT_ID, thread_id, body.content)
    return {"run_id": run_id, "thread_id": thread_id}


@router.get("/threads/{thread_id}/state")
async def get_state(thread_id: str, request: Request) -> Any:
    coord = get_coordinator(request)
    state = await coord.runtime.get_state(AGENT_ID, thread_id)
    return state


@router.delete("/threads/{thread_id}", status_code=204)
async def cancel_thread(thread_id: str, request: Request) -> None:
    coord = get_coordinator(request)
    await coord.runtime.cancel_thread(AGENT_ID, thread_id)


@router.post(
    "/threads/{thread_id}/tool_calls/{tool_call_id}/resolve", status_code=204
)
async def resolve_tool(
    thread_id: str,
    tool_call_id: str,
    body: ResolveToolBody,
    request: Request,
) -> None:
    coord = get_coordinator(request)
    try:
        await coord.resolve_user_deferred(
            thread_id=thread_id,
            tool_call_id=tool_call_id,
            approved=body.approved,
            answer=body.answer,
            payload=body.payload,
        )
    except ToolResolutionStaleError as exc:
        # The store has been reconciled to FAILED — there's no parked
        # activity to resolve. Tell the client they should refresh.
        raise HTTPException(
            status_code=409,
            detail={
                "code": "stale_tool_call",
                "tool_call_id": exc.tool_call_id,
                "reason": exc.reason,
            },
        ) from exc


@router.get("/threads/{thread_id}/events")
async def stream_events(thread_id: str, request: Request) -> EventSourceResponse:
    coord = get_coordinator(request)
    channel = f"thread:{thread_id}"

    async def event_source() -> Any:
        yield {"comment": "connected"}
        subscription = coord.stores.publisher.subscribe(channel)
        try:
            async for event in subscription:
                if await request.is_disconnected():
                    break
                event_type = str(event.get("type") or "message")
                yield {"event": event_type, "data": json.dumps(event)}
        except asyncio.CancelledError:
            raise

    return EventSourceResponse(event_source())


def _preview_for_messages(messages: list[Any]) -> str:
    for message in messages:
        if getattr(message, "role", None) != "user":
            continue
        content = message.content
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = " ".join(
                str(block.get("text", "")).strip()
                for block in content
                if isinstance(block, dict)
            ).strip()
        else:
            continue
        if text:
            return text[:80] + ("…" if len(text) > 80 else "")
    return ""
