"""One activity-scoped publisher for lifecycle and provider streaming events."""

from typing import cast
from actant.core import JSONObject, JSONValue
from actant.llm.messages import Message
from actant.tools.base import ToolResult
from actant.runtime.events.publisher import EventSink
from actant.runtime.events.streaming import StreamListener


class RuntimeEvents(StreamListener):
    def __init__(self, thread_id: str, sink: EventSink) -> None:
        self.thread_id = thread_id
        self.sink = sink

    async def emit(self, event_type: str, data: JSONObject) -> None:
        await self.sink.publish(
            f"thread:{self.thread_id}",
            {
                "type": event_type,
                "thread_id": self.thread_id,
                "data": data,
            },
        )

    async def on_user_message(self, content: str | list[dict[str, object]]) -> None:
        await self.emit("user_message", {"content": cast(JSONValue, content)})

    async def on_assistant_message(self, message: Message) -> None:
        tool_calls = cast(list[JSONValue], [tc.to_dict() for tc in (message.tool_calls or [])])
        payload: JSONObject = {
            "content": cast(JSONValue, message.content),
            "thought_summary": message.thought_summary,
            "tool_calls": tool_calls,
        }
        await self.emit("assistant_message", payload)

    async def on_turn_start(self, turn: int, turn_id: str | None = None) -> None:
        payload: JSONObject = {"turn": turn}
        if turn_id is not None:
            payload["turn_id"] = turn_id
            payload["turn_uid"] = turn_id
        await self.emit("turn_start", payload)

    async def on_tool_call(self, tool_call_id: str, name: str, args: JSONObject) -> None:
        await self.emit("tool_call", {"tool_call_id": tool_call_id, "name": name, "args": args})

    async def on_tool_result(
        self, tool_call_id: str, result: ToolResult, turn_id: str | None = None
    ) -> None:
        payload: JSONObject = {
            "tool_call_id": tool_call_id,
            "output": str(result.output) if result.output is not None else None,
            "error": result.error,
            "result": cast(JSONValue, {"result": result.output, **result.to_dict()}),
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
            payload["turn_uid"] = turn_id
        await self.emit("tool_result", payload)

    async def on_tool_waiting(
        self,
        tool_call_id: str,
        prompt: str,
        turn_id: str | None = None,
        wait_request: JSONObject | None = None,
    ) -> None:
        payload: JSONObject = {"tool_call_id": tool_call_id, "prompt": prompt}
        if wait_request is not None:
            payload["wait_request"] = wait_request
            kind = wait_request.get("kind")
            payload["wait_kind"] = kind if isinstance(kind, str) else None
            wait_payload = wait_request.get("payload")
            payload["wait_payload"] = wait_payload if isinstance(wait_payload, dict) else {}
        if turn_id is not None:
            payload["turn_id"] = turn_id
            payload["turn_uid"] = turn_id
        await self.emit("tool_waiting", payload)

    async def on_tool_resolved(
        self, tool_call_id: str, result: ToolResult, turn_id: str | None = None
    ) -> None:
        payload: JSONObject = {
            "tool_call_id": tool_call_id,
            "output": str(result.output) if result.output is not None else None,
            "result": cast(JSONValue, {"result": result.output, **result.to_dict()}),
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
            payload["turn_uid"] = turn_id
        await self.emit("tool_resolved", payload)

    async def on_complete(self, success: bool, reason: str, message: str) -> None:
        await self.emit(
            "complete",
            {"success": success, "reason": reason, "message": message},
        )

    async def on_error(self, error: Exception) -> None:
        await self.emit("error", {"message": str(error)})

    async def on_text_delta(self, delta: str) -> None:
        await self.emit("text_delta", {"delta": delta})

    async def on_thinking_delta(self, delta: str) -> None:
        await self.emit("thinking_delta", {"delta": delta})

    async def on_tool_call_start(self, tool_call_id: str, name: str) -> None:
        await self.emit("tool_call_start", {"tool_call_id": tool_call_id, "name": name})

    async def on_tool_call_args_delta(self, tool_call_id: str, delta: str) -> None:
        await self.emit("tool_call_args_delta", {"tool_call_id": tool_call_id, "delta": delta})

    async def on_tool_call_args_complete(self, tool_call_id: str) -> None:
        await self.emit("tool_call_args_complete", {"tool_call_id": tool_call_id})

    async def on_usage(self, response_id: str, model: str, usage: JSONObject, status: str) -> None:
        await self.emit(
            "model_usage",
            {"response_id": response_id, "model": model, "usage": usage, "status": status},
        )

    async def on_stream_reset(self) -> None:
        await self.emit("stream_reset", {})
