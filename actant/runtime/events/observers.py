"""Observational callbacks cannot change execution outcomes.

Scope is immutable per activity, not inferred from whichever turn emitted last.
Cancellation still propagates; durable completion callbacks do not use these adapters.
"""

from __future__ import annotations
import logging
from actant.core import JSONObject
from actant.llm.messages import Message
from actant.tools.base import ToolResult
from actant.runtime.events.lifecycle import AgentThreadHooks
from actant.runtime.events.streaming import StreamListener
from actant.runtime.events.publisher import EventSink

log = logging.getLogger(__name__)


class ScopedEventSink:
    def __init__(
        self, sink: EventSink, agent_id: str, run_id: str | None, turn_id: str | None
    ) -> None:
        self.sink = sink
        self.identity: JSONObject = {"agent_id": agent_id}
        if run_id is not None:
            self.identity["run_id"] = run_id
        if turn_id is not None:
            self.identity.update({"turn_id": turn_id, "turn_uid": turn_id})

    async def publish(self, channel: str, event: JSONObject) -> None:
        data = event.get("data")
        payload = {**event, "data": {**self.identity, **(data if isinstance(data, dict) else {})}}
        try:
            await self.sink.publish(channel, payload)
        except Exception:
            log.exception("runtime event publication failed")


class ObservedHooks(AgentThreadHooks):
    def __init__(self, observer: AgentThreadHooks) -> None:
        self.observer = observer

    async def on_user_message(self, content: str | list[dict[str, object]]) -> None:
        try:
            await self.observer.on_user_message(content)
        except Exception:
            log.exception("runtime observer failed")

    async def on_assistant_message(self, message: Message) -> None:
        try:
            await self.observer.on_assistant_message(message)
        except Exception:
            log.exception("runtime observer failed")

    async def on_turn_start(self, turn: int, turn_id: str | None = None) -> None:
        try:
            await self.observer.on_turn_start(turn, turn_id)
        except Exception:
            log.exception("runtime observer failed")

    async def on_tool_call(self, tool_call_id: str, name: str, args: JSONObject) -> None:
        try:
            await self.observer.on_tool_call(tool_call_id, name, args)
        except Exception:
            log.exception("runtime observer failed")

    async def on_tool_result(
        self, tool_call_id: str, result: ToolResult, turn_id: str | None = None
    ) -> None:
        try:
            await self.observer.on_tool_result(tool_call_id, result, turn_id)
        except Exception:
            log.exception("runtime observer failed")

    async def on_tool_waiting(
        self,
        tool_call_id: str,
        prompt: str,
        turn_id: str | None = None,
        wait_request: JSONObject | None = None,
    ) -> None:
        try:
            await self.observer.on_tool_waiting(tool_call_id, prompt, turn_id, wait_request)
        except Exception:
            log.exception("runtime observer failed")

    async def on_tool_resolved(
        self, tool_call_id: str, result: ToolResult, turn_id: str | None = None
    ) -> None:
        try:
            await self.observer.on_tool_resolved(tool_call_id, result, turn_id)
        except Exception:
            log.exception("runtime observer failed")

    async def on_complete(self, success: bool, reason: str, message: str) -> None:
        try:
            await self.observer.on_complete(success, reason, message)
        except Exception:
            log.exception("runtime observer failed")

    async def on_error(self, error: Exception) -> None:
        try:
            await self.observer.on_error(error)
        except Exception:
            log.exception("runtime observer failed")


class ObservedStream(StreamListener):
    def __init__(self, observer: StreamListener) -> None:
        self.observer = observer

    async def on_text_delta(self, delta: str) -> None:
        try:
            await self.observer.on_text_delta(delta)
        except Exception:
            log.exception("runtime observer failed")

    async def on_thinking_delta(self, delta: str) -> None:
        try:
            await self.observer.on_thinking_delta(delta)
        except Exception:
            log.exception("runtime observer failed")

    async def on_tool_call_start(self, tool_call_id: str, name: str) -> None:
        try:
            await self.observer.on_tool_call_start(tool_call_id, name)
        except Exception:
            log.exception("runtime observer failed")

    async def on_tool_call_args_delta(self, tool_call_id: str, delta: str) -> None:
        try:
            await self.observer.on_tool_call_args_delta(tool_call_id, delta)
        except Exception:
            log.exception("runtime observer failed")

    async def on_tool_call_args_complete(self, tool_call_id: str) -> None:
        try:
            await self.observer.on_tool_call_args_complete(tool_call_id)
        except Exception:
            log.exception("runtime observer failed")

    async def on_usage(self, response_id: str, model: str, usage: JSONObject, status: str) -> None:
        try:
            await self.observer.on_usage(response_id, model, usage, status)
        except Exception:
            log.exception("runtime observer failed")

    async def on_stream_reset(self) -> None:
        try:
            await self.observer.on_stream_reset()
        except Exception:
            log.exception("runtime observer failed")

    def cancel_requested(self) -> bool:
        return self.observer.cancel_requested()
