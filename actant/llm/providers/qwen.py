"""Qwen adapter for DashScope's OpenAI-compatible endpoint."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_union_param import ChatCompletionToolUnionParam
from openai.types.chat.completion_create_params import CompletionCreateParamsBase

from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers._shared import (
    env_api_key,
    normalize_json_schema,
    sanitize_tool_messages,
    split_tool_content,
)

if TYPE_CHECKING:
    from actant.runtime.hooks import StreamListener

DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
ToolSchema = dict[str, object]
RequestParams = CompletionCreateParamsBase


class QwenProvider:
    """LLMClient implementation for Qwen non-streaming chat completions."""

    def __init__(
        self,
        model_id: str,
        *,
        api_key: str | None = None,
        base_url: str = DASHSCOPE_BASE_URL,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self.model_id = model_id
        self.client = client or openai.AsyncOpenAI(
            api_key=env_api_key("DASHSCOPE_API_KEY", api_key),
            base_url=base_url,
        )

    @staticmethod
    def convert_messages(system: str, messages: Sequence[Message]) -> list[ToolSchema]:
        chat_messages: list[ToolSchema] = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        for message in sanitize_tool_messages(messages):
            if message.role == "user":
                chat_messages.append({"role": "user", "content": cast(object, message.content)})
            elif message.role == "assistant":
                assistant: ToolSchema = {
                    "role": "assistant",
                    "content": cast(object, message.content) if message.content else None,
                }
                if message.tool_calls is not None:
                    assistant["tool_calls"] = [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in message.tool_calls
                    ]
                chat_messages.append(assistant)
            elif message.role == "tool":
                text, _images = split_tool_content(message.content)
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id or "",
                        "content": text,
                    }
                )
        return chat_messages

    @staticmethod
    def convert_tools(tools: list[dict]) -> list[ToolSchema]:
        chat_tools: list[ToolSchema] = []
        for tool in tools or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            chat_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": normalize_json_schema(function.get("parameters", {})),
                    },
                }
            )
        return chat_tools

    def _request_params(
        self, system: str, messages: Sequence[Message], tools: list[dict]
    ) -> RequestParams:
        params: RequestParams = {
            "model": self.model_id,
            "messages": cast(
                list[ChatCompletionMessageParam],
                self.convert_messages(system, messages),
            ),
        }
        converted_tools = self.convert_tools(tools)
        if converted_tools:
            params["tools"] = cast(list[ChatCompletionToolUnionParam], converted_tools)
        return params

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None" = None,
    ) -> Message:
        text = ""
        thought = ""
        tool_call_state: dict[int, dict[str, str]] = {}

        stream = await self.client.chat.completions.create(
            **self._request_params(system, messages, tools),
            stream=True,
            extra_body={"enable_thinking": True},
        )
        async for chunk in stream:
            if listener is not None and listener.cancel_requested():
                await stream.close()
                raise StreamCancelled
            choices = getattr(chunk, "choices", []) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue
            content_delta = cast(str | None, getattr(delta, "content", None))
            if content_delta:
                text += content_delta
                if listener is not None:
                    await listener.on_text_delta(content_delta)
            thinking_delta = cast(str | None, getattr(delta, "reasoning_content", None))
            if thinking_delta:
                thought += thinking_delta
                if listener is not None:
                    await listener.on_thinking_delta(thinking_delta)
            for tool_delta in getattr(delta, "tool_calls", []) or []:
                index = getattr(tool_delta, "index", 0) or 0
                state = tool_call_state.setdefault(index, {"id": "", "name": "", "arguments": ""})
                tc_id = getattr(tool_delta, "id", None)
                if tc_id:
                    state["id"] = tc_id
                fn = getattr(tool_delta, "function", None)
                if fn is not None:
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        state["name"] = fn_name
                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        state["arguments"] += fn_args

        tool_calls: list[ToolCall] = []
        for index in sorted(tool_call_state):
            state = tool_call_state[index]
            tool_calls.append(
                ToolCall(
                    id=state["id"] or f"call_{uuid.uuid4().hex}",
                    function=ToolCallFunction(
                        name=state["name"],
                        arguments=state["arguments"],
                    ),
                )
            )

        return Message(
            role="assistant",
            content=text or None,
            tool_calls=tool_calls or None,
            thought_summary=thought or None,
        )
