"""Gemini adapter."""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from google import genai  # pyright: ignore[reportAttributeAccessIssue]
from google.genai import types

from actant.llm.errors import StreamCancelled
from actant.llm.messages import Message, ToolCall, ToolCallFunction
from actant.llm.providers._shared import env_api_key, sanitize_tool_messages

if TYPE_CHECKING:
    from actant.runtime.hooks import StreamListener


def dereference_schema(schema: "ToolSchema") -> "ToolSchema":
    defs = schema.get("$defs", {})
    if not isinstance(defs, dict) or not defs:
        return schema

    def resolve(node: object) -> object:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                ref_name = ref.rsplit("/", 1)[-1]
                if ref_name in defs:
                    return resolve(defs[ref_name])
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return cast("ToolSchema", resolve(schema))


def strip_unsupported_schema_keys(schema: object) -> object:
    """Drop JSON-schema keys Gemini's FunctionDeclaration validator
    rejects, AND coerce union types like ``type: ["integer", "string"]``
    into a single Gemini ``Schema.type`` enum (which is a single value,
    not a list).

    Coercion rules, in order:

    1. Drop ``"null"`` entries — those mark optional fields and Gemini
       handles optionality via the parent schema's ``required`` list.
    2. If ``"string"`` is in the surviving union, pick it. Strings are
       the most permissive carrier — the agent can always serialize a
       number/boolean as text and the tool's parser coerces. Picking a
       narrower type (``"integer"`` from ``["integer", "string"]``)
       would make Gemini reject calls where the agent sends a stringy
       value the tool expects to handle (e.g. ``revision: "head"``).
    3. Else pick the first remaining type.
    """
    unsupported = {"additionalProperties", "title", "default"}
    if isinstance(schema, dict):
        result: dict[object, object] = {}
        for k, v in schema.items():
            if k in unsupported:
                continue
            if k == "type" and isinstance(v, list):
                non_null = [t for t in v if t != "null"]
                if not non_null:
                    result[k] = "string"
                elif "string" in non_null:
                    result[k] = "string"
                else:
                    result[k] = non_null[0]
                continue
            result[k] = strip_unsupported_schema_keys(v)
        return result
    if isinstance(schema, list):
        return [strip_unsupported_schema_keys(item) for item in schema]
    return schema


THINKING_BUDGETS: dict[str, int] = {
    "none": 0,
    "low": 4096,
    "med": 8192,
    "medium": 8192,
    "high": 24576,
}
ToolSchema = dict[str, object]


class GeminiProvider:
    """LLMClient implementation for Gemini generate_content."""

    def __init__(
        self,
        model_id: str = "gemini-3-pro-preview",
        *,
        api_key: str | None = None,
        thinking_level: str = "med",
        client: genai.Client | None = None,
        check_thinking_support: bool = True,
    ) -> None:
        self.model_id = model_id.removeprefix("gemini/")
        self.thinking_level = thinking_level
        self.client = client or genai.Client(api_key=env_api_key("GEMINI_API_KEY", api_key))
        self._supports_thinking = (
            self._check_thinking_support() if check_thinking_support else False
        )

    def _check_thinking_support(self) -> bool:
        try:
            model_info = self.client.models.get(model=self.model_id)
        except Exception:
            return False
        return bool(getattr(model_info, "thinking", False))

    @staticmethod
    def convert_arguments(args: str | ToolSchema | list[object] | None) -> ToolSchema:
        if args is None:
            return {}
        if isinstance(args, dict):
            return args
        if isinstance(args, list):
            return {"items": args}
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def encode_signature(signature: bytes | None) -> str | None:
        if signature is None:
            return None
        return base64.b64encode(signature).decode("utf-8")

    @staticmethod
    def decode_signature(signature: str | None) -> bytes | None:
        if signature is None:
            return None
        return base64.b64decode(signature)

    def convert_tools(self, tools: list[dict]) -> list[types.FunctionDeclaration]:
        declarations: list[types.FunctionDeclaration] = []
        for tool in tools or []:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                parameters = strip_unsupported_schema_keys(
                    dereference_schema(cast(ToolSchema, parameters))
                )
            declarations.append(
                types.FunctionDeclaration(
                    name=str(function.get("name", "")),
                    description=str(function.get("description", "")),
                    parameters=cast(types.Schema | None, parameters),
                )
            )
        return declarations

    def content_blocks_to_parts(self, content: list[ToolSchema]) -> list[types.Part]:
        parts: list[types.Part] = []
        for block in content:
            if block.get("type") == "text":
                parts.append(types.Part(text=str(block.get("text", ""))))
            elif block.get("type") == "image":
                source = block.get("source")
                if isinstance(source, Mapping) and source.get("type") == "base64":
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=source.get("media_type", "image/png"),
                                data=base64.b64decode(cast(str, source["data"])),
                            )
                        )
                    )
        return parts

    def convert_message(self, message: Message) -> types.Content:
        role = "model" if message.role == "assistant" else message.role
        parts: list[types.Part] = []
        content = message.content

        if role == "model" and message.tool_calls is not None:
            if content:
                parts.extend(
                    self.content_blocks_to_parts(cast(list[ToolSchema], content))
                    if isinstance(content, list)
                    else [types.Part(text=content)]
                )
            for tool_call in message.tool_calls:
                google_extra = tool_call.extra_content.get("google")
                fallback_signature = (
                    cast(str | None, google_extra.get("thought_signature"))
                    if isinstance(google_extra, dict)
                    else None
                )
                raw_signature = tool_call.thought_signature or fallback_signature
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            name=tool_call.function.name,
                            args=self.convert_arguments(tool_call.function.arguments),
                        ),
                        thought_signature=self.decode_signature(raw_signature),
                    )
                )
        elif message.role == "tool":
            response_body: ToolSchema
            if isinstance(content, list):
                blocks = cast(list[ToolSchema], content)
                text_parts = [
                    str(block.get("text", "")) for block in blocks if block.get("type") == "text"
                ]
                response_body = {"result": "\n".join(text_parts)} if text_parts else {}
                for block in blocks:
                    if block.get("type") == "image":
                        parts.extend(self.content_blocks_to_parts([block]))
            else:
                response_body = self.convert_arguments(content)
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=message.name or message.tool_call_id or "",
                        response=response_body,
                    )
                )
            )
            role = "user"
        elif content:
            parts.extend(
                self.content_blocks_to_parts(cast(list[ToolSchema], content))
                if isinstance(content, list)
                else [types.Part(text=content)]
            )

        if not parts:
            parts.append(types.Part(text=""))
        return types.Content(role=role, parts=parts)

    def _thinking_config(self) -> types.ThinkingConfig | None:
        if not self._supports_thinking:
            return None
        budget = THINKING_BUDGETS.get(self.thinking_level, 0)
        return types.ThinkingConfig(
            include_thoughts=budget > 0,
            thinking_budget=budget,
        )

    def _build_contents(self, messages: Sequence[Message]) -> list[types.Content]:
        return [self.convert_message(message) for message in sanitize_tool_messages(messages)]

    def _build_config(self, system: str, tools: list[dict]) -> types.GenerateContentConfig:
        tool_declarations = self.convert_tools(tools)
        return types.GenerateContentConfig(
            system_instruction=system,
            tools=(
                [types.Tool(function_declarations=tool_declarations)]
                if tool_declarations
                else None
            ),
            thinking_config=self._thinking_config(),
            temperature=1.0,
        )

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None" = None,
    ) -> Message:
        text = ""
        thought = ""
        tool_calls: list[ToolCall] = []

        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_id,
            contents=self._build_contents(messages),
            config=self._build_config(system, tools),
        )
        async for chunk in stream:
            if listener is not None and listener.cancel_requested():
                raise StreamCancelled
            candidates = getattr(chunk, "candidates", []) or []
            if not candidates:
                continue
            content = getattr(candidates[0], "content", None)
            parts = getattr(content, "parts", []) if content is not None else []
            for part in parts or []:
                part_text = cast(str | None, getattr(part, "text", None))
                if part_text:
                    if bool(getattr(part, "thought", False)):
                        thought += part_text
                        if listener is not None:
                            await listener.on_thinking_delta(part_text)
                    else:
                        text += part_text
                        if listener is not None:
                            await listener.on_text_delta(part_text)
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    args = getattr(function_call, "args", {})
                    signature = self.encode_signature(
                        cast(bytes | None, getattr(part, "thought_signature", None))
                    )
                    tool_calls.append(
                        ToolCall(
                            id=cast(str | None, getattr(function_call, "id", None))
                            or f"call_{uuid.uuid4().hex}",
                            function=ToolCallFunction(
                                name=cast(str | None, getattr(function_call, "name", None)) or "",
                                arguments=(json.dumps(args) if isinstance(args, dict) else ""),
                            ),
                            thought_signature=signature,
                            extra_content=(
                                {"google": {"thought_signature": signature}} if signature else {}
                            ),
                        )
                    )

        return Message(
            role="assistant",
            content=text or None,
            tool_calls=tool_calls or None,
            thought_summary=thought or None,
        )
