from __future__ import annotations

import json
import os
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

from actant.llm.base import LLMClient
from actant.llm.messages import Message, ToolCall, ToolCallFunction

if TYPE_CHECKING:
    from actant.runtime.events.streaming import StreamListener


class DemoLLM:
    """Deterministic local model for exercising the complete demo without API keys."""

    model_id = "demo/deterministic"

    async def complete(
        self,
        system: str,
        messages: Sequence[Message],
        tools: list[dict],
        listener: "StreamListener | None" = None,
    ) -> Message:
        del tools
        latest = messages[-1] if messages else Message(role="user", content="")
        text = latest.content if isinstance(latest.content, str) else ""
        lowered = text.lower()

        if latest.role == "tool":
            return await self._text(
                "Done — the result came back and the conversation resumed from durable state.",
                listener,
            )

        if "leaf summarizer" in system:
            return await self._text(
                "Durable delegation verified\n\n- The nested subagent completed.\n"
                "- Its result returned through the parent tool call.\n"
                "- The parent can now resume exactly once.",
                listener,
            )

        if "focused research subagent" in system:
            if "approval" in lowered:
                return await self._tool_call(
                    "request_approval",
                    {"action": "allow the researcher to finish its delegated task"},
                    listener,
                )
            return await self._tool_call(
                "task",
                {
                    "subagent": "summarizer",
                    "message": "Summarize why durable nested agent delegation is useful.",
                },
                listener,
            )

        if "approval" in lowered and not any(
            word in lowered for word in ("delegate", "subagent", "research")
        ):
            return await self._tool_call(
                "request_approval",
                {"action": "run the deterministic release-readiness check"},
                listener,
            )
        if "mixed" in lowered or "parallel" in lowered:
            return await self._tool_calls(
                [
                    ("get_current_time", {}),
                    (
                        "request_approval",
                        {"action": "complete the mixed parallel tool group"},
                    ),
                ],
                listener,
            )
        if "weather" in lowered:
            return await self._tool_calls(
                [
                    ("get_weather", {"location": "New York, NY"}),
                    ("get_weather", {"location": "London, UK"}),
                    ("get_weather", {"location": "Tokyo, Japan"}),
                ],
                listener,
            )
        if "pizza" in lowered:
            return await self._tool_call(
                "ask_user",
                {
                    "question": "What kind of pizza sounds good right now?",
                    "options": [
                        "Classic pepperoni",
                        "Spicy and meaty",
                        "Veggie-loaded",
                        "Surprise me",
                    ],
                },
                listener,
            )
        if "choose" in lowered or "question" in lowered:
            return await self._tool_call(
                "ask_user",
                {
                    "question": "Which behavior should the demo verify?",
                    "options": ["Durable approval", "Nested subagent", "Streaming response"],
                },
                listener,
            )
        if any(word in lowered for word in ("delegate", "subagent", "research")):
            message = (
                "Request approval from the human, then return a concise confirmation."
                if "approval" in lowered
                else "Demonstrate nested durable delegation and return a concise result."
            )
            return await self._tool_call(
                "task",
                {
                    "subagent": "researcher",
                    "message": message,
                },
                listener,
            )
        return await self._text(
            "Actant is running in deterministic demo mode. Ask for an approval, a choice, "
            "or a subagent delegation to exercise durable runtime behavior.",
            listener,
        )

    async def _text(self, text: str, listener: "StreamListener | None") -> Message:
        if listener is not None:
            midpoint = max(1, len(text) // 2)
            await listener.on_text_delta(text[:midpoint])
            await listener.on_text_delta(text[midpoint:])
        return Message(role="assistant", content=text)

    async def _tool_call(
        self,
        name: str,
        arguments: dict[str, object],
        listener: "StreamListener | None",
    ) -> Message:
        return await self._tool_calls([(name, arguments)], listener)

    async def _tool_calls(
        self,
        calls: list[tuple[str, dict[str, object]]],
        listener: "StreamListener | None",
    ) -> Message:
        tool_calls: list[ToolCall] = []
        for name, arguments in calls:
            call_id = f"demo_{uuid.uuid4().hex[:12]}"
            encoded = json.dumps(arguments)
            if listener is not None:
                await listener.on_tool_call_start(call_id, name)
                await listener.on_tool_call_args_delta(call_id, encoded)
                await listener.on_tool_call_args_complete(call_id)
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    function=ToolCallFunction(name=name, arguments=encoded),
                )
            )
        return Message(role="assistant", tool_calls=tool_calls)


def build_llm() -> tuple[LLMClient, str]:
    """Pick the LLM provider + model.

    If ``ACTANT_PROVIDER`` is set, it pins the provider explicitly
    (one of: ``fake``, ``anthropic``, ``openai``, ``gemini``). Otherwise, falls
    back to whichever API key is set first in the env, in order:
    anthropic → openai → gemini. With no key, it uses ``DemoLLM``.

    Returns ``(client, model_id)`` for display.
    """
    forced = (os.getenv("ACTANT_PROVIDER") or "").strip().lower()

    def configured_model(provider: str) -> str:
        model = (os.getenv("ACTANT_MODEL") or "").strip()
        if not model:
            raise RuntimeError(
                f"ACTANT_MODEL is required when using the {provider} provider. "
                "Model IDs are application configuration, not Actant defaults."
            )
        return model

    if forced == "fake":
        return DemoLLM(), DemoLLM.model_id

    if forced == "openai" or (
        not forced and not os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY")
    ):
        from actant.llm.providers.openai import OpenAIProvider

        model = configured_model("openai")
        return OpenAIProvider(model_id=model), model

    if forced == "anthropic" or (not forced and os.getenv("ANTHROPIC_API_KEY")):
        from actant.llm.providers.anthropic import AnthropicProvider

        model = configured_model("anthropic")
        return AnthropicProvider(model_id=model), model

    if forced == "gemini" or (
        not forced and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    ):
        from actant.llm.providers.gemini import GeminiProvider

        model = configured_model("gemini")
        return GeminiProvider(model_id=model), model

    return DemoLLM(), DemoLLM.model_id
