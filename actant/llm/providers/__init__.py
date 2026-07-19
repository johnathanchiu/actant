"""LLM provider implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from actant.llm.providers.anthropic import AnthropicProvider
    from actant.llm.providers.fake import FakeLLM, FakeResponse
    from actant.llm.providers.gemini import GeminiProvider
    from actant.llm.providers.openai import OpenAIProvider
    from actant.llm.providers.qwen import QwenProvider

__all__ = [
    "AnthropicProvider",
    "FakeLLM",
    "FakeResponse",
    "GeminiProvider",
    "OpenAIProvider",
    "QwenProvider",
]


def __getattr__(name: str) -> Any:
    if name == "AnthropicProvider":
        from actant.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider
    if name in {"FakeLLM", "FakeResponse"}:
        from actant.llm.providers import fake

        return getattr(fake, name)
    if name == "GeminiProvider":
        from actant.llm.providers.gemini import GeminiProvider

        return GeminiProvider
    if name == "OpenAIProvider":
        from actant.llm.providers.openai import OpenAIProvider

        return OpenAIProvider
    if name == "QwenProvider":
        from actant.llm.providers.qwen import QwenProvider

        return QwenProvider
    raise AttributeError(f"module 'actant.llm.providers' has no attribute {name!r}")
