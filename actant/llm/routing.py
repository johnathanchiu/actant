"""Model-id based provider routing."""

from __future__ import annotations

from actant.llm.base import LLMClient
from actant.llm.providers.anthropic import AnthropicProvider
from actant.llm.providers.gemini import GeminiProvider
from actant.llm.providers.openai import OpenAIProvider
from actant.llm.providers.qwen import QwenProvider

MODEL_PREFIXES: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
    "gemini": "gemini",
    "qwen": "qwen",
}


def provider_for_model(model_id: str) -> str:
    if "/" in model_id:
        namespace = model_id.split("/", 1)[0]
        if namespace in MODEL_PREFIXES:
            return MODEL_PREFIXES[namespace]

    for prefix, provider in MODEL_PREFIXES.items():
        if model_id.startswith(prefix):
            return provider

    raise ValueError(
        f"Cannot determine provider for model {model_id!r}. "
        f"Known prefixes: {sorted(MODEL_PREFIXES)}"
    )


def llm_for_model(model_id: str, *, thinking_level: str = "med") -> LLMClient:
    provider = provider_for_model(model_id)
    if provider == "anthropic":
        return AnthropicProvider(model_id=model_id, thinking_level=thinking_level)
    if provider == "openai":
        return OpenAIProvider(model_id=model_id, thinking_level=thinking_level)
    if provider == "gemini":
        return GeminiProvider(model_id=model_id, thinking_level=thinking_level)
    if provider == "qwen":
        return QwenProvider(model_id=model_id)
    raise ValueError(f"Unsupported provider: {provider}")
