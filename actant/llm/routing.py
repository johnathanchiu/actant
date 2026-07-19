"""Model-id based provider routing."""

from __future__ import annotations

from actant.llm.base import LLMClient

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
        try:
            from actant.llm.providers.anthropic import AnthropicProvider
        except ModuleNotFoundError as exc:
            if exc.name == "anthropic":
                raise ImportError(
                    "Anthropic support requires `pip install 'actant[anthropic]'`."
                ) from exc
            raise
        return AnthropicProvider(model_id=model_id, thinking_level=thinking_level)
    if provider == "openai":
        try:
            from actant.llm.providers.openai import OpenAIProvider
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                raise ImportError(
                    "OpenAI support requires `pip install 'actant[openai]'`."
                ) from exc
            raise
        return OpenAIProvider(model_id=model_id, thinking_level=thinking_level)
    if provider == "gemini":
        try:
            from actant.llm.providers.gemini import GeminiProvider
        except ImportError as exc:
            if exc.name in {"google", "google.genai"}:
                raise ImportError(
                    "Gemini support requires `pip install 'actant[gemini]'`."
                ) from exc
            raise
        return GeminiProvider(model_id=model_id, thinking_level=thinking_level)
    if provider == "qwen":
        try:
            from actant.llm.providers.qwen import QwenProvider
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                raise ImportError("Qwen support requires `pip install 'actant[qwen]'`.") from exc
            raise
        return QwenProvider(model_id=model_id)
    raise ValueError(f"Unsupported provider: {provider}")
