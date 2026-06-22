from __future__ import annotations

import os

from actant.llm.base import LLMClient


def build_llm() -> tuple[LLMClient, str]:
    """Pick the LLM provider + model.

    If ``ACTANT_PROVIDER`` is set, it pins the provider explicitly
    (one of: ``anthropic``, ``openai``, ``gemini``). Otherwise, falls
    back to whichever API key is set first in the env, in order:
    anthropic → openai → gemini.

    Returns ``(client, model_id)`` for display.
    """
    forced = (os.getenv("ACTANT_PROVIDER") or "").strip().lower()

    if forced == "openai" or (not forced and not os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENAI_API_KEY")):
        from actant.llm.providers.openai import OpenAIProvider

        model = os.getenv("ACTANT_MODEL", "gpt-5.4-nano")
        return OpenAIProvider(model_id=model), model

    if forced == "anthropic" or (not forced and os.getenv("ANTHROPIC_API_KEY")):
        from actant.llm.providers.anthropic import AnthropicProvider

        model = os.getenv("ACTANT_MODEL", "claude-sonnet-4-20250514")
        return AnthropicProvider(model_id=model), model

    if forced == "gemini" or (not forced and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))):
        from actant.llm.providers.gemini import GeminiProvider

        model = os.getenv("ACTANT_MODEL", "gemini-2.5-pro")
        return GeminiProvider(model_id=model), model

    raise RuntimeError(
        "No LLM API key found. Set one of ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "or GEMINI_API_KEY before starting the demo server."
    )
