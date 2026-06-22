"""LLM error types."""

from __future__ import annotations


class StreamCancelled(Exception):
    """Raised when a stream listener requests cancellation mid-generation."""
