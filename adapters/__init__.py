"""Model adapter exports and factory functions.

This module provides singleton instances of adapters for different use cases
and factory functions to create/retrieve them.
"""

import os

from .base import ModelAdapter
from .anthropic import AnthropicAdapter
from .gemini import GeminiAdapter, GrokAdapter
from .gpt import GPTAdapter

__all__ = [
    "ModelAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "GrokAdapter",
    "GPTAdapter",
    "get_reasoning_adapter",
    "get_vision_adapter",
    "get_long_context_adapter",
    "get_search_adapter",
    "get_gpt_adapter",
]

# Singleton adapter instances (lazy initialized)
_reasoning_adapter: ModelAdapter | None = None
_reasoning_model: str | None = None
_vision_adapter: GeminiAdapter | None = None
_long_context_adapter: GeminiAdapter | None = None
_search_adapter: GrokAdapter | None = None
_gpt_adapter: GPTAdapter | None = None


def get_reasoning_adapter() -> ModelAdapter:
    """Get the main reasoning adapter.

    Default model is Claude Opus, but it can be overridden with `AGENT_MODEL`.

    Supported overrides:
    - `openrouter/openai/gpt-5.2` -> `GPTAdapter`
    - `openrouter/google/gemini-*` -> `GeminiAdapter` (with reasoning)
    - anything else -> `AnthropicAdapter` (OpenRouter Claude)
    """
    global _reasoning_adapter, _reasoning_model

    model = os.environ.get("AGENT_MODEL", "openrouter/anthropic/claude-opus-4.6")

    # Recreate singleton if model changed at runtime
    if _reasoning_adapter is None or _reasoning_model != model:
        _reasoning_model = model

        if model.startswith("openrouter/openai/") or "gpt" in model.lower():
            _reasoning_adapter = GPTAdapter(
                model=model,
                max_tokens=16000,
                timeout=600,
            )
        elif model.startswith("openrouter/google/") or "gemini" in model.lower():
            _reasoning_adapter = GeminiAdapter(
                model=model,
                reasoning={"effort": "high"},
                max_tokens=16000,
                timeout=600,
            )
        else:
            _reasoning_adapter = AnthropicAdapter(
                model=model,
                reasoning={"effort": "high"},
                max_tokens=64000,
                timeout=600,
                provider_order=["anthropic"],
            )

    return _reasoning_adapter


def get_vision_adapter() -> GeminiAdapter:
    """Get the vision adapter (Gemini Flash).

    Used for image/audio/PDF processing and multimodal queries.
    """
    global _vision_adapter
    if _vision_adapter is None:
        _vision_adapter = GeminiAdapter(
            model=os.environ.get("GEMINI_MODEL", "gemini/gemini-3-flash-preview"),
            max_tokens=4000,
            timeout=120,
        )
    return _vision_adapter


def get_long_context_adapter() -> GeminiAdapter:
    """Get the long context adapter (Gemini Flash).

    Used for searching/analyzing large files (stream.txt, etc).
    Higher token limit and timeout than vision adapter.
    """
    global _long_context_adapter
    if _long_context_adapter is None:
        _long_context_adapter = GeminiAdapter(
            model=os.environ.get("GEMINI_MODEL", "gemini/gemini-3-flash-preview"),
            max_tokens=8000,
            timeout=300,
        )
    return _long_context_adapter


def get_search_adapter() -> GrokAdapter:
    """Get the search adapter (Grok).

    Used for web search queries with real-time information.
    """
    global _search_adapter
    if _search_adapter is None:
        _search_adapter = GrokAdapter(
            model=os.environ.get("SEARCH_MODEL", "openrouter/x-ai/grok-4.1-fast"),
            timeout=600,
        )
    return _search_adapter


def get_gpt_adapter() -> GPTAdapter:
    """Get the GPT adapter (GPT-5.2 via OpenRouter).

    Not used by default in the main agent loop.

    Note: uses `GPT_MODEL` (not `AGENT_MODEL`) so enabling GPT as the
    reasoning model doesn't accidentally change this adapter too.
    """
    global _gpt_adapter
    if _gpt_adapter is None:
        _gpt_adapter = GPTAdapter(
            model=os.environ.get("GPT_MODEL", "openrouter/openai/gpt-5.2"),
            max_tokens=16000,
            timeout=600,
        )
    return _gpt_adapter
