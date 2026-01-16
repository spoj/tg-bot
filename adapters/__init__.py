"""Model adapter exports and factory functions.

This module provides singleton instances of adapters for different use cases
and factory functions to create/retrieve them.
"""

from .base import ModelAdapter
from .anthropic import AnthropicAdapter
from .gemini import GeminiAdapter, GrokAdapter

__all__ = [
    "ModelAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
    "GrokAdapter",
    "get_reasoning_adapter",
    "get_vision_adapter",
    "get_long_context_adapter",
    "get_search_adapter",
]

# Singleton adapter instances (lazy initialized)
_reasoning_adapter: AnthropicAdapter | None = None
_vision_adapter: GeminiAdapter | None = None
_long_context_adapter: GeminiAdapter | None = None
_search_adapter: GrokAdapter | None = None


def get_reasoning_adapter() -> AnthropicAdapter:
    """Get the main reasoning adapter (Claude Opus).

    Used for the main agent loop with deep thinking.
    Features: prompt caching, extended thinking, tool use.
    """
    global _reasoning_adapter
    if _reasoning_adapter is None:
        _reasoning_adapter = AnthropicAdapter(
            model="openrouter/anthropic/claude-opus-4.5",
            reasoning={"max_tokens": 30000},
            max_tokens=32000,
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
            model="gemini/gemini-3-flash-preview",
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
            model="gemini/gemini-3-flash-preview",
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
            model="openrouter/x-ai/grok-4.1-fast",
            timeout=600,
        )
    return _search_adapter
