"""Model configuration and LLM call wrappers."""

from litellm import acompletion

# --- Configs by Purpose ---
REASONING = {
    "model": "openrouter/anthropic/claude-opus-4.5",
    "reasoning": {"effort": "high"},
    "max_tokens": 16000,
    "timeout": 600,
}

VISION = {
    "model": "gemini/gemini-3-flash-preview",
    "max_tokens": 4000,
    "timeout": 120,
}

SEARCH = {
    "model": "openrouter/x-ai/grok-4.1-fast",
    "timeout": 600,
}

LONG_CONTEXT = {
    "model": "gemini/gemini-3-flash-preview",
    "max_tokens": 8000,
    "timeout": 300,
}


async def reasoning_complete(messages: list, **kwargs):
    """For tasks requiring deep thinking - agent loop, rotation, synthesis."""
    config = REASONING.copy()
    config.update(kwargs)
    return await acompletion(messages=messages, **config)


async def vision_complete(messages: list, **kwargs):
    """For image/PDF/attachment processing."""
    config = VISION.copy()
    config.update(kwargs)
    return await acompletion(messages=messages, **config)


async def search_complete(messages: list, **kwargs):
    """For web search queries."""
    config = SEARCH.copy()
    config.update(kwargs)
    return await acompletion(messages=messages, **config)


async def long_context_complete(messages: list, **kwargs):
    """For long context queries (e.g., searching large files)."""
    config = LONG_CONTEXT.copy()
    config.update(kwargs)
    return await acompletion(messages=messages, **config)
