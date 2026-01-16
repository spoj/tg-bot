"""Model adapter protocol definition.

This module defines the interface that all model adapters must implement.
Adapters handle the translation between LLM-agnostic session messages and
provider-specific API formats.
"""

from typing import Protocol, runtime_checkable, Any

from session import Message


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for LLM model adapters.

    Adapters encapsulate all provider-specific logic:
    - Message format conversion (render)
    - Response parsing
    - Provider-specific features (caching, extended thinking)
    - LLM API calls

    Implementations must handle:
    - Converting semantic Messages to provider JSON format
    - Parsing provider responses back to semantic Messages
    - Provider-specific optimizations (caching breakpoints, etc.)
    """

    @property
    def model_name(self) -> str:
        """Return the model identifier for logging."""
        ...

    def render_messages(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> list[dict]:
        """Convert semantic messages to provider-specific JSON format.

        Handles:
        - Message structure differences between providers
        - Multimodal content format (image_url vs file)
        - Caching hints/breakpoints (e.g., Anthropic cache_control)
        - Tool call/result format conversion

        Args:
            system_prompt: The system prompt to use.
            messages: List of semantic messages.

        Returns:
            List of message dicts in provider-specific format.
        """
        ...

    def parse_response(self, response: Any) -> Message:
        """Parse provider API response into semantic Message.

        Extracts:
        - Text content
        - Tool calls (normalized to ToolCall objects)
        - Extended thinking/reasoning (if supported)
        - Provider-specific metadata worth preserving

        Args:
            response: Raw response from litellm acompletion.

        Returns:
            Semantic Message object.
        """
        ...

    async def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> tuple[Message, dict[str, Any]]:
        """Run completion and return (message, usage_stats).

        Orchestrates the full completion flow:
        1. render_messages() to convert to provider format
        2. API call via litellm
        3. parse_response() to extract semantic message

        Does NOT handle retries - that's the caller's responsibility.
        This keeps the adapter focused on format translation.

        Args:
            system_prompt: The system prompt to use.
            messages: List of semantic messages (conversation history).
            tools: Optional list of tool definitions (OpenAI format).
            **kwargs: Additional arguments passed to litellm.

        Returns:
            Tuple of (parsed_message, usage_stats_dict).
            Usage dict may contain: input_tokens, output_tokens,
            cache_read, cache_write, etc.
        """
        ...
