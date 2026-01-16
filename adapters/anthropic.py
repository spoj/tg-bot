"""Anthropic Claude adapter with prompt caching and extended thinking.

This adapter handles Claude models via OpenRouter, implementing:
- Prompt caching (cache_control blocks for 90% cost reduction on cache hits)
- Extended thinking preservation (reasoning_details)
- Multimodal content (images, files)
"""

import json
from typing import Any

from litellm import acompletion

from session import Message, MessageRole, ToolCall, ToolResult, Attachment


class AnthropicAdapter:
    """Adapter for Claude models via OpenRouter.

    Features:
    - Prompt caching via cache_control blocks (ephemeral breakpoints)
    - Extended thinking preservation for prefix consistency
    - OpenRouter provider routing
    - Multimodal content handling
    """

    def __init__(
        self,
        model: str = "openrouter/anthropic/claude-opus-4.5",
        reasoning: dict | None = None,
        max_tokens: int = 16000,
        timeout: int = 600,
        provider_order: list[str] | None = None,
    ):
        """Initialize Anthropic adapter.

        Args:
            model: Model identifier (OpenRouter format).
            reasoning: Reasoning config (e.g., {"effort": "high"}).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            provider_order: OpenRouter provider routing order.
        """
        self._model = model
        self.config: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        if reasoning:
            self.config["reasoning"] = reasoning
        if provider_order:
            self.config["extra_body"] = {
                "provider": {"order": provider_order, "allow_fallbacks": False}
            }

    @property
    def model_name(self) -> str:
        """Return model identifier for logging."""
        return self._model

    def render_messages(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> list[dict]:
        """Render to Anthropic format with cache_control on last message.

        Caching strategy:
        - Add cache_control: {type: ephemeral} to the last message
        - This creates a cache breakpoint at the end of known context
        - Subsequent tool loop iterations hit cache for all prior messages

        Args:
            system_prompt: The system prompt.
            messages: Semantic messages to render.

        Returns:
            List of message dicts in Anthropic/OpenAI format.
        """
        result: list[dict] = [{"role": "system", "content": system_prompt}]

        for i, msg in enumerate(messages):
            rendered = self._render_message(msg)
            is_last = i == len(messages) - 1

            # Handle tool results which render as multiple messages
            if isinstance(rendered, list):
                for j, r in enumerate(rendered):
                    # Cache control on very last item of very last message
                    if is_last and j == len(rendered) - 1:
                        self._add_cache_control(r)
                    result.append(r)
            else:
                if is_last:
                    self._add_cache_control(rendered)
                result.append(rendered)

        return result

    def _render_message(self, msg: Message) -> dict | list[dict]:
        """Convert a single semantic message to Anthropic format."""
        if msg.role == MessageRole.USER:
            content = self._build_content_blocks(msg)
            return {"role": "user", "content": content}

        elif msg.role == MessageRole.ASSISTANT:
            result: dict[str, Any] = {"role": "assistant", "content": msg.content}

            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

            # Preserve extended thinking for prefix consistency
            # This ensures cache hits when the model returns the same reasoning
            if msg.thinking:
                result["reasoning_details"] = msg.thinking

            return result

        elif msg.role == MessageRole.TOOL_RESULT:
            # Each tool result becomes a separate message
            return [
                {"role": "tool", "tool_call_id": tr.call_id, "content": tr.content}
                for tr in msg.tool_results
            ]

        elif msg.role == MessageRole.SYSTEM:
            return {"role": "system", "content": msg.content}

        raise ValueError(f"Unknown role: {msg.role}")

    def _build_content_blocks(self, msg: Message) -> list[dict] | str:
        """Build content blocks for user message with attachments."""
        # Simple case: no attachments, just return string
        if not msg.attachments:
            return msg.content or ""

        # Build content block array for multimodal
        blocks: list[dict] = []

        if msg.content:
            blocks.append({"type": "text", "text": msg.content})

        for att in msg.attachments:
            if att.mime_type.startswith("image/"):
                blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{att.mime_type};base64,{att.data_b64}"
                        },
                    }
                )
            else:
                # Audio, video, PDF use file format
                blocks.append(
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:{att.mime_type};base64,{att.data_b64}"
                        },
                    }
                )

        return blocks if blocks else ""

    def _add_cache_control(self, rendered: dict) -> None:
        """Add cache_control to a rendered message (mutates in place).

        Anthropic prompt caching requires cache_control blocks to mark
        where to cache. We add ephemeral breakpoints to enable caching
        of the context up to that point.
        """
        content = rendered.get("content")

        if isinstance(content, str):
            # Convert string to block format with cache_control
            rendered["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            # Add to first block (must be a dict)
            if isinstance(content[0], dict):
                content[0]["cache_control"] = {"type": "ephemeral"}

    def parse_response(self, response: Any) -> Message:
        """Parse Anthropic response into semantic Message."""
        raw = response.choices[0].message

        msg = Message(
            role=MessageRole.ASSISTANT,
            content=raw.content.strip() if raw.content else None,
        )

        # Extract tool calls
        if raw.tool_calls:
            msg.tool_calls = [
                ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                    call_id=tc.id,
                )
                for tc in raw.tool_calls
            ]

        # Extract extended thinking (Anthropic-specific)
        provider_fields = getattr(raw, "provider_specific_fields", None) or {}
        if provider_fields.get("reasoning_details"):
            msg.thinking = provider_fields["reasoning_details"]

        return msg

    async def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> tuple[Message, dict[str, Any]]:
        """Run completion via litellm.

        Args:
            system_prompt: The system prompt.
            messages: Conversation history as semantic messages.
            tools: Optional tool definitions (OpenAI format).
            **kwargs: Additional litellm arguments.

        Returns:
            Tuple of (parsed_message, usage_stats).
        """
        rendered = self.render_messages(system_prompt, messages)

        call_kwargs = {**self.config, **kwargs}
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"

        response = await acompletion(messages=rendered, **call_kwargs)

        parsed = self.parse_response(response)
        usage = self._extract_usage(response)

        return parsed, usage

    def _extract_usage(self, response: Any) -> dict[str, Any]:
        """Extract usage statistics from response."""
        usage = getattr(response, "usage", None)
        if not usage:
            return {}

        result: dict[str, Any] = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

        # Extract cache stats from prompt_tokens_details (LiteLLM format)
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            result["cache_read"] = getattr(details, "cached_tokens", 0) or 0
            result["cache_write"] = getattr(details, "cache_write_tokens", 0) or 0

        return result
