"""Gemini adapter for vision, long-context, and general tasks.

This adapter handles Gemini models, implementing:
- Simpler message format (no caching)
- File-based multimodal format for non-images
- Optional reasoning/thinking support (for reasoning-capable models via OpenRouter)
"""

import json
from typing import Any

from litellm import acompletion

from session import Message, MessageRole, ToolCall, Attachment


class GeminiAdapter:
    """Adapter for Google Gemini models.

    Simpler than Anthropic:
    - No prompt caching mechanisms
    - Uses 'file' format for audio, video, PDF (not image_url)
    - Optional reasoning support for thinking-capable models (e.g., Gemini 3.1 Pro)
    """

    def __init__(
        self,
        model: str = "gemini/gemini-3-flash-preview",
        max_tokens: int = 4000,
        timeout: int = 120,
        reasoning: dict | None = None,
        provider_order: list[str] | None = None,
    ):
        """Initialize Gemini adapter.

        Args:
            model: Model identifier (litellm format).
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            reasoning: Optional reasoning config (e.g., {"effort": "high"}).
                       Passed via extra_body for OpenRouter Gemini models.
            provider_order: Optional OpenRouter provider routing order.
        """
        self._model = model
        self.config: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        extra_body: dict[str, Any] = {}
        if reasoning:
            extra_body["reasoning"] = reasoning
        if provider_order:
            extra_body["provider"] = {
                "order": provider_order,
                "allow_fallbacks": False,
            }
        if extra_body:
            self.config["extra_body"] = extra_body

    @property
    def model_name(self) -> str:
        """Return model identifier for logging."""
        return self._model

    def render_messages(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> list[dict]:
        """Render to Gemini format.

        Gemini uses implicit caching (automatic prefix-stable caching)
        so no cache_control breakpoints are needed.

        Args:
            system_prompt: The system prompt.
            messages: Semantic messages to render.

        Returns:
            List of message dicts in OpenAI-compatible format.
        """
        result: list[dict] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            rendered = self._render_message(msg)
            # Handle tool results which render as multiple messages
            if isinstance(rendered, list):
                result.extend(rendered)
            else:
                result.append(rendered)

        return result

    def _render_message(self, msg: Message) -> dict | list[dict]:
        """Convert a single semantic message to Gemini format."""
        if msg.role == MessageRole.USER:
            content = self._build_content_blocks(msg)
            return {"role": "user", "content": content}

        elif msg.role == MessageRole.ASSISTANT:
            # If we have the raw message from the API, replay it verbatim.
            # This avoids 500s from OpenRouter when we reconstruct the
            # message format ourselves (e.g. reasoning_details).
            if "raw_assistant" in msg.metadata:
                return {"role": "assistant", **msg.metadata["raw_assistant"]}

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
        """Build content blocks with Gemini's file format.

        Gemini uses:
        - image_url for images (same as OpenAI)
        - file format for audio, video, PDF
        """
        # Simple case: no attachments
        if not msg.attachments:
            return msg.content or ""

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
                # Gemini uses 'file' format for audio, video, PDF
                blocks.append(
                    {
                        "type": "file",
                        "file": {
                            "file_data": f"data:{att.mime_type};base64,{att.data_b64}"
                        },
                    }
                )

        return blocks if blocks else ""

    def parse_response(self, response: Any) -> Message:
        """Parse Gemini response into semantic Message.

        Stores the raw assistant message dict in metadata so it can be
        replayed verbatim in subsequent requests (avoids 500s from
        OpenRouter when we reconstruct the message ourselves).
        """
        raw = response.choices[0].message

        msg = Message(
            role=MessageRole.ASSISTANT,
            content=raw.content.strip() if raw.content else None,
        )

        # Extract tool calls (needed for agent loop logic)
        if raw.tool_calls:
            msg.tool_calls = [
                ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                    call_id=tc.id,
                )
                for tc in raw.tool_calls
            ]

        # Extract reasoning/thinking for display/logging
        provider_fields = getattr(raw, "provider_specific_fields", None) or {}
        reasoning = (
            provider_fields.get("reasoning_content")
            or provider_fields.get("reasoning_details")
            or provider_fields.get("reasoning")
            or getattr(raw, "reasoning_content", None)
        )
        if isinstance(reasoning, str) and reasoning:
            msg.thinking = reasoning

        # Stash the raw message dict so _render_message can replay it as-is
        msg.metadata["raw_assistant"] = raw.model_dump()

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

        # Extract cache stats from prompt_tokens_details (OpenRouter/LiteLLM format)
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            result["cache_read"] = getattr(details, "cached_tokens", 0) or 0
            result["cache_write"] = getattr(details, "cache_write_tokens", 0) or 0

        return result


class GrokAdapter:
    """Adapter for Grok models (search-focused).

    Used for web search queries via OpenRouter.
    Supports the 'online' mode for real-time search.
    """

    def __init__(
        self,
        model: str = "openrouter/x-ai/grok-4.1-fast",
        timeout: int = 600,
    ):
        """Initialize Grok adapter.

        Args:
            model: Model identifier (OpenRouter format).
            timeout: Request timeout in seconds.
        """
        self._model = model
        self.config: dict[str, Any] = {
            "model": model,
            "timeout": timeout,
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
        """Render to simple format for Grok."""
        result: list[dict] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if msg.role == MessageRole.USER:
                result.append({"role": "user", "content": msg.content or ""})
            elif msg.role == MessageRole.ASSISTANT:
                result.append({"role": "assistant", "content": msg.content or ""})

        return result

    def parse_response(self, response: Any) -> Message:
        """Parse Grok response into semantic Message."""
        raw = response.choices[0].message

        return Message(
            role=MessageRole.ASSISTANT,
            content=raw.content.strip() if raw.content else None,
        )

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
            messages: Conversation history.
            tools: Not used for Grok (search-only).
            **kwargs: Additional arguments (e.g., extra_body for plugins).

        Returns:
            Tuple of (parsed_message, usage_stats).
        """
        rendered = self.render_messages(system_prompt, messages)

        call_kwargs = {**self.config, **kwargs}
        # Grok doesn't use tools in our search use case

        response = await acompletion(messages=rendered, **call_kwargs)

        parsed = self.parse_response(response)
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        }

        return parsed, usage
