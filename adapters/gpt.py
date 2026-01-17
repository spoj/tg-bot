"""OpenRouter GPT adapter (OpenAI-compatible chat).

This adapter is similar to the Anthropic adapter but targets OpenAI-style
models routed through OpenRouter (e.g. GPT-5.2).

Features:
- OpenAI-compatible message / tool call formatting
- Multimodal content blocks (image_url + file)
- Optional OpenRouter provider routing order
"""

import json
from typing import Any

from litellm import acompletion

from session import Message, MessageRole, ToolCall


class GPTAdapter:
    """Adapter for GPT models via OpenRouter."""

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-5.2",
        max_tokens: int = 16000,
        timeout: int = 600,
        provider_order: list[str] | None = None,
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
    ):
        self._model = model
        self.config: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if reasoning_effort:
            self.config["reasoning_effort"] = reasoning_effort
        if verbosity:
            self.config["verbosity"] = verbosity

        if provider_order:
            self.config["extra_body"] = {
                "provider": {"order": provider_order, "allow_fallbacks": False}
            }

    @property
    def model_name(self) -> str:
        return self._model

    def render_messages(
        self, system_prompt: str, messages: list[Message]
    ) -> list[dict]:
        result: list[dict] = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            rendered = self._render_message(msg)
            if isinstance(rendered, list):
                result.extend(rendered)
            else:
                result.append(rendered)

        return result

    def _render_message(self, msg: Message) -> dict | list[dict]:
        if msg.role == MessageRole.USER:
            content = self._build_content_blocks(msg)
            return {"role": "user", "content": content}

        if msg.role == MessageRole.ASSISTANT:
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

            # Preserve any stored thinking if present (for logging / debugging)
            if msg.thinking:
                result["reasoning"] = msg.thinking

            return result

        if msg.role == MessageRole.TOOL_RESULT:
            return [
                {"role": "tool", "tool_call_id": tr.call_id, "content": tr.content}
                for tr in msg.tool_results
            ]

        if msg.role == MessageRole.SYSTEM:
            return {"role": "system", "content": msg.content}

        raise ValueError(f"Unknown role: {msg.role}")

    def _build_content_blocks(self, msg: Message) -> list[dict] | str:
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
                # OpenRouter supports a generalized 'file' block for some providers.
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
        raw = response.choices[0].message

        msg = Message(
            role=MessageRole.ASSISTANT,
            content=raw.content.strip() if raw.content else None,
        )

        if raw.tool_calls:
            msg.tool_calls = [
                ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                    call_id=tc.id,
                )
                for tc in raw.tool_calls
            ]

        # Best-effort extraction of any reasoning field
        provider_fields = getattr(raw, "provider_specific_fields", None) or {}
        reasoning = (
            provider_fields.get("reasoning")
            or provider_fields.get("reasoning_details")
            or getattr(raw, "reasoning", None)
        )
        if isinstance(reasoning, str) and reasoning:
            msg.thinking = reasoning

        return msg

    async def complete(
        self,
        system_prompt: str,
        messages: list[Message],
        tools: list[dict] | None = None,
        **kwargs,
    ) -> tuple[Message, dict[str, Any]]:
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
        usage = getattr(response, "usage", None)
        if not usage:
            return {}

        result: dict[str, Any] = {
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

        # Best-effort cached token extraction (LiteLLM/OpenAI-style)
        # Some providers expose this as usage.prompt_tokens_details.cached_tokens.
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            result["cache_read"] = getattr(details, "cached_tokens", 0) or 0
            result["cache_write"] = getattr(details, "cache_write_tokens", 0) or 0

        # Fallback: sometimes cached token count is on the usage object.
        if "cache_read" not in result:
            result["cache_read"] = getattr(usage, "cached_tokens", 0) or 0

        return result
