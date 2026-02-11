"""Claude API wrapper with tool-use, prompt caching, extended thinking, and streaming."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import anthropic

from inference_agent.core.protocols import LLMResponse, ToolCall, ToolDefinition

logger = logging.getLogger(__name__)

# Retry settings for rate limits (429) and overload (529)
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0


class ClaudeClient:
    """LLMClient implementation using the Anthropic Python SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4096,
    ):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._default_max_tokens = max_tokens
        self._total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }

    async def chat(
        self,
        messages: list[dict],
        tools: list[ToolDefinition],
        system: list[dict],
        max_tokens: int = 0,
        thinking: Optional[dict] = None,
    ) -> LLMResponse:
        """Send a message to Claude with tools, caching, and optional thinking.

        Args:
            messages: Conversation history (user/assistant turns).
            tools: Tool definitions. Cache control is added to the last one.
            system: System prompt as content blocks. Last stable block should
                have cache_control already set by the caller.
            max_tokens: Override default max_tokens (0 = use default).
            thinking: If set, e.g. {"type": "enabled", "budget_tokens": 4096}.
        """
        effective_max_tokens = max_tokens or self._default_max_tokens

        # Convert ToolDefinition objects to Anthropic API format
        tool_defs = _build_tool_defs(tools)

        # Build the request
        request: dict = {
            "model": self._model,
            "max_tokens": effective_max_tokens,
            "system": system,
            "tools": tool_defs,
            "messages": messages,
        }

        if thinking:
            request["thinking"] = thinking
            # Interleaved thinking allows thinking between tool calls
            request["extra_headers"] = {
                "anthropic-beta": "interleaved-thinking-2025-05-14"
            }

        # Retry on rate limit / overload
        response = await self._call_with_retry(request)

        # Parse the response
        return self._parse_response(response)

    async def _call_with_retry(self, request: dict) -> anthropic.types.Message:
        """Call the API with exponential backoff on 429/529."""
        backoff = INITIAL_BACKOFF
        last_error = None
        extra_headers = request.pop("extra_headers", None)
        if extra_headers:
            request["extra_headers"] = extra_headers

        for attempt in range(MAX_RETRIES + 1):
            try:
                return await self._client.messages.create(**request)
            except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "API error (attempt %d/%d): %s. Retrying in %.1fs",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        type(e).__name__,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
        raise last_error  # type: ignore[misc]

    def _parse_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Extract text, tool calls, and thinking from the API response."""
        content_blocks = []
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append({"type": "text", "text": block.text})
                text_parts.append(block.text)
            elif block.type == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))
            elif block.type == "thinking":
                content_blocks.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                })

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_creation_input_tokens": getattr(
                response.usage, "cache_creation_input_tokens", 0
            ),
            "cache_read_input_tokens": getattr(
                response.usage, "cache_read_input_tokens", 0
            ),
        }

        # Accumulate totals
        for key in self._total_usage:
            self._total_usage[key] += usage.get(key, 0)

        return LLMResponse(
            content_blocks=content_blocks,
            text="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            usage=usage,
        )

    @property
    def total_usage(self) -> dict:
        return dict(self._total_usage)

    @staticmethod
    def format_tool_result(
        tool_call_id: str,
        content: str | list | dict,
        is_error: bool = False,
    ) -> dict:
        """Format a tool result message for the conversation history.

        Args:
            tool_call_id: The ID from the tool_use block.
            content: String, list of content blocks, or dict (JSON-serialized).
            is_error: Whether this is an error result.
        """
        if isinstance(content, dict):
            import json
            content = json.dumps(content, default=str)
        if isinstance(content, str):
            result_content = content
        else:
            # List of content blocks (text + images)
            result_content = content

        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result_content,
            "is_error": is_error,
        }


def _build_tool_defs(tools: list[ToolDefinition]) -> list[dict]:
    """Convert ToolDefinition objects to Anthropic API tool dicts.

    Adds cache_control to the LAST tool definition for prompt caching.
    """
    if not tools:
        return []

    defs = []
    for tool in tools:
        d = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        if tool.cache_control:
            d["cache_control"] = tool.cache_control
        defs.append(d)

    # Add prompt caching to the last tool definition
    defs[-1]["cache_control"] = {"type": "ephemeral"}
    return defs
