"""Shared token-count parsing for remote VLM workflow blocks.

Provider APIs already return usage on every call. Blocks previously dropped
it; these helpers normalize the three envelopes we see (chat completions,
Responses/Anthropic, Gemini) into ``(input_tokens, output_tokens)``. Missing
or malformed counts become ``None``, never ``0``, so a caller can tell
"usage omitted" from a real zero-token call.
"""

from typing import Any, List, Optional, Tuple

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import INTEGER_KIND

OptionalTokenCount = Optional[int]


TOKEN_OUTPUT_DEFINITIONS: List[OutputDefinition] = [
    OutputDefinition(name="input_tokens", kind=[INTEGER_KIND]),
    OutputDefinition(name="output_tokens", kind=[INTEGER_KIND]),
]


def as_optional_int(value: Any) -> OptionalTokenCount:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def parse_chat_completion_usage(
    usage: Any,
) -> Tuple[OptionalTokenCount, OptionalTokenCount]:
    """OpenRouter / OpenAI chat-completions: ``prompt_tokens`` / ``completion_tokens``."""
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        return as_optional_int(usage.get("prompt_tokens")), as_optional_int(
            usage.get("completion_tokens")
        )
    return as_optional_int(getattr(usage, "prompt_tokens", None)), as_optional_int(
        getattr(usage, "completion_tokens", None)
    )


def parse_responses_api_usage(
    usage: Any,
) -> Tuple[OptionalTokenCount, OptionalTokenCount]:
    """Anthropic / OpenAI Responses / xAI: ``input_tokens`` / ``output_tokens``."""
    if usage is None:
        return None, None
    if isinstance(usage, dict):
        return as_optional_int(usage.get("input_tokens")), as_optional_int(
            usage.get("output_tokens")
        )
    return as_optional_int(getattr(usage, "input_tokens", None)), as_optional_int(
        getattr(usage, "output_tokens", None)
    )


def parse_gemini_usage_metadata(
    usage_metadata: Any,
) -> Tuple[OptionalTokenCount, OptionalTokenCount]:
    """Gemini ``usageMetadata``: prompt vs candidates + thinking tokens."""
    if usage_metadata is None:
        return None, None
    if isinstance(usage_metadata, dict):
        prompt = as_optional_int(usage_metadata.get("promptTokenCount"))
        candidates = as_optional_int(usage_metadata.get("candidatesTokenCount"))
        thoughts = as_optional_int(usage_metadata.get("thoughtsTokenCount"))
    else:
        prompt = as_optional_int(getattr(usage_metadata, "promptTokenCount", None))
        candidates = as_optional_int(
            getattr(usage_metadata, "candidatesTokenCount", None)
        )
        thoughts = as_optional_int(getattr(usage_metadata, "thoughtsTokenCount", None))
    if candidates is None and thoughts is None:
        output = None
    else:
        output = (candidates or 0) + (thoughts or 0)
    return prompt, output
