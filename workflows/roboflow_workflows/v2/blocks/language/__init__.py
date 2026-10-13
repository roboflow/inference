"""Selectively exported active blocks for the opt-in V2 catalogue."""

from roboflow_workflows.v2.blocks.language.anthropic import (
    AnthropicClaudeBlock,
)
from roboflow_workflows.v2.blocks.language.compatible_vendors import (
    GoogleGemmaBlock,
    SpaceXAIBlock,
)
from roboflow_workflows.v2.blocks.language.gemini import (
    GoogleGeminiBlock,
)
from roboflow_workflows.v2.blocks.language.openai import (
    OpenAIBlock,
    OpenAICompatibleBlock,
)
from roboflow_workflows.v2.blocks.language.openrouter import (
    OpenRouterBlock,
)
from roboflow_workflows.v2.blocks.language.openrouter_vendors import (
    KimiOpenRouterBlock,
    MetaVlmBlock,
    ZaiVlmBlock,
)

BLOCKS = (
    OpenRouterBlock,
    KimiOpenRouterBlock,
    MetaVlmBlock,
    ZaiVlmBlock,
    GoogleGemmaBlock,
    SpaceXAIBlock,
    OpenAIBlock,
    OpenAICompatibleBlock,
    AnthropicClaudeBlock,
    GoogleGeminiBlock,
)

__all__ = [
    "BLOCKS",
    "OpenRouterBlock",
    "KimiOpenRouterBlock",
    "MetaVlmBlock",
    "ZaiVlmBlock",
    "GoogleGemmaBlock",
    "SpaceXAIBlock",
    "OpenAIBlock",
    "OpenAICompatibleBlock",
    "AnthropicClaudeBlock",
    "GoogleGeminiBlock",
]
