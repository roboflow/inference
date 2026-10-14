"""Selectively exported active blocks for the opt-in V2 catalogue."""

from roboflow_workflows.v2.blocks.language.anthropic_claude.v5 import AnthropicClaudeBlock
from roboflow_workflows.v2.blocks.language.google_gemma.v4 import GoogleGemmaBlock
from roboflow_workflows.v2.blocks.language.spacexai.v3 import SpaceXAIBlock
from roboflow_workflows.v2.blocks.language.google_gemini.v6 import GoogleGeminiBlock
from roboflow_workflows.v2.blocks.language.open_ai.v7 import OpenAIBlock
from roboflow_workflows.v2.blocks.language.openai_compatible.v1 import OpenAICompatibleBlock
from roboflow_workflows.v2.blocks.language.openrouter.v3 import OpenRouterBlock
from roboflow_workflows.v2.blocks.language.kimi_openrouter.v3 import KimiOpenRouterBlock
from roboflow_workflows.v2.blocks.language.meta_vlm.v3 import MetaVlmBlock
from roboflow_workflows.v2.blocks.language.zai_vlm.v2 import ZaiVlmBlock

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
