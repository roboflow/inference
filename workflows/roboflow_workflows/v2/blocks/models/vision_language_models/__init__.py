"""Selectively exported active blocks for the opt-in V2 catalogue."""

from roboflow_workflows.v2.blocks.models.vision_language_models.anthropic_claude.v5 import (
    AnthropicClaudeBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.google_gemma.v4 import (
    GoogleGemmaBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.spacexai.v3 import (
    SpaceXAIBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.google_gemini.v6 import (
    GoogleGeminiBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.open_ai.v7 import (
    OpenAIBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.openai_compatible.v1 import (
    OpenAICompatibleBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.openrouter.v3 import (
    OpenRouterBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.kimi_openrouter.v3 import (
    KimiOpenRouterBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.meta_vlm.v3 import (
    MetaVlmBlock,
)
from roboflow_workflows.v2.blocks.models.vision_language_models.zai_vlm.v2 import (
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
