"""Active generation blocks with caller-selected service access."""

from roboflow_workflows.v2.blocks.generation.stability import (
    StabilityAIImageGenerationBlock,
    StabilityAIInpaintingBlock,
    StabilityAIOutpaintingBlock,
)

BLOCKS = (
    StabilityAIImageGenerationBlock,
    StabilityAIInpaintingBlock,
    StabilityAIOutpaintingBlock,
)

__all__ = [
    "BLOCKS",
    "StabilityAIImageGenerationBlock",
    "StabilityAIInpaintingBlock",
    "StabilityAIOutpaintingBlock",
]
