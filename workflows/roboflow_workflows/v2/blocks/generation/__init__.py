"""Active generation blocks with caller-selected service access."""

from roboflow_workflows.v2.blocks.generation.stability_ai_image_gen.v1 import StabilityAIImageGenerationBlock
from roboflow_workflows.v2.blocks.generation.stability_ai_inpainting.v1 import StabilityAIInpaintingBlock
from roboflow_workflows.v2.blocks.generation.stability_ai_outpainting.v1 import StabilityAIOutpaintingBlock

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
