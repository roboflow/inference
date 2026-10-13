"""What every annotation block shares: one output, a fresh canvas, the result.

Blocks run once per image on the image's device. The canvas is a contiguous
copy of the input queued on the caller's current stream; nothing here waits
for the device or changes streams. The input image is never written.
"""

from typing import Any, Dict

import torch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.declaration import Block, Output


class AnnotationBlock(Block):
    """Base of the annotation blocks: a pure function of image and parameters."""

    prunable = True
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description=(
                "New annotated image with the input's identity, frame, source, "
                "time and parent row."
            ),
        ),
    }


def canvas_of(image: ImageData, *, block: str) -> torch.Tensor:
    """Return a contiguous copy of an RGB image's pixels to draw on.

    Args:
        image: The block's input image.
        block: Block type for the error message.

    Returns:
        New ``(3, H, W)`` uint8 tensor on the image's device.

    Raises:
        ValueError: When the image does not have 3 channels.
    """
    if image.channels != 3:
        raise ValueError(f"{block} draws on 3-channel RGB images, got {image!r}")

    canvas = image.tensor_image.clone(memory_format=torch.contiguous_format)

    return canvas


def annotated(image: ImageData, canvas: torch.Tensor) -> Dict[str, Any]:
    """Return the block result: ``canvas`` as a new image like ``image``."""
    result = {"image": image.with_pixels(canvas)}

    return result
