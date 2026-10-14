"""Convert an RGB image to one grayscale channel: ``roboflow_core/convert_grayscale@v1``.

Port of V1 ``roboflow_core/convert_grayscale@v1``. The pixels stay on the
image's device and match OpenCV's ``cv2.cvtColor(..., COLOR_BGR2GRAY)`` on the
same colors exactly. OpenCV 4.x computes ``uint8`` gray in fixed point::

    gray = (9798 * R + 19235 * G + 3735 * B + 2**14) >> 15

The sum stays below ``2**23``, so ``int32`` is exact on every device.

The result is the same image with new pixels: ``image_id``, parent and root
mappings, source and time are kept, and the input tensor is not modified.

Difference from V1: V1 raised an OpenCV error for an image that already has
one channel. Its description promised that such an image comes back
unchanged, and this port does that: it returns the input image itself.
"""

from typing import Any, Dict
import torch
from pydantic import Field
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


RED_WEIGHT_Q15 = 9798

GREEN_WEIGHT_Q15 = 19235

BLUE_WEIGHT_Q15 = 3735

Q15_SHIFT = 15

Q15_HALF = 1 << (Q15_SHIFT - 1)

class ConvertGrayscaleBlock(Block):
    """Convert one RGB image to a single grayscale channel.

    Example: a pure red pixel ``(255, 0, 0)`` becomes ``76`` and pure green
    becomes ``150``. A one-channel image is returned unchanged.
    """

    type = "roboflow_core/convert_grayscale@v1"
    outputs = {
        "image": Output(
            IMAGE_KIND,
            source="image",
            description="One-channel image with the input's identity and geometry.",
        )
    }
    prunable = True

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description="RGB image to convert; a one-channel image passes through."
        )

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Convert ``image`` to grayscale.

        Args:
            image: RGB or already one-channel image.

        Returns:
            ``image``: the one-channel image.
        """
        if image.channels == 1:
            return {"image": image}

        gray = image.with_pixels(rgb_to_gray(image.tensor_image))
        result = {"image": gray}

        return result

def rgb_to_gray(pixels: torch.Tensor) -> torch.Tensor:
    """Weigh RGB channels into gray exactly as OpenCV does for ``uint8``.

    Args:
        pixels: ``(3, height, width)`` ``uint8`` RGB tensor on any device.

    Returns:
        A new ``(1, height, width)`` ``uint8`` tensor on the same device.
    """
    channels = pixels.to(torch.int32)
    weighted = (
        channels[0] * RED_WEIGHT_Q15
        + channels[1] * GREEN_WEIGHT_Q15
        + channels[2] * BLUE_WEIGHT_Q15
        + Q15_HALF
    ) >> Q15_SHIFT
    gray = weighted.to(torch.uint8).unsqueeze(0)

    return gray
