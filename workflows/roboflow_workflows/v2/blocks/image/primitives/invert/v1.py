"""Invert every pixel of one image (``255 - value``).

Canonical type: ``roboflow_core/invert@v1``.
"""

from typing import Any, Dict
from pydantic import Field
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

import roboflow_workflows.v2.blocks.image._shared.blocks as _helpers


class InvertBlock(Block):
    """Invert every pixel of one image (``255 - value``).

    Returns a new tensor with the input's identity and provenance; the input
    image is not modified.
    """

    type = "roboflow_core/invert@v1"
    outputs = {"image": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to invert.")

    def run(self, *, image: ImageData) -> Dict[str, Any]:
        """Invert ``image``.

        Args:
            image: Image to invert.

        Returns:
            ``image``: the same image with pixels ``255 - value``.
        """
        inverted = image.with_pixels(255 - image.tensor_image)
        result = {"image": inverted}

        return result
