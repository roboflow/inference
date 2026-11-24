"""Resize one image to a fixed width and height.

Canonical type: ``roboflow_core/resize@v1``.
"""

from typing import Any, Dict, Literal
from pydantic import Field, StrictInt
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

import roboflow_workflows.v2.blocks.image._shared.blocks as _helpers


class ResizeBlock(Block):
    """Resize one image to a fixed width and height.

    The aspect ratio is not kept. The result has a new ``image_id``; its
    provenance records the actual size ratio, so coordinates found in the
    resized image map back to the input and to the root.
    """

    type = "roboflow_core/resize@v1"
    outputs = {"image": Output(IMAGE_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to resize.")
        width: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1, description="Width of the result in pixels."
        )
        height: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1, description="Height of the result in pixels."
        )
        interpolation: Literal["bilinear", "nearest"] = Field(
            default="bilinear",
            description="Pixel sampling: antialiased bilinear or nearest neighbour.",
        )

    def run(
        self, *, image: ImageData, width: int, height: int, interpolation: str
    ) -> Dict[str, Any]:
        """Resize ``image`` to ``width`` x ``height``.

        Args:
            image: Image to resize.
            width: Width of the result in pixels.
            height: Height of the result in pixels.
            interpolation: ``"bilinear"`` or ``"nearest"``.

        Returns:
            ``image``: the resized image on the input's device.
        """
        resized = image.resize((height, width), interpolation=interpolation)
        result = {"image": resized}

        return result
