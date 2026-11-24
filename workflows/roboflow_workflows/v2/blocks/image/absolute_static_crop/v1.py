"""Crop one rectangle, given by centre and size in pixels, from each image.

Canonical type: ``roboflow_core/absolute_static_crop@v1``.
"""

from typing import Any, Dict
from pydantic import Field, StrictInt
from roboflow_workflows.v2.blocks.image._shared.static_crops import centered_rectangle
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class AbsoluteStaticCropBlock(Block):
    """Crop one rectangle, given by centre and size in pixels, from each image.

    Port of V1 ``roboflow_core/absolute_static_crop@v1``. Example: with
    ``x_center=10`` and ``width=5`` the crop spans ``x`` from
    ``round(7.5) = 8`` to ``13`` (exclusive).
    """

    type = "roboflow_core/absolute_static_crop@v1"
    outputs = {
        "crops": Output(
            IMAGE_KIND,
            source="image",
            description="The crop; absent when the rectangle misses the image.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to crop.")
        x_center: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1, description="Horizontal centre of the crop in pixels."
        )
        y_center: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1, description="Vertical centre of the crop in pixels."
        )
        width: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1, description="Width of the crop in pixels, before clipping."
        )
        height: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=1, description="Height of the crop in pixels, before clipping."
        )

    def run(
        self, *, image: ImageData, x_center: int, y_center: int, width: int, height: int
    ) -> Dict[str, Any]:
        """Crop the centred rectangle from ``image``.

        Args:
            image: Image to crop.
            x_center: Horizontal centre in pixels.
            y_center: Vertical centre in pixels.
            width: Width in pixels.
            height: Height in pixels.

        Returns:
            ``crops``: the clipped crop, or ``None`` when nothing remains.
        """
        rectangle = centered_rectangle(
            x_center, y_center=y_center, width=width, height=height
        )
        # Clips to the image; None when nothing remains.
        result = {"crops": image.crop(rectangle)}

        return result
