"""Crop one rectangle, given as fractions of the image size, from each image.

Canonical type: ``roboflow_core/relative_statoic_crop@v1``.
"""

from typing import Any, Dict
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.image._shared.static_crops import centered_rectangle
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class RelativeStaticCropBlock(Block):
    """Crop one rectangle, given as fractions of the image size, from each image.

    Port of V1 ``roboflow_core/relative_statoic_crop@v1`` (V1 type spelling).
    Fractions become pixels first: ``x_center_px = round(W * x_center)`` and
    ``width_px = round(W * width)``, likewise for ``y`` and ``H``; then the
    absolute rule applies.

    Difference from V1: the rectangle is clipped to the image. V1 sliced with
    a negative start, so a rectangle overhanging the left or top edge produced
    no crop or a wrongly placed one. Example on a 100-pixel-wide image:
    ``x_center=0.1, width=0.5`` gives ``x`` from ``-15`` to ``35``; V2 crops
    ``[0, 35)``, V1 returned nothing.
    """

    type = "roboflow_core/relative_statoic_crop@v1"
    outputs = {
        "crops": Output(
            IMAGE_KIND,
            source="image",
            description="The crop; absent when the rectangle misses the image.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to crop.")
        x_center: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=1,
            description=(
                "Horizontal centre of the crop as a fraction of the image "
                "width, 0..1."
            ),
        )
        y_center: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=1,
            description=(
                "Vertical centre of the crop as a fraction of the image "
                "height, 0..1."
            ),
        )
        width: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=1,
            description="Crop width as a fraction of the image width, 0..1.",
        )
        height: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=1,
            description="Crop height as a fraction of the image height, 0..1.",
        )

    def run(
        self,
        *,
        image: ImageData,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
    ) -> Dict[str, Any]:
        """Crop the centred rectangle, given in fractions, from ``image``.

        Args:
            image: Image to crop.
            x_center: Horizontal centre as a fraction of the width.
            y_center: Vertical centre as a fraction of the height.
            width: Width as a fraction of the image width.
            height: Height as a fraction of the image height.

        Returns:
            ``crops``: the clipped crop, or ``None`` when nothing remains.
        """
        image_height, image_width = image.size_hw
        rectangle = centered_rectangle(
            round(image_width * x_center),
            y_center=round(image_height * y_center),
            width=round(image_width * width),
            height=round(image_height * height),
        )
        # Clips to the image; None when nothing remains.
        result = {"crops": image.crop(rectangle)}

        return result
