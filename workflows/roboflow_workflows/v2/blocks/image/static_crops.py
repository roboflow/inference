"""Centred static crops: one rectangle per image, in pixels or in fractions.

* ``v2/absolute_static_crop@v1``: centre and size in pixels.
* ``v2/relative_static_crop@v1``: centre and size as fractions of the image.

Both ports keep V1's rectangle arithmetic, including Python ``round`` (half to
even)::

    x_min = round(x_center - width / 2)      x_max = x_min + width
    y_min = round(y_center - height / 2)     y_max = y_min + height

The rectangle is clipped to the image. When nothing remains, the step has no
crop for that image and downstream steps skip it. A crop is a new image whose
provenance places it in its parent and root; source and time are kept.
Neither block adds a nesting axis: one image in, at most one crop out.
"""

from typing import Any, Dict, Tuple

from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

Rectangle = Tuple[int, int, int, int]

RELATIVE_FIELD_DESCRIPTION = "{} as a fraction of the image {}, 0..1."


class AbsoluteStaticCropBlock(Block):
    """Crop one rectangle, given by centre and size in pixels, from each image.

    Port of V1 ``roboflow_core/absolute_static_crop@v1``. Example: with
    ``x_center=10`` and ``width=5`` the crop spans ``x`` from
    ``round(7.5) = 8`` to ``13`` (exclusive).
    """

    type = "v2/absolute_static_crop@v1"
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

    type = "v2/relative_static_crop@v1"
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
            description=RELATIVE_FIELD_DESCRIPTION.format(
                "Horizontal centre of the crop", "width"
            ),
        )
        y_center: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=1,
            description=RELATIVE_FIELD_DESCRIPTION.format(
                "Vertical centre of the crop", "height"
            ),
        )
        width: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=1,
            description=RELATIVE_FIELD_DESCRIPTION.format("Crop width", "width"),
        )
        height: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            ge=0,
            le=1,
            description=RELATIVE_FIELD_DESCRIPTION.format("Crop height", "height"),
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


def centered_rectangle(
    x_center: int, *, y_center: int, width: int, height: int
) -> Rectangle:
    """Return the V1 rectangle ``(x_min, y_min, x_max, y_max)`` around a centre.

    Args:
        x_center: Horizontal centre in pixels.
        y_center: Vertical centre in pixels.
        width: Width in pixels.
        height: Height in pixels.

    Returns:
        Corners in pixels with exclusive upper bounds; not clipped. ``round``
        is Python's half-to-even rounding, as in V1.
    """
    x_min = round(x_center - width / 2)
    y_min = round(y_center - height / 2)
    rectangle = (x_min, y_min, x_min + width, y_min + height)

    return rectangle
