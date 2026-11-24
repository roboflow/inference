"""Crop the same configured rectangles out of every image.

Canonical type: ``roboflow_core/static_crop@v1``.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple
from pydantic import AfterValidator, Field, StrictInt
from roboflow_workflows.v2.blocks.image._shared.blocks import (
    Rectangle,
    _require_rectangles,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class StaticCropBlock(Block):
    """Crop the same configured rectangles out of every image.

    Rectangles are ``[x0, y0, x1, y1]`` pixel coordinates with exclusive upper
    bounds, clipped to the image. Unlike ``roboflow_core/crop@v1``, every rectangle keeps its
    position: crop ``k`` always comes from rectangle ``k``. That is why
    ``crops`` is declared stationary, so a window operator may collect each
    region over time (``[N, regions] -> [N, regions, T]``).

    A rectangle entirely outside an image never shifts the others. With
    ``outside="error"`` (default) the step fails and names the rectangle; with
    ``outside="none"`` that position holds ``None``, which downstream blocks
    skip like any missing value.

    Not a V1 port. The V1 static crops are ``roboflow_core/absolute_static_crop@v1``
    and ``roboflow_core/relative_statoic_crop@v1``.
    """

    type = "roboflow_core/static_crop@v1"
    outputs = {
        "crops": Output(
            IMAGE_KIND,
            expand="regions",
            stationary=True,
            source="image",
            description="One crop per configured rectangle, in configured order.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to crop.")
        regions: Annotated[
            List[Tuple[StrictInt, StrictInt, StrictInt, StrictInt]],
            AfterValidator(_require_rectangles),
        ] = Field(
            description=(
                "Fixed rectangles [x0, y0, x1, y1] in pixels with x0 <= x1 and "
                "y0 <= y1. Upper bounds are exclusive. Written in the workflow, "
                "never selected, so positions mean the same region every time."
            ),
            examples=[[[0, 0, 64, 64], [32, 32, 96, 96]]],
        )
        outside: Literal["error", "none"] = Field(
            default="error",
            description=(
                "What a rectangle entirely outside the image produces: a step "
                "error, or None at its position."
            ),
        )

    def run(
        self, *, image: ImageData, regions: Sequence[Rectangle], outside: str
    ) -> Dict[str, Any]:
        """Crop every configured rectangle from ``image``.

        Args:
            image: Image to crop.
            regions: Rectangles ``(x0, y0, x1, y1)`` in configured order.
            outside: ``"error"`` or ``"none"``, for rectangles outside the image.

        Returns:
            ``crops``: a ``Batch`` with one entry per rectangle at its position;
            ``None`` for a rectangle outside the image when ``outside="none"``.

        Raises:
            ValueError: When a rectangle is outside the image and
                ``outside="error"``.
        """
        crops: List[Optional[ImageData]] = []
        for position, region in enumerate(regions):
            crop = image.crop(region)
            if crop is None and outside == "error":
                raise ValueError(
                    f"rectangle {position} {list(region)} lies outside the "
                    f"{image.width}x{image.height} image; set outside='none' to "
                    "keep its position empty"
                )
            crops.append(crop)

        result = {"crops": Batch.of(crops)}

        return result
