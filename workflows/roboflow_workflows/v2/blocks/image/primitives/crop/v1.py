"""Crop rectangles out of one image.

Canonical type: ``roboflow_core/crop@v1``.
"""

from typing import Annotated, Any, Dict, List, Sequence, Tuple
from pydantic import AfterValidator, Field, StrictInt
from roboflow_workflows.v2.blocks.image._shared.blocks import (
    Rectangle,
    _require_rectangles,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import (
    CROP_SUMMARY_KIND,
    IMAGE_KIND,
)
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class CropBlock(Block):
    """Crop rectangles out of one image.

    Rectangles are ``[x0, y0, x1, y1]`` pixel coordinates with exclusive upper
    bounds. Each rectangle is clipped to the image. A rectangle that is empty
    after clipping is omitted from ``crops``; surviving crops keep the position
    of their rectangle as local index, so ``crops`` may be sparse, for example
    ``(0,)`` and ``(2,)``. Crops are contiguous copies on the image's device,
    each with a new ``image_id``; their provenance places them in the image
    and in its root.

    ``crops`` is dynamic nesting: which rectangles survive depends on the size
    of each image.

    Not a V1 port. The V1 crop by detections is ``roboflow_core/dynamic_crop@v1``.
    """

    type = "roboflow_core/crop@v1"
    outputs = {
        "crops": Output(
            IMAGE_KIND,
            expand="regions",
            source="image",
            description="Crops indexed by the position of their rectangle.",
        ),
        "summary": Output(
            CROP_SUMMARY_KIND,
            source="image",
            description="Image size, crop count and the kept rectangle positions.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to crop.")
        regions: Annotated[
            List[Tuple[StrictInt, StrictInt, StrictInt, StrictInt]]
            | Ref(LIST_OF_VALUES_KIND),
            AfterValidator(_require_rectangles),
        ] = Field(
            description=(
                "Rectangles [x0, y0, x1, y1] in pixels with x0 <= x1 and "
                "y0 <= y1. Upper bounds are exclusive."
            ),
            examples=[[[0, 0, 64, 64], [32, 32, 96, 96]]],
        )

    def run(self, *, image: ImageData, regions: Sequence[Rectangle]) -> Dict[str, Any]:
        """Crop ``regions`` from ``image``.

        Args:
            image: Image to crop.
            regions: Rectangles ``(x0, y0, x1, y1)`` in order.

        Returns:
            ``crops``: a ``Batch`` of crops indexed by the positions of the
            rectangles that survived clipping. ``summary``: a mapping with
            ``crop_count``, ``image_height``, ``image_width``, ``kept_regions``
            and ``crop_dimensions``.
        """
        crops: List[ImageData] = []
        kept_regions: List[int] = []
        for position, region in enumerate(regions):
            crop = image.crop(region)
            if crop is None:
                continue
            crops.append(crop)
            kept_regions.append(position)

        summary = {
            "crop_count": len(crops),
            "image_height": image.height,
            "image_width": image.width,
            "kept_regions": kept_regions,
            "crop_dimensions": [list(crop.size_hw) for crop in crops],
        }
        indices = [(position,) for position in kept_regions]
        result = {"crops": Batch.of(crops, indices=indices), "summary": summary}

        return result
