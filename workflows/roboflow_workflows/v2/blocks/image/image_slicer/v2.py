"""Overlapping equal-size slices of an image, for small-object detection (SAHI).

``roboflow_core/image_slicer@v2`` ports V1 ``roboflow_core/image_slicer@v2``. The slice
grid is V1's, number for number::

    slice_w = min(slice_width, W)            overlap_w = int(ratio_w * slice_w)
    x anchors: 0, stride, 2 * stride, ... < W  with stride = slice_w - overlap_w
    each anchor moves left to W - slice_w when it would overhang the image
    the same for y; slices run row by row (y outer, x inner); duplicates dropped

So every slice has the same size and lies inside the image. Example: a
1000x600 image with 640x640 slices and ratio 0.2 gives two 640x600 slices,
``[0, 0, 640, 600]`` and ``[360, 0, 1000, 600]``.

Each slice is a crop: provenance places it in the image and its root, so
predictions on slices map back (see ``roboflow_core/detections_stitch@v1``).
"""

from typing import Any, Dict, List, Sequence, Tuple
from pydantic import Field, StrictFloat, StrictInt
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


Rectangle = Tuple[int, int, int, int]

class ImageSlicerBlock(Block):
    """Cut an image into overlapping slices of one size.

    ``slices`` is a new ``regions`` axis: slice ``k`` is the ``k``-th rectangle
    of the grid described in the module docstring. Positions depend on the
    image size, so the axis is not declared stationary. Slices are contiguous
    copies on the image's device.
    """

    type = "roboflow_core/image_slicer@v2"
    outputs = {
        "slices": Output(
            IMAGE_KIND,
            expand="regions",
            source="image",
            description="Slices in grid order, row by row.",
        ),
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image to slice.")
        slice_width: StrictInt | Ref(INTEGER_KIND) = Field(
            default=640,
            ge=1,
            description=(
                "Slice width in pixels; capped at the image width. Ideally the "
                "detection model's input width."
            ),
        )
        slice_height: StrictInt | Ref(INTEGER_KIND) = Field(
            default=640,
            ge=1,
            description=(
                "Slice height in pixels; capped at the image height. Ideally the "
                "detection model's input height."
            ),
        )
        overlap_ratio_width: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.2,
            ge=0,
            lt=1,
            description="Fraction of the slice width shared by horizontal neighbours.",
        )
        overlap_ratio_height: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.2,
            ge=0,
            lt=1,
            description="Fraction of the slice height shared by vertical neighbours.",
        )

    def run(
        self,
        *,
        image: ImageData,
        slice_width: int,
        slice_height: int,
        overlap_ratio_width: float,
        overlap_ratio_height: float,
    ) -> Dict[str, Any]:
        """Slice ``image``.

        Args:
            image: Image to slice.
            slice_width: Slice width in pixels.
            slice_height: Slice height in pixels.
            overlap_ratio_width: Horizontal overlap fraction, ``[0, 1)``.
            overlap_ratio_height: Vertical overlap fraction, ``[0, 1)``.

        Returns:
            ``slices``: a ``Batch`` of slices indexed ``(0,)``, ``(1,)``, ...
        """
        rectangles = slice_rectangles(
            image.size_hw,
            slice_wh=(slice_width, slice_height),
            overlap_ratio_wh=(overlap_ratio_width, overlap_ratio_height),
        )
        slices = [image.crop(rectangle) for rectangle in rectangles]
        result = {"slices": Batch.of(slices)}

        return result

def slice_rectangles(
    image_hw: Sequence[int],
    *,
    slice_wh: Sequence[int],
    overlap_ratio_wh: Sequence[float],
) -> List[Rectangle]:
    """Return the V1 ``image_slicer@v2`` slice grid for one image size.

    Args:
        image_hw: Image ``(height, width)`` in pixels.
        slice_wh: Requested slice ``(width, height)`` in pixels.
        overlap_ratio_wh: Overlap fractions ``(width, height)`` in ``[0, 1)``.

    Returns:
        Rectangles ``(x_min, y_min, x_max, y_max)`` row by row, without
        duplicates; all of one size and inside the image.
    """
    image_height, image_width = image_hw
    x_anchors, slice_width = _axis_anchors(
        image_width, slice_size=slice_wh[0], overlap_ratio=overlap_ratio_wh[0]
    )
    y_anchors, slice_height = _axis_anchors(
        image_height, slice_size=slice_wh[1], overlap_ratio=overlap_ratio_wh[1]
    )
    rectangles: Dict[Rectangle, None] = {}
    for y_min in y_anchors:
        for x_min in x_anchors:
            rectangle = (x_min, y_min, x_min + slice_width, y_min + slice_height)
            rectangles.setdefault(rectangle, None)

    return list(rectangles)

def _axis_anchors(
    image_size: int, *, slice_size: int, overlap_ratio: float
) -> Tuple[List[int], int]:
    # Anchors step by the stride; an anchor overhanging the end moves back so
    # its slice ends at the image edge.
    slice_size = min(slice_size, image_size)
    stride = slice_size - int(overlap_ratio * slice_size)
    anchors = [
        min(start, image_size - slice_size) for start in range(0, image_size, stride)
    ]

    return anchors, slice_size
