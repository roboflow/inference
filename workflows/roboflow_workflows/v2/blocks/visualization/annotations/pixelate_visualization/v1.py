"""Pixelated detections: ``roboflow_core/pixelate_visualization@v1``.

Port of V1 ``roboflow_core/pixelate_visualization@v1``
(``sv.PixelateAnnotator``). Boxes only, as V1; masks are not read. Rows are
pixelated in order on the device::

    region  = box clipped to the frame, integer
    smaller side < pixel_size   region filled with its mean color (truncated)
    otherwise                   bilinear downscale by 1 / pixel_size
                                (round(w / pixel_size) cells), nearest upscale

Cell layout and mean fills equal V1. Cell colors follow OpenCV's sample
positions with float weights; they may differ from its fixed point by 1
level. Host boundary: the box table; every row's resize tables travel to
the device in one copy per call.
"""

from typing import Any, Dict, Optional, Tuple
import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    check_literals,
    checked_int,
    reject_copy_image,
)
from roboflow_workflows.v2.blocks.visualization.annotations.raster import (
    LinearTables,
    NearestTables,
    apply_linear,
    apply_nearest,
    linear_tables,
    nearest_tables,
    upload_tables,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    boxes_of,
    clipped_corners,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class PixelateVisualizationBlock(Block):
    """Pixelate each detection's box in a copy of the image.

    Example: a 100 x 60 box with ``pixel_size=20`` (default) becomes a
    5 x 3 grid of flat color cells.
    """

    type = "roboflow_core/pixelate_visualization@v1"
    prunable = True
    metadata = {
        "name": "Pixelate Visualization",
        "port_of": "roboflow_core/pixelate_visualization@v1",
        "section": "visualization",
    }
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

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(
            description=(
                "RGB image to draw on; not modified. The output is a new image "
                "with the same identity, source and time."
            ),
            examples=["$inputs.image", "$steps.mask_visualization.image"],
        )
        predictions: Ref(*BOX_KINDS) = Field(
            description=(
                "Detections, instance segmentation (boxes are used) or keypoint "
                "predictions with boxes."
            ),
            examples=["$steps.face_detection.predictions"],
        )
        pixel_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=20,
            description="Cell size in pixels; >= 1.",
            examples=[20, "$inputs.pixel_size"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_literals(self) -> "PixelateVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self, *, image: ImageData, predictions: Any, pixel_size: int
    ) -> Dict[str, Any]:
        """Pixelate every box of ``predictions`` in a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (masks are ignored).
            pixel_size: Cell size.

        Returns:
            ``image``: the new image.

        Raises:
            ValueError: On an invalid pixel size, keypoints without boxes or
                predictions of another workflow input.
        """
        pixel_size = _CHECKS["pixel_size"](pixel_size)
        canvas = canvas_of(image, block=self.type)
        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
        )
        regions = [
            (x1, y1, x2, y2)
            for x1, y1, x2, y2 in clipped_corners(read_rows(boxes).xyxy, image.size_hw)
            if x2 > x1 and y2 > y1
        ]
        host = [
            _cell_tables((y2 - y1, x2 - x1), pixel_size) for x1, y1, x2, y2 in regions
        ]
        uploaded = upload_tables(
            [table for pair in host if pair is not None for table in pair],
            device=canvas.device,
        )
        # Mean-filled rows have no tables; resampled rows take theirs in order.
        on_device = iter(zip(uploaded[0::2], uploaded[1::2]))
        # Rows overlap: each region is read after the earlier rows are painted.
        for (x1, y1, x2, y2), pair in zip(regions, host):
            region = canvas[:, y1:y2, x1:x2]
            if pair is None:
                canvas[:, y1:y2, x1:x2] = _mean_filled(region)
            else:
                down, up = next(on_device)
                canvas[:, y1:y2, x1:x2] = apply_nearest(apply_linear(region, down), up)
        result = annotated(image, canvas)

        return result

def _cell_tables(
    size_hw: Tuple[int, int], pixel_size: int
) -> Optional[Tuple[LinearTables, NearestTables]]:
    # sv.PixelateAnnotator: a region smaller than one cell gets its mean color
    # (None); otherwise cv2.resize(fx=1/p) bilinear down, nearest back up.
    # The cell grid size rounds half to even, as Python's round does.
    height, width = size_hw
    if min(height, width) < pixel_size:
        return None

    factor = 1.0 / pixel_size
    small_hw = (round(height * factor), round(width * factor))
    down = linear_tables(size_hw, small_hw, scale_xy=(1.0 / factor, 1.0 / factor))
    up = nearest_tables(small_hw, size_hw)

    return down, up

def _mean_filled(region: torch.Tensor) -> torch.Tensor:
    # Truncated mean in integers (MPS has no float64).
    height, width = int(region.shape[1]), int(region.shape[2])
    mean = (region.sum(dim=(1, 2), dtype=torch.int64) // (height * width)).to(
        torch.uint8
    )
    filled = mean.view(3, 1, 1).expand(3, height, width).clone()

    return filled

_CHECKS = {
    "pixel_size": lambda value: checked_int(value, name="pixel_size", minimum=1),
}
