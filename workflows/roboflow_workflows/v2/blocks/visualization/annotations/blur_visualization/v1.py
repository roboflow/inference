"""Blurred detections: ``roboflow_core/blur_visualization@v1``.

Port of V1 ``roboflow_core/blur_visualization@v1`` (V1's mask-aware blur).
Rows are blurred in order on the device; a later row blurs earlier results::

    boxes   box grown by padding, clipped; cv2.blur of that region
    masks   region = box and the mask's pixel extent, grown by padding;
            mask grown by padding (elliptic kernel); cv2.blur of the region,
            copied back only where the (grown) mask is set

Pixels equal V1 for odd ``kernel_size`` (default 15). For even sizes
OpenCV's 8-bit box filter rounds some halves differently; pixels may differ
by 1 level. Host boundary: the box table, plus for mask carriers one
transfer of the masks' pixel extents and, with padding, one upload of the
growth kernel per call.
"""

from typing import Any, Dict, Literal
import numpy as np
import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    check_literals,
    checked_int,
    checked_mask_quality,
    reject_copy_image,
)
from roboflow_workflows.v2.blocks.visualization.annotations.raster import (
    box_blur,
    elliptic_kernel,
    grow_mask,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    MASK_QUALITIES,
    boxes_of,
    clipped_corners,
    dense_masks_of,
    has_masks,
    mask_extents,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class BlurVisualizationBlock(Block):
    """Blur each detection (its mask when present) in a copy of the image.

    Example: faces from a detector with ``kernel_size=15`` (default) are
    averaged over 15 x 15 pixel windows; a segmentation blurs only mask pixels.
    """

    type = "roboflow_core/blur_visualization@v1"
    prunable = True
    metadata = {
        "name": "Blur Visualization",
        "port_of": "roboflow_core/blur_visualization@v1",
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
                "Detections, instance segmentation (dense, RLE or view) or keypoint "
                "predictions with boxes. Masks are followed when present, which "
                "builds one full-image mask per detection (see mask_quality)."
            ),
            examples=["$steps.face_detection.predictions"],
        )
        kernel_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=15,
            description="Side of the averaging window in pixels; >= 1.",
            examples=[15, "$inputs.kernel_size"],
        )
        padding: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description="Pixels added around each box or mask; >= 0.",
            examples=[0, 10],
        )
        mask_quality: Literal[MASK_QUALITIES] | Ref(STRING_KIND) = Field(
            default="accurate",
            description=(
                "How instance masks are drawn; used when the predictions carry "
                "instance masks (dense, RLE or a mask view). Each read builds one "
                "full-image mask per detection, at either quality. accurate: the masks "
                "the model defines (a low-resolution mask view is upscaled with the "
                "model's own interpolation). coarse: a mask view's grid cells are "
                "thresholded, then expanded, so edges can be blocky and small parts "
                "can be lost; it skips the accurate upscaling only. "
                "roboflow_core/roboflow_instance_segmentation_model@v4 currently "
                "returns full-resolution masks, so both give the same result for it; "
                "they differ only for producers of compact mask views."
            ),
            examples=["accurate", "coarse", "$inputs.mask_quality"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_literals(self) -> "BlurVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        kernel_size: int,
        padding: int,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Blur every row of ``predictions`` in a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier; masks are followed when present.
            kernel_size: Averaging window side.
            padding: Growth of boxes and masks.
            mask_quality: ``accurate`` or ``coarse`` masks.

        Returns:
            ``image``: the new image.

        Raises:
            ValueError: On invalid parameters, keypoints without boxes or
                predictions of another workflow input.
        """
        kernel_size = _CHECKS["kernel_size"](kernel_size)
        padding = _CHECKS["padding"](padding)
        _CHECKS["mask_quality"](mask_quality)
        canvas = canvas_of(image, block=self.type)
        if has_masks(predictions):
            dense = dense_masks_of(predictions, quality=mask_quality, block=self.type)
            dense = placed_in(dense, image, block=self.type)
            _blur_masks(canvas, read_rows(dense).xyxy, dense.mask, kernel_size, padding)
        else:
            boxes = placed_in(
                boxes_of(predictions, block=self.type), image, block=self.type
            )
            _blur_boxes(canvas, read_rows(boxes).xyxy, kernel_size, padding)
        result = annotated(image, canvas)

        return result

def _blur_boxes(
    canvas: torch.Tensor, xyxy: np.ndarray, kernel_size: int, padding: int
) -> None:
    if padding:
        xyxy = xyxy + np.array([-1, -1, 1, 1], dtype=np.float64) * padding
    for x1, y1, x2, y2 in clipped_corners(xyxy, canvas.shape[1:]):
        if x2 <= x1 or y2 <= y1:
            continue
        canvas[:, y1:y2, x1:x2] = box_blur(canvas[:, y1:y2, x1:x2], kernel_size)

def _blur_masks(
    canvas: torch.Tensor,
    xyxy: np.ndarray,
    masks: torch.Tensor,
    kernel_size: int,
    padding: int,
) -> None:
    height, width = int(canvas.shape[1]), int(canvas.shape[2])
    masks = masks.to(canvas.device)
    extents = mask_extents(masks)
    growth = None
    if padding > 0 and any(extent is not None for extent in extents):
        growth = elliptic_kernel(padding, device=canvas.device)

    for index, ((x1, y1, x2, y2), extent) in enumerate(
        zip(clipped_corners(xyxy, canvas.shape[1:]), extents)
    ):
        if extent is None:
            continue
        left, top, right, bottom = extent
        region_x1 = max(min(x1, left) - padding, 0)
        region_y1 = max(min(y1, top) - padding, 0)
        region_x2 = min(max(x2, right) + padding, width)
        region_y2 = min(max(y2, bottom) + padding, height)
        if region_x2 <= region_x1 or region_y2 <= region_y1:
            continue
        inside = masks[index, region_y1:region_y2, region_x1:region_x2]
        if padding > 0:
            inside = grow_mask(inside, padding, kernel=growth)
        region = canvas[:, region_y1:region_y2, region_x1:region_x2]
        blurred = box_blur(region, kernel_size)
        canvas[:, region_y1:region_y2, region_x1:region_x2] = torch.where(
            inside, blurred, region
        )

_CHECKS = {
    "kernel_size": lambda value: checked_int(value, name="kernel_size", minimum=1),
    "padding": lambda value: checked_int(value, name="padding", minimum=0),
    "mask_quality": checked_mask_quality,
}
