"""Cover everything outside the detections: ``v2/background_color_visualization@v1``.

Port of V1 ``roboflow_core/background_color_visualization@v1``
(``BackgroundColorAnnotator``)::

    covered = round(color * opacity + image * (1 - opacity))   every pixel
    kept    = union of the instance masks      if the carrier has masks
              union of the boxes [y1:y2, x1:x2] otherwise (ends exclusive)
    output  = image where kept, else covered

The blend is ``blend.blend`` (exact for opacity 0, 0.25, 0.5, 0.75 and 1,
else within 1 level at .5 ties). Kept pixels are V1's, with one deliberate
difference: boxes are clipped to the frame first. V1 slices with the
truncated corners as they are, so a box starting left of or above the
image (x1 or y1 <= -1) indexes from the far edge and keeps the wrong
pixels or none.

    host    boxes: one transfer of boxes, then integer corners
    device  copy of the image; kept region; one blend

Box carriers never read masks. Mask carriers (dense, RLE or a mask view)
build one full-image mask per row at ``mask_quality``.
"""

from typing import Any, Dict

import torch
from pydantic import Field, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.blend import blend
from roboflow_workflows.v2.blocks.visualization.annotations.colors import parse_color
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    AnnotationParams,
    MaskQuality,
    Number,
    check_literals,
    checked_mask_quality,
    checked_opacity,
    mask_quality_field,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    boxes_of,
    clipped_corners,
    dense_masks_of,
    has_masks,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Ref


class BackgroundColorVisualizationBlock(AnnotationBlock):
    """Cover the image outside the detections with a color, on a copy.

    Example: ``color="BLACK", opacity=0.5`` (defaults) darkens everything
    outside the objects by half; ``opacity=1`` leaves only the objects on a
    solid background. Instance masks keep their exact shapes; detections
    without masks keep their boxes.
    """

    type = "v2/background_color_visualization@v1"
    metadata = {
        "name": "Background Color Visualization",
        "port_of": "roboflow_core/background_color_visualization@v1",
        "section": "visualization",
    }

    class Params(AnnotationParams):
        predictions: Ref(*BOX_KINDS) = Field(
            description=(
                "Regions to keep: detections and keypoints with boxes keep "
                "their boxes; instance segmentation (dense, RLE or view) keeps "
                "its masks. Predictions of another frame of the same workflow "
                "input are mapped into this image and clipped to it."
            ),
            examples=[
                "$steps.object_detection.predictions",
                "$steps.instance_segmentation.segmentation",
            ],
        )
        color: str | Ref(STRING_KIND) = Field(
            default="BLACK",
            description=(
                "Color covering the background: #RRGGBB, rgb(R, G, B), "
                "bgr(B, G, R) or a name such as WHITE."
            ),
            examples=["BLACK", "#FFFFFF", "rgb(255, 255, 255)", "$inputs.color"],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.5,
            description=(
                "Weight of the background color, in [0, 1]: 0 leaves the "
                "image, 1 replaces the background."
            ),
            examples=[0.5, "$inputs.opacity"],
        )
        mask_quality: MaskQuality = mask_quality_field(
            reads="the predictions are instance masks"
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "BackgroundColorVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        color: str,
        opacity: float,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Cover ``image`` outside ``predictions``, on a copy.

        Args:
            image: RGB image; not modified.
            predictions: A box or mask carrier (see ``rows``).
            color: Background color string.
            opacity: Background color weight.
            mask_quality: ``accurate`` or ``coarse``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, keypoints without boxes or
                predictions of another workflow input.
        """
        values = dict(color=color, opacity=opacity, mask_quality=mask_quality)
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        kept = self._kept_region(predictions, image, quality=mask_quality)
        layer = torch.tensor(parse_color(color, field="color"), dtype=torch.uint8)
        layer = layer.to(canvas.device).view(3, 1, 1).expand_as(canvas)
        result = annotated(image, blend(canvas, layer, opacity=opacity, region=~kept))

        return result

    def _kept_region(
        self, predictions: Any, image: ImageData, *, quality: str
    ) -> torch.Tensor:
        """Return the ``(H, W)`` bool pixels left uncovered, on the image's device."""
        height, width = image.size_hw
        if has_masks(predictions):
            dense = dense_masks_of(predictions, quality=quality, block=self.type)
            dense = placed_in(dense, image, block=self.type)
            # No rows give an all-False (H, W) union.
            kept = dense.mask.to(image.device).any(dim=0)
            return kept

        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
        )
        rows = read_rows(boxes)
        kept = torch.zeros((height, width), dtype=torch.bool, device=image.device)
        for x1, y1, x2, y2 in clipped_corners(rows.xyxy, image.size_hw):
            kept[y1:y2, x1:x2] = True

        return kept


_CHECKS = {
    "color": lambda value: parse_color(value, field="color"),
    "opacity": checked_opacity,
    "mask_quality": checked_mask_quality,
}
