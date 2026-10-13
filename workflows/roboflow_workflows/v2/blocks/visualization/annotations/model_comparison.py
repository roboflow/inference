"""Two models' predictions compared: ``v2/model_comparison_visualization@v1``.

Port of V1 ``roboflow_core/model_comparison_visualization@v1``. Each pixel is
sorted by which model covers it, then blended once::

    covered_a = union of A's instance masks (mask carriers) or boxes (others)
    covered_b = the same for B

    only A   -> color_a            \\
    only B   -> color_b             }  round(color * opacity + pixel * (1 - opacity))
    neither  -> background_color   /
    both     -> unchanged

Empty predictions cover nothing, so two empty inputs tint the whole image
with ``background_color``, as in V1. Masks are read at ``mask_quality``
only for mask carriers; a box carrier never touches masks.

Everything runs on the image's device; nothing is copied to the host. The
blend is ``blend.blend``: equal to V1 for opacity 0, 0.25, 0.5, 0.75 and 1,
otherwise within 1 level at exact .5 ties (V1's default 0.7 included).

Differences from V1:

- A box is clipped to the image. V1 truncates box corners to integers and
  slices with them, so a negative corner counts from the far edge: such a box
  covered nothing (or a wrong band) in V1.
- ``copy_image`` is rejected (see ``params``); predictions of another frame
  of the same input are mapped into the image (see ``rows.placed_in``).
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
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    RGB,
    parse_color,
)
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
    dense_masks_of,
    has_masks,
    placed_in,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Ref

_PREDICTIONS_DESCRIPTION = (
    "Detections, instance segmentation (dense, RLE or view) or keypoint "
    "predictions with boxes, of this image or of another frame of the same "
    "workflow input. Instance masks are compared when present, boxes otherwise."
)


class ModelComparisonVisualizationBlock(AnnotationBlock):
    """Color where only one of two models predicted something.

    Port of V1 roboflow_core/model_comparison_visualization@v1.

    Example: connect ``predictions_a: $steps.model_a.predictions`` and
    ``predictions_b: $steps.model_b.predictions``. Objects only model A
    found turn green, objects only model B found turn red, objects both
    found stay as they are and everything else is darkened.
    """

    type = "v2/model_comparison_visualization@v1"
    metadata = {
        "name": "Model Comparison Visualization",
        "port_of": "roboflow_core/model_comparison_visualization@v1",
        "section": "visualization",
    }

    class Params(AnnotationParams):
        predictions_a: Ref(*BOX_KINDS) = Field(
            description=f"Predictions of model A. {_PREDICTIONS_DESCRIPTION}",
            examples=["$steps.model_a.predictions"],
        )
        color_a: str | Ref(STRING_KIND) = Field(
            default="GREEN",
            description=(
                "Color of pixels only model A covers: #RRGGBB, rgb(R, G, B), "
                "bgr(B, G, R) or a name."
            ),
            examples=["GREEN", "#00FF00", "$inputs.color_a"],
        )
        predictions_b: Ref(*BOX_KINDS) = Field(
            description=f"Predictions of model B. {_PREDICTIONS_DESCRIPTION}",
            examples=["$steps.model_b.predictions"],
        )
        color_b: str | Ref(STRING_KIND) = Field(
            default="RED",
            description="Color of pixels only model B covers; formats as color_a.",
            examples=["RED", "#FF0000", "$inputs.color_b"],
        )
        background_color: str | Ref(STRING_KIND) = Field(
            default="BLACK",
            description="Color of pixels neither model covers; formats as color_a.",
            examples=["BLACK", "#808080", "$inputs.background_color"],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.7,
            description=(
                "Weight of the colors in [0, 1]; 0 leaves the image unchanged. "
                "Pixels both models cover are never changed."
            ),
            examples=[0.7, "$inputs.opacity"],
        )
        mask_quality: MaskQuality = mask_quality_field(
            reads="predictions_a or predictions_b carry instance masks"
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "ModelComparisonVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        predictions_a: Any,
        color_a: str,
        predictions_b: Any,
        color_b: str,
        background_color: str,
        opacity: float,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Color the pixels the two models disagree on, on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions_a: Model A's box or mask carrier (see ``rows``).
            color_a: Color of pixels only A covers.
            predictions_b: Model B's box or mask carrier.
            color_b: Color of pixels only B covers.
            background_color: Color of pixels neither covers.
            opacity: Color weight in [0, 1].
            mask_quality: Mask quality for mask carriers.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, keypoints without boxes, masks
                of another size than the image, or predictions of another
                workflow input.
        """
        values = dict(
            color_a=color_a,
            color_b=color_b,
            background_color=background_color,
            opacity=opacity,
            mask_quality=mask_quality,
        )
        for name, check in _CHECKS.items():
            check(values[name])

        canvas = canvas_of(image, block=self.type)
        covered_a = self._coverage(predictions_a, image, mask_quality=mask_quality)
        covered_b = self._coverage(predictions_b, image, mask_quality=mask_quality)
        only_a = covered_a & ~covered_b
        only_b = covered_b & ~covered_a

        colors = {
            name: _broadcast_color(canvas, parse_color(values[name], field=name))
            for name in ("color_a", "color_b", "background_color")
        }
        layer = torch.where(only_a, colors["color_a"], colors["background_color"])
        layer = torch.where(only_b, colors["color_b"], layer)
        both = covered_a & covered_b
        blended = blend(canvas, layer, opacity=float(opacity), region=~both)
        result = annotated(image, blended)

        return result

    def _coverage(
        self, predictions: Any, image: ImageData, *, mask_quality: str
    ) -> torch.Tensor:
        # (H, W) bool on the image's device: the union of masks or boxes.
        height, width = image.size_hw
        device = image.tensor_image.device
        if has_masks(predictions):
            dense = dense_masks_of(predictions, quality=mask_quality, block=self.type)
            masks = placed_in(dense, image, block=self.type).mask
            if tuple(masks.shape[1:]) != (height, width):
                raise ValueError(
                    f"{self.type}: masks of (h, w)={tuple(masks.shape[1:])} do "
                    f"not match the image's {(height, width)}"
                )
            coverage = masks.to(device=device, dtype=torch.bool).any(dim=0)

            return coverage

        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
        )
        # V1 truncates corners toward zero (astype(int)); comparing with
        # pixel indices then clips the box to the image.
        corners = boxes.xyxy.reshape(-1, 4).to(device).to(torch.int64)
        xs = torch.arange(width, device=device)
        ys = torch.arange(height, device=device)
        in_x = (xs >= corners[:, 0:1]) & (xs < corners[:, 2:3])
        in_y = (ys >= corners[:, 1:2]) & (ys < corners[:, 3:4])
        # (H, N) @ (N, W): how many boxes cover each pixel; counts are exact.
        counts = in_y.T.to(torch.float32) @ in_x.to(torch.float32)
        coverage = counts > 0

        return coverage


def _broadcast_color(canvas: torch.Tensor, rgb: RGB) -> torch.Tensor:
    # (3, 1, 1) on the canvas's device: broadcasts over a (3, H, W) image.
    color = torch.tensor(rgb, dtype=torch.uint8, device=canvas.device).view(3, 1, 1)

    return color


_CHECKS = {
    "color_a": lambda value: parse_color(value, field="color_a"),
    "color_b": lambda value: parse_color(value, field="color_b"),
    "background_color": lambda value: parse_color(value, field="background_color"),
    "opacity": checked_opacity,
    "mask_quality": checked_mask_quality,
}
