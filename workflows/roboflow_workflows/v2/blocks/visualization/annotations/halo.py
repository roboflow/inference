"""Glows around instance masks: ``v2/halo_visualization@v2``.

Port of V1 ``roboflow_core/halo_visualization@v2`` (``sv.HaloAnnotator``);
the older ``halo_visualization@v1`` is not ported. Per image::

    colored = black image with masks painted, largest first  (blend.paint_instances)
    glow    = cv2.blur(colored, kernel_size)                  (raster.box_blur)
    glow[union of masks] = 0                     # the glow stays outside objects
    gray    = cv2 BGR2GRAY of glow                # OpenCV's 15-bit fixed point
    alpha   = opacity * gray / max(gray)
    output  = trunc(image * (1 - alpha) + glow * opacity)

An image whose glow is all black (no rows, or ``max(gray) == 0``) is
returned unchanged, as in V1. Pixels inside the masks never change.

Equality with V1, for the same dense masks:

- tested on CPU only, against one OpenCV 4.12 build (arm64, NEON): 645
  cases (kernels 1 to 64, dim colors included) and 2,610 targeted row-end
  probes, with no Halo mismatch. ``box_blur`` rounds each mean as that
  ``cv2.blur`` does. This matters most for dim colors: a glow of 1 level
  can be ``max(gray)``, where ``alpha`` is the full ``opacity``, so 1 level
  of blur rounding would move such a pixel by up to ~160 levels;
- this is not a guarantee for every kernel or backend: for kernels above 16,
  ``cv2.blur`` rounds a few pixels at each row end in float64 and the rest
  in float32, and that split depends on its SIMD backend; ``box_blur`` uses
  float32 everywhere. CUDA and MPS were not verified;
- MPS has no float64 and uses float32, so a pixel may be 1 level lower or
  higher where the float64 sum is within float32 rounding of an integer;
- V1 converts sums above 255 with NumPy's ``uint8`` cast, which wraps on
  common platforms (357 becomes 101); this port saturates them at 255. Such
  sums occur where a bright channel of the glow meets a bright image.

    device  all work: painting, blur, gray, blend; no host read and no wait
    host    only RLE masks are decoded there, as their provider does

Masks are dense full-image masks at ``mask_quality`` (``rows``); a mask view
is materialized once.
"""

from typing import Any, Dict

import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.blend import (
    device_colors,
    paint_instances,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    palette_indices,
    resolve_palette,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    ColorParams,
    MaskQuality,
    Number,
    check_literals,
    checked_choice,
    checked_int,
    checked_mask_quality,
    checked_opacity,
    mask_quality_field,
)
from roboflow_workflows.v2.blocks.visualization.annotations.raster import box_blur
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    MASK_KINDS,
    dense_masks_of,
    placed_in,
    tracker_ids_of,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Ref

# cv2.cvtColor COLOR_BGR2GRAY for uint8 (OpenCV 4.12): 15-bit fixed-point
# weights of R, G, B, rounded; equal to OpenCV on every tested pixel.
_GRAY_WEIGHTS_RGB = (9798, 19235, 3735)
_GRAY_SHIFT = 15


class HaloVisualizationBlock(AnnotationBlock):
    """Draw a soft glow, in each object's color, around its mask on a copy of the image.

    Example: the defaults blur each mask's color over 40 px around the
    object at up to 80% strength; objects themselves are left as they are.
    """

    type = "v2/halo_visualization@v2"
    metadata = {
        "name": "Halo Visualization",
        "port_of": "roboflow_core/halo_visualization@v2",
        "section": "visualization",
    }

    class Params(ColorParams):
        predictions: Ref(*MASK_KINDS) = Field(
            description=(
                "Instance segmentation: dense or RLE InstanceDetections, or a "
                "mask view (SegmentationView). The glow surrounds each mask "
                "(see mask_quality)."
            ),
            examples=["$steps.instance_segmentation.segmentation"],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.8,
            description=(
                "Strength of the glow in [0, 1]: its weight where it is "
                "brightest; 0 leaves the image unchanged."
            ),
            examples=[0.8, "$inputs.opacity"],
        )
        kernel_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=40,
            description=(
                "Side in pixels of the square blur that spreads each mask's "
                "color; larger values give wider, softer glows. >= 1."
            ),
            examples=[40, "$inputs.kernel_size"],
        )
        mask_quality: MaskQuality = mask_quality_field(
            reads="the predictions are instance masks (always for this block)"
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "HaloVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        color_palette: str,
        palette_size: int,
        custom_colors: Any,
        color_axis: str,
        opacity: float,
        kernel_size: int,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Draw the glow around every mask of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: Instance masks (dense, RLE or view).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            opacity: Glow strength.
            kernel_size: Blur side.
            mask_quality: ``accurate`` or ``coarse``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters or predictions of another
                workflow input.
        """
        values = dict(
            color_axis=color_axis,
            opacity=opacity,
            kernel_size=kernel_size,
            mask_quality=mask_quality,
        )
        for name, check in _CHECKS.items():
            check(values[name])
        palette = resolve_palette(
            color_palette, palette_size=palette_size, custom_colors=custom_colors
        )
        canvas = canvas_of(image, block=self.type)
        dense = dense_masks_of(predictions, quality=mask_quality, block=self.type)
        dense = placed_in(dense, image, block=self.type)
        if len(dense.class_id) == 0:
            result = annotated(image, canvas)
            return result

        rows = dense.bboxes_metadata or [{}] * len(dense.class_id)
        indices = palette_indices(
            color_axis, class_id=dense.class_id, tracker_id=tracker_ids_of(rows)
        )
        colors = device_colors(palette, indices.to(canvas.device))
        halo = with_halo(
            canvas,
            dense.mask.to(canvas.device),
            colors,
            opacity=opacity,
            kernel_size=kernel_size,
        )
        result = annotated(image, halo)

        return result


def with_halo(
    canvas: torch.Tensor,
    masks: torch.Tensor,
    colors: torch.Tensor,
    *,
    opacity: float,
    kernel_size: int,
) -> torch.Tensor:
    """Return a copy of ``canvas`` with ``sv.HaloAnnotator``'s glow around the masks.

    Args:
        canvas: ``(3, H, W)`` uint8 RGB image; not modified.
        masks: ``(N, H, W)`` bool masks on the same device, ``N >= 1``.
        colors: ``(N, 3)`` uint8 RGB colors on the same device.
        opacity: Glow strength in [0, 1].
        kernel_size: Blur side, >= 1.

    Returns:
        New contiguous ``(3, H, W)`` uint8 image.
    """
    colored = paint_instances(torch.zeros_like(canvas), masks, colors)
    glow = box_blur(colored, kernel_size).masked_fill(masks.any(dim=0), 0)
    gray = _gray(glow)
    peak = gray.max()

    # V1's float64 arithmetic, in V1's operation order; MPS lacks float64.
    dtype = torch.float32 if canvas.device.type == "mps" else torch.float64
    alpha = (opacity * gray.to(dtype)) / peak.clamp(min=1).to(dtype)
    mixed = canvas.to(dtype) * (1 - alpha) + glow.to(dtype) * opacity
    blended = mixed.clamp(max=255).to(torch.uint8)
    halo = torch.where(peak > 0, blended, canvas).contiguous()

    return halo


def _gray(rgb: torch.Tensor) -> torch.Tensor:
    # (3, H, W) uint8 -> (H, W) int32, as cv2.cvtColor(COLOR_BGR2GRAY) on BGR.
    red, green, blue = rgb.to(torch.int32).unbind(0)
    weight_red, weight_green, weight_blue = _GRAY_WEIGHTS_RGB
    weighted = red * weight_red + green * weight_green + blue * weight_blue
    gray = (weighted + (1 << (_GRAY_SHIFT - 1))) >> _GRAY_SHIFT

    return gray


_CHECKS = {
    "color_axis": lambda value: checked_choice(
        value, name="color_axis", choices=COLOR_AXES
    ),
    "opacity": checked_opacity,
    "kernel_size": lambda value: checked_int(value, name="kernel_size", minimum=1),
    "mask_quality": checked_mask_quality,
}
