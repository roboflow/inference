"""Instance mask overlays: ``v2/mask_visualization@v1``.

Port of V1 ``roboflow_core/mask_visualization@v1`` (``sv.MaskAnnotator``)::

    masks   = dense_masks_of(predictions, quality=mask_quality)   (rows)
    colored = image with masks painted, largest first     (blend.paint_instances)
    output  = round(colored * opacity + image * (1 - opacity))

``accurate`` masks give V1's pixels for the same dense masks (the blend is
exact at opacity 0.25 steps, else within 1 level at .5 ties; see ``blend``). A mask view
is upscaled once (``full_res``); ``coarse`` uses its nearest-cell masks
instead and differs at cell edges. Dense and RLE masks are the same at both
qualities. All work stays on the device; only
RLE masks are decoded on the host, as their provider does.
"""

from typing import Any, Dict

from pydantic import Field, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.blend import (
    blend,
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
    checked_opacity,
    mask_quality_field,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    MASK_KINDS,
    dense_masks_of,
    placed_in,
    tracker_ids_of,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine.declaration import Ref


class MaskVisualizationBlock(AnnotationBlock):
    """Overlay each instance mask, in its color, on a copy of the image.

    Example: ``opacity=0.5`` (default) mixes each covered pixel half and
    half with its mask's color; smaller masks are drawn over larger ones.
    """

    type = "v2/mask_visualization@v1"
    metadata = {
        "name": "Mask Visualization",
        "port_of": "roboflow_core/mask_visualization@v1",
        "section": "visualization",
    }

    class Params(ColorParams):
        predictions: Ref(*MASK_KINDS) = Field(
            description=(
                "Instance segmentation: dense or RLE InstanceDetections, or a "
                "mask view (SegmentationView). Painting uses one full-image mask "
                "per detection (see mask_quality)."
            ),
            examples=["$steps.instance_segmentation.segmentation"],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.5,
            description="Weight of the mask color, in [0, 1].",
            examples=[0.5, "$inputs.opacity"],
        )
        mask_quality: MaskQuality = mask_quality_field(
            reads="the predictions are instance masks (always for this block)"
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "MaskVisualizationBlock.Params":
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
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Overlay every mask of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: Instance masks (dense, RLE or view).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            opacity: Mask color weight.
            mask_quality: ``accurate`` or ``coarse``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters or predictions of another
                workflow input.
        """
        values = dict(opacity=opacity, color_axis=color_axis)
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
        colored = paint_instances(canvas, dense.mask.to(canvas.device), colors)
        result = annotated(image, blend(canvas, colored, opacity=opacity))

        return result


_CHECKS = {
    "opacity": checked_opacity,
    "color_axis": lambda value: checked_choice(
        value, name="color_axis", choices=COLOR_AXES
    ),
}
