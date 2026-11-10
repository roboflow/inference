"""Instance mask overlays: ``roboflow_core/mask_visualization@v1``.

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

from typing import Any, Dict, List, Literal

from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
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
    PALETTE_NAMES,
    palette_indices,
    resolve_palette,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    check_literals,
    check_palette_literals,
    checked_choice,
    checked_opacity,
    reject_copy_image,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    MASK_KINDS,
    MASK_QUALITIES,
    dense_masks_of,
    placed_in,
    tracker_ids_of,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class MaskVisualizationBlock(Block):
    """Overlay each instance mask, in its color, on a copy of the image.

    Example: ``opacity=0.5`` (default) mixes each covered pixel half and
    half with its mask's color; smaller masks are drawn over larger ones.
    """

    type = "roboflow_core/mask_visualization@v1"
    prunable = True
    metadata = {
        "name": "Mask Visualization",
        "port_of": "roboflow_core/mask_visualization@v1",
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
        color_palette: Literal[PALETTE_NAMES] | Ref(STRING_KIND) = Field(
            default="DEFAULT",
            description="Palette of the drawn elements.",
            examples=["DEFAULT", "ROBOFLOW", "Matplotlib Viridis", "$inputs.palette"],
        )
        palette_size: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Number of colors taken from a Matplotlib palette; >= 1.",
            examples=[10, "$inputs.palette_size"],
        )
        custom_colors: List[str] | Ref(LIST_OF_VALUES_KIND) = Field(
            default_factory=list,
            description=(
                "Colors of the CUSTOM palette: #RRGGBB, rgb(R, G, B), "
                "bgr(B, G, R) or a color name such as WHITE."
            ),
            examples=[["#FF0000", "#00FF00", "#0000FF"], "$inputs.custom_colors"],
        )
        color_axis: Literal[COLOR_AXES] | Ref(STRING_KIND) = Field(
            default="CLASS",
            description=(
                "How a row picks its palette color: CLASS (class id), INDEX (row "
                "position) or TRACK (bboxes_metadata tracker_id). Negative ids are "
                "grey."
            ),
            examples=["CLASS", "$inputs.color_axis"],
        )
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
        mask_quality: Literal[MASK_QUALITIES] | Ref(STRING_KIND) = Field(
            default="accurate",
            description=(
                "How instance masks are drawn; used when the predictions are instance "
                "masks (always for this block). Each read builds one full-image mask "
                "per detection, at either quality. accurate: the masks the model "
                "defines (a low-resolution mask view is upscaled with the model's own "
                "interpolation). coarse: a mask view's grid cells are thresholded, "
                "then expanded, so edges can be blocky and small parts can be lost; it "
                "skips the accurate upscaling only. "
                "The model block's fast implementation retains compact scores when "
                "supported, so accurate and coarse can differ. Its accurate "
                "implementation and dense-only models already supply binary masks; "
                "both settings then use those same masks."
            ),
            examples=["accurate", "coarse", "$inputs.mask_quality"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_palette_literals(self) -> "MaskVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

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
