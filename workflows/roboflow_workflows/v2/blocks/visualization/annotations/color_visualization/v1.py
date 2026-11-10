"""Solid box fills: ``roboflow_core/color_visualization@v1``.

Port of V1 ``roboflow_core/color_visualization@v1`` (``sv.ColorAnnotator``)::

    corners = boxes.astype(int)                    truncated, as V1
    colored = image; for each row in order:
              colored[y1..y2, x1..x2] = color      both ends inclusive, clipped
    output  = round(colored * opacity + image * (1 - opacity))

A later row paints over an earlier one. The filled layer equals V1's
``cv2.rectangle(thickness=-1)``; the blend is ``blend.blend`` (exact for
opacity 0, 0.25, 0.5, 0.75 and 1, else within 1 level at .5 ties).

    host    one transfer of boxes/classes/confidences; palette lookup
    device  copy of the image; one slice write per row; one blend

Boxes are read from any box carrier; masks are never read, so a mask view is
not upscaled.
"""

from typing import Any, Dict, List, Literal
import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.blend import blend
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
    colors_of,
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
    BOX_KINDS,
    boxes_of,
    placed_in,
    read_rows,
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


class ColorVisualizationBlock(Block):
    """Fill each detection's box with its color, blended over a copy of the image.

    Example: ``opacity=0.5`` (default) mixes every pixel of a box half and
    half with the box color; ``opacity=1`` paints solid boxes. Where boxes
    overlap, the later row's color is used.
    """

    type = "roboflow_core/color_visualization@v1"
    prunable = True
    metadata = {
        "name": "Color Visualization",
        "port_of": "roboflow_core/color_visualization@v1",
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
        predictions: Ref(*BOX_KINDS) = Field(
            description=(
                "Detections, instance segmentation (dense, RLE or view) or keypoint "
                "predictions with boxes, of this image or of another frame of the "
                "same workflow input (for example a crop or its parent). Those are "
                "mapped into this image; parts outside it are clipped."
            ),
            examples=[
                "$steps.object_detection.predictions",
                "$steps.segmentation.segmentation",
            ],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.5,
            description=(
                "Weight of the box color, in [0, 1]: 0 leaves the image, 1 "
                "paints solid boxes."
            ),
            examples=[0.5, "$inputs.opacity"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_palette_literals(self) -> "ColorVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

        @model_validator(mode="after")
        def _check_literals(self) -> "ColorVisualizationBlock.Params":
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
    ) -> Dict[str, Any]:
        """Fill every box of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            opacity: Box color weight.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, keypoints without boxes or
                predictions of another workflow input.
        """
        values = dict(opacity=opacity, color_axis=color_axis)
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
        )
        palette = resolve_palette(
            color_palette, palette_size=palette_size, custom_colors=custom_colors
        )
        rows = read_rows(boxes)
        if len(rows) == 0:
            result = annotated(image, canvas)
            return result

        indices = palette_indices(
            color_axis, class_id=rows.class_id, tracker_id=rows.tracker_ids()
        )
        colors = torch.from_numpy(colors_of(palette, indices)).to(canvas.device)
        colored = canvas.clone()
        height, width = image.size_hw
        for (x1, y1, x2, y2), color in zip(rows.xyxy.astype(int).tolist(), colors):
            # cv2.rectangle fills both corners inclusively, in either order.
            left, right = max(min(x1, x2), 0), min(max(x1, x2), width - 1)
            top, bottom = max(min(y1, y2), 0), min(max(y1, y2), height - 1)
            if left <= right and top <= bottom:
                colored[:, top : bottom + 1, left : right + 1] = color.view(3, 1, 1)
        result = annotated(image, blend(canvas, colored, opacity=opacity))

        return result

_CHECKS = {
    "opacity": checked_opacity,
    "color_axis": lambda value: checked_choice(
        value, name="color_axis", choices=COLOR_AXES
    ),
}
