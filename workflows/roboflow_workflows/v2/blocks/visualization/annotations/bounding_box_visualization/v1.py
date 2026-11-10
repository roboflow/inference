"""Box outlines: ``roboflow_core/bounding_box_visualization@v1``.

The default accurate implementation preserves V1 pixels
(``sv.BoxAnnotator`` for ``roundness == 0``, ``sv.RoundBoxAnnotator``
otherwise)::

    host    one transfer of boxes/classes/confidences; outline pixels from cv2
    device  copy of the image; one indexed write of every outline

Boxes are read from any box carrier; masks are never read, so a mask view is
not upscaled.
"""

from typing import Any, Dict, List, Literal, Optional

import cv2
import numpy as np
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    gpu_draw_boxes,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    check_literals,
    check_palette_literals,
    checked_int,
    checked_number,
    reject_copy_image,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    boxes_of,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    bounds_scan,
    rectangle_scan,
    stroke_reach,
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
from roboflow_workflows.v2.engine.implementations import Implementation

BLOCK_TYPE = "roboflow_core/bounding_box_visualization@v1"


class AccurateBoundingBoxes(Implementation):
    """Render exact OpenCV outlines, including rounded corners."""

    name = "accurate"
    quality = ("accurate",)

    def run(
        self,
        *,
        image: ImageData,
        predictions: Any,
        color_palette: str,
        palette_size: int,
        custom_colors: Any,
        color_axis: str,
        thickness: int,
        roundness: float,
    ) -> Dict[str, Any]:
        """Draw every box of ``predictions`` onto a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            thickness: Outline width.
            roundness: Corner roundness in [0, 1].

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, keypoints without boxes or
                predictions of another workflow input.
        """
        values = dict(thickness=thickness, roundness=roundness)
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=BLOCK_TYPE)
        boxes = placed_in(
            boxes_of(predictions, block=BLOCK_TYPE), image, block=BLOCK_TYPE
        )
        rows = read_rows(boxes)
        colors = row_colors(
            rows.class_id,
            rows.tracker_ids(),
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
        )

        self._draw_boxes(
            canvas,
            xyxy=rows.xyxy.astype(int),
            colors=colors,
            thickness=thickness,
            roundness=roundness,
        )
        result = annotated(image, canvas)

        return result

    def _draw_boxes(self, canvas, *, xyxy, colors, thickness, roundness):
        strokes = Strokes(tuple(canvas.shape[-2:]))
        for corners, color in zip(xyxy.tolist(), colors):
            add_box(
                strokes, corners, color=color, thickness=thickness, roundness=roundness
            )
        strokes.paint(canvas)


class FastBoundingBoxes(AccurateBoundingBoxes):
    """Draw tensor bands with approximate outline rasterization.

    All palettes, widths, color axes and prediction-frame mappings are shared
    with the accurate implementation. Rounded boxes and devices other than
    CPU/CUDA use the exact renderer; square boxes use tensor border bands.
    """

    name = "fast"
    quality = ("fast",)

    def _draw_boxes(self, canvas, *, xyxy, colors, thickness, roundness):
        if len(xyxy) == 0:
            return

        if roundness or canvas.device.type not in ("cpu", "cuda"):
            super()._draw_boxes(
                canvas,
                xyxy=xyxy,
                colors=colors,
                thickness=thickness,
                roundness=roundness,
            )
            return

        gpu_draw_boxes(
            canvas,
            xyxy=xyxy,
            colors_rgb=colors,
            thickness=thickness,
            roundness=roundness,
        )


class BoundingBoxVisualizationBlock(Block):
    """Draw each detection's box outline onto a copy of the image.

    The default accurate implementation matches V1 pixels. The fast
    implementation uses square border bands whose widths and joins can
    rasterize differently from OpenCV. Rounded boxes use accurate rendering.
    Both honor the same parameters and leave the input image untouched.

    Example: ``thickness=2`` (default) gives V1's 2 px square boxes;
    ``roundness=0.5`` rounds each corner by a quarter of the box's shorter
    side.
    """

    type = BLOCK_TYPE
    implementations = (AccurateBoundingBoxes, FastBoundingBoxes)
    prunable = True
    metadata = {
        "name": "Bounding Box Visualization",
        "port_of": "roboflow_core/bounding_box_visualization@v1",
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
        thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Outline width in pixels; >= 1.",
            examples=[2, "$inputs.thickness"],
        )
        roundness: Number | Ref(FLOAT_KIND) = Field(
            default=0.0,
            description="Corner roundness in [0, 1]; 0 draws square corners.",
            examples=[0.0, 0.5, "$inputs.roundness"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_palette_literals(self) -> "BoundingBoxVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

        @model_validator(mode="after")
        def _check_literals(self) -> "BoundingBoxVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self


def add_box(
    strokes: Strokes,
    corners: Any,
    *,
    color: Any,
    thickness: int,
    roundness: Optional[float] = 0.0,
) -> None:
    """Add one box outline as V1 draws it.

    Args:
        strokes: Strokes of the frame.
        corners: Integer ``(x1, y1, x2, y2)``.
        color: RGB color.
        thickness: Outline width.
        roundness: ``0`` for ``cv2.rectangle``; otherwise the corner
            roundness of ``sv.RoundBoxAnnotator``.
    """
    x1, y1, x2, y2 = (int(value) for value in corners)
    reach = stroke_reach(thickness)
    if not roundness:
        strokes.add(
            lambda mask: cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness),
            color=color,
            scan=rectangle_scan((x1, y1, x2, y2), reach=reach),
        )
        return

    if abs(x1 - x2) < abs(y1 - y2):
        radius = int((x2 - x1) // 2 * roundness)
    else:
        radius = int((y2 - y1) // 2 * roundness)

    def draw(mask: np.ndarray) -> None:
        # sv.RoundBoxAnnotator: per corner a quarter arc, then the next side.
        centres = (
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x2 - radius, y2 - radius),
            (x1 + radius, y2 - radius),
        )
        sides = (
            ((x1 + radius, y1), (x2 - radius, y1)),
            ((x2, y1 + radius), (x2, y2 - radius)),
            ((x1 + radius, y2), (x2 - radius, y2)),
            ((x1, y1 + radius), (x1, y2 - radius)),
        )
        for centre, side, start in zip(centres, sides, (180, 270, 0, 90)):
            cv2.ellipse(
                mask, centre, (radius, radius), 0, start, start + 90, 1, thickness
            )
            cv2.line(mask, side[0], side[1], 1, thickness)

    strokes.add(
        draw,
        color=color,
        scan=bounds_scan(np.array([[x1, y1], [x2, y2]]), reach=reach),
    )


_CHECKS = {
    "thickness": lambda value: checked_int(value, name="thickness", minimum=1),
    "roundness": lambda value: checked_number(
        value, name="roundness", minimum=0.0, maximum=1.0
    ),
}
