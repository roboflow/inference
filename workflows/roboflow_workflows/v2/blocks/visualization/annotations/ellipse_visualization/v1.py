"""Ground ellipses under boxes: ``roboflow_core/ellipse_visualization@v1``.

Port of V1 ``roboflow_core/ellipse_visualization@v1``
(``sv.EllipseAnnotator``). Pixels equal V1. Per row, from the integer box
corners::

    centre = (int((x1 + x2) / 2), y2)                  # bottom centre
    axes   = (x2 - x1, int(0.35 * (x2 - x1)))          # semi-axes, as V1
    cv2.ellipse(centre, axes, 0, start_angle, end_angle, thickness, LINE_4)

The horizontal semi-axis is the full box width, so the ellipse is twice as
wide as the box; this is V1's shape. The default arc (-45 to 235 degrees)
leaves a gap at the top, behind the object.

    host    one transfer of boxes/classes/confidences; arc pixels from cv2
    device  copy of the image; one indexed write of every arc

Boxes are read from any box carrier; masks are never read. Arcs beyond the
image are clipped by OpenCV, as in V1. A box with ``x2 < x1`` draws nothing;
V1 fails on it (OpenCV rejects the negative axis).
"""

from typing import Any, Dict, List, Literal
import cv2
import numpy as np
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    PALETTE_NAMES,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    check_literals,
    check_palette_literals,
    reject_copy_image,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    BOX_KINDS,
    boxes_of,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.shapes import (
    checked_outline_or_fill,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    bounds_scan,
    stroke_reach,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


_HEIGHT_RATIO = 0.35

class EllipseVisualizationBlock(Block):
    """Draw an elliptical arc under each detection onto a copy of the image.

    Example: the defaults draw a 2 px arc from -45 to 235 degrees around each
    box's bottom centre, open at the top; ``start_angle=0, end_angle=360``
    closes it.
    """

    type = "roboflow_core/ellipse_visualization@v1"
    prunable = True
    metadata = {
        "name": "Ellipse Visualization",
        "port_of": "roboflow_core/ellipse_visualization@v1",
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
            description=(
                "Arc width in pixels (>= 1), or -1 to fill the sector between "
                "the angles."
            ),
            examples=[2, "$inputs.thickness"],
        )
        start_angle: StrictInt | Ref(INTEGER_KIND) = Field(
            default=-45,
            description=(
                "Arc start in degrees, clockwise from the positive x axis "
                "(0 = right, 90 = down)."
            ),
            examples=[-45, 0, "$inputs.start_angle"],
        )
        end_angle: StrictInt | Ref(INTEGER_KIND) = Field(
            default=235,
            description=(
                "Arc end in degrees, as start_angle; start_angle + 360 draws the "
                "whole ellipse."
            ),
            examples=[235, 360, "$inputs.end_angle"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_palette_literals(self) -> "EllipseVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

        @model_validator(mode="after")
        def _check_literals(self) -> "EllipseVisualizationBlock.Params":
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
        thickness: int,
        start_angle: int,
        end_angle: int,
    ) -> Dict[str, Any]:
        """Draw an arc under every box of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            thickness: Arc width, or ``-1`` to fill.
            start_angle: Arc start in degrees.
            end_angle: Arc end in degrees.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, keypoints without boxes or
                predictions of another workflow input.
        """
        values = dict(thickness=thickness, start_angle=start_angle, end_angle=end_angle)
        for name, check in _CHECKS.items():
            check(values[name])
        canvas = canvas_of(image, block=self.type)
        boxes = placed_in(
            boxes_of(predictions, block=self.type), image, block=self.type
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

        strokes = Strokes(image.size_hw)
        reach = stroke_reach(thickness)
        for (x1, _, x2, y2), color in zip(rows.xyxy.astype(int).tolist(), colors):
            width = x2 - x1
            if width < 0:
                continue
            centre_x = int((x1 + x2) / 2)
            axes = (width, int(_HEIGHT_RATIO * width))
            bounds = np.array(
                [
                    [centre_x - axes[0], y2 - axes[1]],
                    [centre_x + axes[0], y2 + axes[1]],
                ]
            )
            strokes.add(
                lambda mask: cv2.ellipse(
                    mask,
                    (centre_x, y2),
                    axes,
                    0.0,
                    start_angle,
                    end_angle,
                    1,
                    thickness,
                    cv2.LINE_4,
                ),
                color=color,
                scan=bounds_scan(bounds, reach=reach),
            )
        strokes.paint(canvas)
        result = annotated(image, canvas)

        return result

def _checked_angle(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer number of degrees, got {value!r}")

    return value

_CHECKS = {
    "thickness": lambda value: checked_outline_or_fill(value, name="thickness"),
    "start_angle": lambda value: _checked_angle(value, name="start_angle"),
    "end_angle": lambda value: _checked_angle(value, name="end_angle"),
}
