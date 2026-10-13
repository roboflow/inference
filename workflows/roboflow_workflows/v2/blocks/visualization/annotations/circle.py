"""Circles around boxes: ``v2/circle_visualization@v1``.

Port of V1 ``roboflow_core/circle_visualization@v1`` (``sv.CircleAnnotator``).
Pixels equal V1. Per row, from the integer box corners::

    centre = ((x1 + x2) // 2, (y1 + y2) // 2)
    radius = int(distance from centre to (x1, y1))     # half the diagonal
    cv2.circle(centre, radius, thickness)              # -1 fills the circle

    host    one transfer of boxes/classes/confidences; circle pixels from cv2
    device  copy of the image; one indexed write of every circle

Boxes are read from any box carrier; masks are never read, so a mask view is
not upscaled. Circles beyond the image are clipped by OpenCV, as in V1.
"""

from math import sqrt
from typing import Any, Dict

import cv2
import numpy as np
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    BoxColorParams,
    check_literals,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
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
from roboflow_workflows.v2.data.kinds import INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Ref


class CircleVisualizationBlock(AnnotationBlock):
    """Draw a circle through each detection's box corners onto a copy of the image.

    Example: a 40 x 30 box gets a circle of radius 25 around its centre (half
    the box diagonal), 2 px wide by default; ``thickness=-1`` fills it.
    """

    type = "v2/circle_visualization@v1"
    metadata = {
        "name": "Circle Visualization",
        "port_of": "roboflow_core/circle_visualization@v1",
        "section": "visualization",
    }

    class Params(BoxColorParams):
        thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Circle outline width in pixels (>= 1), or -1 to fill it.",
            examples=[2, -1, "$inputs.thickness"],
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "CircleVisualizationBlock.Params":
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
    ) -> Dict[str, Any]:
        """Draw a circle around every box of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            thickness: Outline width, or ``-1`` to fill.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, keypoints without boxes or
                predictions of another workflow input.
        """
        _CHECKS["thickness"](thickness)
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
        for corners, color in zip(rows.xyxy.astype(int).tolist(), colors):
            (centre_x, centre_y), radius = _circle_of(corners)
            square = np.array(
                [
                    [centre_x - radius, centre_y - radius],
                    [centre_x + radius, centre_y + radius],
                ]
            )
            strokes.add(
                lambda mask: cv2.circle(
                    mask, (centre_x, centre_y), radius, 1, thickness
                ),
                color=color,
                scan=bounds_scan(square, reach=reach),
            )
        strokes.paint(canvas)
        result = annotated(image, canvas)

        return result


def _circle_of(corners: Any) -> tuple:
    # sv.CircleAnnotator: integer centre, radius to the top-left corner.
    x1, y1, x2, y2 = corners
    centre = ((x1 + x2) // 2, (y1 + y2) // 2)
    radius = int(sqrt((x1 - centre[0]) ** 2 + (y1 - centre[1]) ** 2))

    return centre, radius


_CHECKS = {
    "thickness": lambda value: checked_outline_or_fill(value, name="thickness"),
}
