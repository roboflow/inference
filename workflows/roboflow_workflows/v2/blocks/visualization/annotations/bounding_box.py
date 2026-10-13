"""Box outlines: ``v2/bounding_box_visualization@v1``.

Port of V1 ``roboflow_core/bounding_box_visualization@v1``. Pixels equal V1
(``sv.BoxAnnotator`` for ``roundness == 0``, ``sv.RoundBoxAnnotator``
otherwise)::

    host    one transfer of boxes/classes/confidences; outline pixels from cv2
    device  copy of the image; one indexed write of every outline

Boxes are read from any box carrier; masks are never read, so a mask view is
not upscaled.
"""

from typing import Any, Dict, Optional

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
    Number,
    check_literals,
    checked_int,
    checked_number,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
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
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, INTEGER_KIND
from roboflow_workflows.v2.engine.declaration import Ref


class BoundingBoxVisualizationBlock(AnnotationBlock):
    """Draw each detection's box outline onto a copy of the image.

    Full port of V1 roboflow_core/bounding_box_visualization@v1; not the
    limited prototype painter v2/box_visualization@v1.

    Example: ``thickness=2`` (default) gives V1's 2 px square boxes;
    ``roundness=0.5`` rounds each corner by a quarter of the box's shorter
    side.
    """

    type = "v2/bounding_box_visualization@v1"
    metadata = {
        "name": "Bounding Box Visualization",
        "port_of": "roboflow_core/bounding_box_visualization@v1",
        "section": "visualization",
    }

    class Params(BoxColorParams):
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

        @model_validator(mode="after")
        def _check_literals(self) -> "BoundingBoxVisualizationBlock.Params":
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
        for corners, color in zip(rows.xyxy.astype(int).tolist(), colors):
            add_box(
                strokes, corners, color=color, thickness=thickness, roundness=roundness
            )
        strokes.paint(canvas)
        result = annotated(image, canvas)

        return result


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
