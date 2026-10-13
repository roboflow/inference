"""Triangle markers on an anchor of each detection: ``v2/triangle_visualization@v1``.

Port of V1 ``roboflow_core/triangle_visualization@v1``
(``sv.TriangleAnnotator``). Pixels equal V1. Per row, with the tip at its
integer anchor (``shapes.marker_anchors``)::

    vertices = (x - base // 2, y - height), (x + base // 2, y - height), (x, y)
    cv2.fillPoly(vertices, color)
    if outline_thickness: cv2.polylines(vertices, closed, black, outline_thickness)

With the default ``TOP_CENTER`` and a positive ``height`` the triangle sits
above the box, pointing down at it; a negative ``height`` puts it below the
anchor, pointing up.

    host    one transfer of boxes/classes/confidences; triangle pixels from cv2;
            CENTER_OF_MASS only: one transfer of mask sums (anchors.mass_centres)
    device  copy of the image; one indexed write of every triangle

Masks are read only for ``position: CENTER_OF_MASS``. Triangles beyond the
image are clipped by OpenCV, as in V1.
"""

from typing import Any, Dict, Literal

import cv2
import numpy as np
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    POSITIONS_WITH_MASS,
    BoxColorParams,
    MaskQuality,
    check_literals,
    checked_choice,
    checked_int,
    checked_mask_quality,
    mask_quality_field,
    row_colors,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    boxes_of,
    placed_in,
    read_rows,
)
from roboflow_workflows.v2.blocks.visualization.annotations.shapes import (
    marker_anchors,
    position_field,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    bounds_scan,
    stroke_reach,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import INTEGER_KIND, STRING_KIND
from roboflow_workflows.v2.engine.declaration import Ref

# sv.TriangleAnnotator's default outline color; V1 has no parameter for it.
OUTLINE_COLOR = (0, 0, 0)


class TriangleVisualizationBlock(AnnotationBlock):
    """Draw a filled triangle on an anchor of each detection onto a copy of the image.

    Example: the defaults draw a 10 x 10 px triangle above each box, its tip
    on the middle of the box's top edge; ``outline_thickness=1`` adds a black
    edge.
    """

    type = "v2/triangle_visualization@v1"
    metadata = {
        "name": "Triangle Visualization",
        "port_of": "roboflow_core/triangle_visualization@v1",
        "section": "visualization",
    }

    class Params(BoxColorParams):
        position: Literal[POSITIONS_WITH_MASS] | Ref(STRING_KIND) = position_field(
            default="TOP_CENTER", marker="triangle tip"
        )
        base: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description="Width of the triangle's flat side in pixels; >= 0.",
            examples=[10, "$inputs.base"],
        )
        height: StrictInt | Ref(INTEGER_KIND) = Field(
            default=10,
            description=(
                "Distance in pixels from the tip up to the flat side; a negative "
                "value puts the flat side below the tip."
            ),
            examples=[10, -10, "$inputs.height"],
        )
        outline_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description=(
                "Width in pixels of a black edge around each triangle; 0 "
                "(default) draws no edge."
            ),
            examples=[0, 2, "$inputs.outline_thickness"],
        )
        mask_quality: MaskQuality = mask_quality_field(
            reads="position is CENTER_OF_MASS"
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "TriangleVisualizationBlock.Params":
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
        position: str,
        base: int,
        height: int,
        outline_thickness: int,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Draw a triangle on every row of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            position: Anchor of each triangle's tip.
            base: Flat side width.
            height: Tip to flat side distance; negative points up.
            outline_thickness: Black edge width; ``0`` for none.
            mask_quality: Mask quality for ``CENTER_OF_MASS``.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, ``CENTER_OF_MASS`` without
                masks, keypoints without boxes or predictions of another
                workflow input.
        """
        values = dict(
            position=position,
            base=base,
            height=height,
            outline_thickness=outline_thickness,
            mask_quality=mask_quality,
        )
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
        if len(rows) == 0:
            result = annotated(image, canvas)
            return result

        anchors = marker_anchors(
            predictions,
            rows,
            position=position,
            mask_quality=mask_quality,
            image=image,
            block=self.type,
        )
        strokes = Strokes(image.size_hw)
        for (x, y), color in zip(anchors.tolist(), colors):
            vertices = np.array(
                [
                    [x - base // 2, y - height],
                    [x + base // 2, y - height],
                    [x, y],
                ],
                np.int32,
            )
            strokes.add(
                lambda mask: cv2.fillPoly(mask, [vertices], 1),
                color=color,
                scan=bounds_scan(vertices, reach=stroke_reach(1)),
            )
            if outline_thickness:
                strokes.add(
                    lambda mask: cv2.polylines(
                        mask, [vertices], True, 1, thickness=outline_thickness
                    ),
                    color=OUTLINE_COLOR,
                    scan=bounds_scan(vertices, reach=stroke_reach(outline_thickness)),
                )
        strokes.paint(canvas)
        result = annotated(image, canvas)

        return result


def _checked_height(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"height must be an integer, got {value!r}")

    return value


_CHECKS = {
    "position": lambda value: checked_choice(
        value, name="position", choices=POSITIONS_WITH_MASS
    ),
    "base": lambda value: checked_int(value, name="base", minimum=0),
    "height": _checked_height,
    "outline_thickness": lambda value: checked_int(
        value, name="outline_thickness", minimum=0
    ),
    "mask_quality": checked_mask_quality,
}
