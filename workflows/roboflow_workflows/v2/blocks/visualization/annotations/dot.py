"""Dots on an anchor of each detection: ``v2/dot_visualization@v1``.

Port of V1 ``roboflow_core/dot_visualization@v1`` (``sv.DotAnnotator``).
Pixels equal V1. Per row, at its integer anchor (``shapes.marker_anchors``)::

    cv2.circle(anchor, radius, color, filled)
    if outline_thickness: cv2.circle(anchor, radius, black, outline_thickness)

    host    one transfer of boxes/classes/confidences; dot pixels from cv2;
            CENTER_OF_MASS only: one transfer of mask sums (anchors.mass_centres)
    device  copy of the image; one indexed write of every dot

Masks are read only for ``position: CENTER_OF_MASS``; a mask view is then
materialized at ``mask_quality``. Dots beyond the image are clipped by
OpenCV, as in V1.
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

# sv.DotAnnotator's default outline color; V1 has no parameter for it.
OUTLINE_COLOR = (0, 0, 0)


class DotVisualizationBlock(AnnotationBlock):
    """Draw a filled dot on an anchor of each detection onto a copy of the image.

    Example: the defaults draw a 4 px radius dot at each box centre;
    ``outline_thickness=2`` adds a black ring around each dot.
    """

    type = "v2/dot_visualization@v1"
    metadata = {
        "name": "Dot Visualization",
        "port_of": "roboflow_core/dot_visualization@v1",
        "section": "visualization",
    }

    class Params(BoxColorParams):
        position: Literal[POSITIONS_WITH_MASS] | Ref(STRING_KIND) = position_field(
            default="CENTER", marker="dot"
        )
        radius: StrictInt | Ref(INTEGER_KIND) = Field(
            default=4,
            description="Dot radius in pixels; >= 0 (0 draws one pixel).",
            examples=[4, "$inputs.radius"],
        )
        outline_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description=(
                "Width in pixels of a black ring around each dot; 0 (default) "
                "draws no ring."
            ),
            examples=[0, 2, "$inputs.outline_thickness"],
        )
        mask_quality: MaskQuality = mask_quality_field(
            reads="position is CENTER_OF_MASS"
        )

        @model_validator(mode="after")
        def _check_literals(self) -> "DotVisualizationBlock.Params":
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
        radius: int,
        outline_thickness: int,
        mask_quality: str,
    ) -> Dict[str, Any]:
        """Draw a dot on every row of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            position: Anchor of each dot.
            radius: Dot radius.
            outline_thickness: Black ring width; ``0`` for none.
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
            radius=radius,
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
            square = np.array([[x - radius, y - radius], [x + radius, y + radius]])
            strokes.add(
                lambda mask: cv2.circle(mask, (x, y), radius, 1, -1),
                color=color,
                scan=bounds_scan(square, reach=stroke_reach(1)),
            )
            if outline_thickness:
                strokes.add(
                    lambda mask: cv2.circle(mask, (x, y), radius, 1, outline_thickness),
                    color=OUTLINE_COLOR,
                    scan=bounds_scan(square, reach=stroke_reach(outline_thickness)),
                )
        strokes.paint(canvas)
        result = annotated(image, canvas)

        return result


_CHECKS = {
    "position": lambda value: checked_choice(
        value, name="position", choices=POSITIONS_WITH_MASS
    ),
    "radius": lambda value: checked_int(value, name="radius", minimum=0),
    "outline_thickness": lambda value: checked_int(
        value, name="outline_thickness", minimum=0
    ),
    "mask_quality": checked_mask_quality,
}
