"""Tint one or more polygon zones on a copy of the image.

Canonical type: ``roboflow_core/polygon_zone_visualization@v1``.
"""

from typing import Any, Dict
import cv2
import numpy as np
from pydantic import Field, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations._shared.zones import (
    DEFAULT_ZONE_COLOR,
    add_tint,
    frame_scratch,
    polygon_zones,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import parse_color
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    check_literals,
    checked_opacity,
    reject_copy_image,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import bounds_scan
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class PolygonZoneVisualizationBlock(Block):
    """Tint one or more polygon zones on a copy of the image.

    Example: a parking-bay polygon with the default color and opacity 0.3
    brightens the bay's pixels by ``0.3 * (91, 181, 115)``; others are kept.
    """

    type = "roboflow_core/polygon_zone_visualization@v1"
    prunable = True
    metadata = {
        "name": "Polygon Zone Visualization",
        "port_of": "roboflow_core/polygon_zone_visualization@v1",
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
        zone: list | Ref(LIST_OF_VALUES_KIND) = Field(
            description=(
                "One polygon [[x, y], ...] with >= 3 points, or a list of "
                "polygons, in image pixels. Read by the zone blocks' parser, so "
                "the same value works for both. An empty list draws nothing."
            ),
            examples=[[[10, 10], [90, 10], [50, 70]], "$inputs.zone"],
        )
        color: str | Ref(STRING_KIND) = Field(
            default=DEFAULT_ZONE_COLOR,
            description="Zone color: #RRGGBB, rgb(R, G, B), bgr(B, G, R) or a name.",
            examples=["#5bb573", "WHITE", "$inputs.zone_color"],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.3,
            description=(
                "Share of the color added to the zone's pixels, in [0, 1] "
                "(V1: pixel + opacity * color, saturated at 255)."
            ),
            examples=[0.3, "$inputs.opacity"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_literals(self) -> "PolygonZoneVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        zone: Any,
        color: str,
        opacity: float,
    ) -> Dict[str, Any]:
        """Tint the polygons of ``zone`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            zone: One polygon or a list of polygons.
            color: Zone color string.
            opacity: Share of the color added, in [0, 1].

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On an invalid zone, color or opacity.
        """
        values = dict(zone=zone, color=color, opacity=opacity)
        checked = {name: check(values[name]) for name, check in _CHECKS.items()}
        canvas = canvas_of(image, block=self.type)

        polygons = checked["zone"]
        if polygons:
            scratch = frame_scratch(image.size_hw)
            cv2.fillPoly(scratch, polygons, 1)
            add_tint(
                canvas,
                scratch,
                scan=bounds_scan(np.concatenate(polygons), reach=0),
                color=checked["color"],
                opacity=checked["opacity"],
            )
        result = annotated(image, canvas)

        return result

_CHECKS = {
    "zone": polygon_zones,
    "color": lambda value: parse_color(value, field="color"),
    "opacity": checked_opacity,
}
