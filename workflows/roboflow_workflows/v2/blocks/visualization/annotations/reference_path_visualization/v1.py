"""Draw a reference path as an open polyline on a copy of the image.

Canonical type: ``roboflow_core/reference_path_visualization@v1``.
"""

from typing import Any, Dict
import cv2
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations._shared.zones import (
    DEFAULT_ZONE_COLOR,
    path_points,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import parse_color
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    check_literals,
    checked_int,
    reject_copy_image,
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


class ReferencePathVisualizationBlock(Block):
    """Draw a reference path as an open polyline on a copy of the image.

    Example: the expected route ``[[10, 90], [50, 40], [90, 40]]`` becomes two
    joined 2-pixel segments in the default color.
    """

    type = "roboflow_core/reference_path_visualization@v1"
    prunable = True
    metadata = {
        "name": "Reference Path Visualization",
        "port_of": "roboflow_core/reference_path_visualization@v1",
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
        reference_path: list | Ref(LIST_OF_VALUES_KIND) = Field(
            description=(
                "Path points [[x, y], ...] in image pixels, joined in order and "
                "not closed. Coordinates after the second of a point are "
                "ignored and an empty list draws nothing, as in V1. More "
                "lenient than path deviation, which needs >= 2 [x, y] points."
            ),
            examples=[[[10, 90], [50, 40], [90, 40]], "$inputs.expected_path"],
        )
        color: str | Ref(STRING_KIND) = Field(
            default=DEFAULT_ZONE_COLOR,
            description="Path color: #RRGGBB, rgb(R, G, B), bgr(B, G, R) or a name.",
            examples=["#5bb573", "WHITE", "$inputs.path_color"],
        )
        thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Path width in pixels; >= 1.",
            examples=[2, "$inputs.thickness"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_literals(self) -> "ReferencePathVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        reference_path: Any,
        color: str,
        thickness: int,
    ) -> Dict[str, Any]:
        """Draw ``reference_path`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            reference_path: Path points.
            color: Path color string.
            thickness: Path width.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On an invalid path, color or thickness.
        """
        values = dict(reference_path=reference_path, color=color, thickness=thickness)
        checked = {name: check(values[name]) for name, check in _CHECKS.items()}
        canvas = canvas_of(image, block=self.type)

        path, thickness = checked["reference_path"], checked["thickness"]
        if len(path):
            strokes = Strokes(image.size_hw)
            strokes.add(
                lambda mask: cv2.polylines(mask, [path], False, 1, thickness),
                color=checked["color"],
                scan=bounds_scan(path, reach=stroke_reach(thickness)),
            )
            strokes.paint(canvas)
        result = annotated(image, canvas)

        return result

_CHECKS = {
    "reference_path": path_points,
    "color": lambda value: parse_color(value, field="color"),
    "thickness": lambda value: checked_int(value, name="thickness", minimum=1),
}
