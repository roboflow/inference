"""Tint a counting line and write its in/out counts on a copy of the image.

Canonical type: ``roboflow_core/line_counter_visualization@v1``.
"""

from typing import Any, Dict
import cv2
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.visualization.annotations._shared.zones import (
    DEFAULT_ZONE_COLOR,
    add_tint,
    counter_label,
    frame_scratch,
    line_points,
)
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import parse_color
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    Number,
    check_literals,
    checked_int,
    checked_number,
    checked_opacity,
    reject_copy_image,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    bounds_scan,
    stroke_reach,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    TextStyle,
    paint_labels,
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


class LineCounterVisualizationBlock(Block):
    """Tint a counting line and write its in/out counts on a copy of the image.

    Example: line ``[[0, 50], [500, 50]]`` with ``count_in: 3`` and
    ``count_out: 1`` gets a tinted 2-pixel line and "in: 3, out: 1" centred
    on ``(0, 50)``.
    """

    type = "roboflow_core/line_counter_visualization@v1"
    prunable = True
    metadata = {
        "name": "Line Counter Visualization",
        "port_of": "roboflow_core/line_counter_visualization@v1",
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
                "The counting line: two different points [[x1, y1], [x2, y2]] "
                "in image pixels. Read by the line counter's parser, so the "
                "same value works for both. The counts are centred on the "
                "first point."
            ),
            examples=[[[0, 50], [500, 50]], "$inputs.line"],
        )
        color: str | Ref(STRING_KIND) = Field(
            default=DEFAULT_ZONE_COLOR,
            description="Line color: #RRGGBB, rgb(R, G, B), bgr(B, G, R) or a name.",
            examples=["#5bb573", "WHITE", "$inputs.line_color"],
        )
        thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=2,
            description="Line width in pixels; >= 1.",
            examples=[2, "$inputs.thickness"],
        )
        text_thickness: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1,
            description="Stroke width of the count text; >= 1.",
            examples=[1, "$inputs.text_thickness"],
        )
        text_scale: Number | Ref(FLOAT_KIND) = Field(
            default=1.0,
            description="OpenCV font scale of the count text; > 0.",
            examples=[1.0, "$inputs.text_scale"],
        )
        count_in: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description=(
                "Objects counted in; >= 0. Usually the line counter's count_in."
            ),
            examples=["$steps.line_counter.count_in"],
        )
        count_out: StrictInt | Ref(INTEGER_KIND) = Field(
            default=0,
            description=(
                "Objects counted out; >= 0. Usually the line counter's count_out."
            ),
            examples=["$steps.line_counter.count_out"],
        )
        opacity: Number | Ref(FLOAT_KIND) = Field(
            default=0.3,
            description=(
                "Share of the color added to the line's pixels, in [0, 1] "
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
        def _check_literals(self) -> "LineCounterVisualizationBlock.Params":
            check_literals(self.__dict__, _CHECKS)

            return self

    def run(
        self,
        *,
        image: ImageData,
        zone: Any,
        color: str,
        thickness: int,
        text_thickness: int,
        text_scale: float,
        count_in: int,
        count_out: int,
        opacity: float,
    ) -> Dict[str, Any]:
        """Tint the line ``zone`` and write the counts on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            zone: Two points of the counting line.
            color: Line color string.
            thickness: Line width.
            text_thickness: Text stroke width.
            text_scale: Text font scale.
            count_in: Objects counted in.
            count_out: Objects counted out.
            opacity: Share of the color added, in [0, 1].

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On an invalid line or parameter.
        """
        values = dict(
            zone=zone,
            color=color,
            thickness=thickness,
            text_thickness=text_thickness,
            text_scale=text_scale,
            count_in=count_in,
            count_out=count_out,
            opacity=opacity,
        )
        checked = {name: check(values[name]) for name, check in _CHECKS.items()}
        canvas = canvas_of(image, block=self.type)

        line = checked["zone"]
        start, end = [(int(x), int(y)) for x, y in line]
        scratch = frame_scratch(image.size_hw)
        cv2.line(scratch, start, end, 1, checked["thickness"])
        add_tint(
            canvas,
            scratch,
            scan=bounds_scan(line, reach=stroke_reach(checked["thickness"])),
            color=checked["color"],
            opacity=checked["opacity"],
        )
        label = counter_label(
            f"in: {checked['count_in']}, out: {checked['count_out']}",
            anchor=start,
            style=TextStyle(
                scale=checked["text_scale"],
                thickness=checked["text_thickness"],
                padding=0,
                border_radius=0,
            ),
        )
        paint_labels(canvas, [label])
        result = annotated(image, canvas)

        return result

_CHECKS = {
    "zone": line_points,
    "color": lambda value: parse_color(value, field="color"),
    "thickness": lambda value: checked_int(value, name="thickness", minimum=1),
    "text_thickness": lambda value: checked_int(
        value, name="text_thickness", minimum=1
    ),
    "text_scale": lambda value: checked_number(
        value, name="text_scale", minimum=0.0, above_minimum=True
    ),
    "count_in": lambda value: checked_int(value, name="count_in", minimum=0),
    "count_out": lambda value: checked_int(value, name="count_out", minimum=0),
    "opacity": checked_opacity,
}
