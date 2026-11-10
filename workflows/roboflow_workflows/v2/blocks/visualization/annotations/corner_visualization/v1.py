"""Box corner marks: ``roboflow_core/corner_visualization@v1``.

Port of V1 ``roboflow_core/corner_visualization@v1``
(``sv.BoxCornerAnnotator``). Pixels equal V1. Per row, from the integer box
corners, each corner gets two ``cv2.line`` strokes of ``corner_length``
pointing along the box edges::

    (x1, y1) ─────▶      ◀───── (x2, y1)
       │                           │
       ▼                           ▼

       ▲                           ▲
       │                           │
    (x1, y2) ─────▶      ◀───── (x2, y2)

A ``corner_length`` longer than the box makes the marks cross; a negative one
points them outward. Marks beyond the image are clipped by OpenCV, as in V1.

    host    one transfer of boxes/classes/confidences; mark pixels from cv2
            (scanned only around the four corners, not the box interior)
    device  copy of the image; one indexed write of every mark

Boxes are read from any box carrier; masks are never read.
"""

from typing import Any, Dict, List, Literal, Tuple
import cv2
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
    checked_int,
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
    Rect,
    Strokes,
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


Segment = Tuple[Tuple[int, int], Tuple[int, int]]

class CornerVisualizationBlock(Block):
    """Mark the four corners of each detection's box onto a copy of the image.

    Example: the defaults draw 4 px wide, 15 px long marks along both edges
    at every box corner, leaving the middle of each edge open.
    """

    type = "roboflow_core/corner_visualization@v1"
    prunable = True
    metadata = {
        "name": "Corner Visualization",
        "port_of": "roboflow_core/corner_visualization@v1",
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
            default=4,
            description="Mark width in pixels; >= 1.",
            examples=[4, "$inputs.thickness"],
        )
        corner_length: StrictInt | Ref(INTEGER_KIND) = Field(
            default=15,
            description=(
                "Length in pixels of each mark along the box edge; a negative "
                "value points the marks outward."
            ),
            examples=[15, "$inputs.corner_length"],
        )

        @model_validator(mode="before")
        @classmethod
        def _reject_copy_image(cls, values: Any) -> Any:
            checked = reject_copy_image(values, title=cls.__name__)

            return checked

        @model_validator(mode="after")
        def _check_palette_literals(self) -> "CornerVisualizationBlock.Params":
            check_palette_literals(
                palette_size=self.palette_size, custom_colors=self.custom_colors
            )

            return self

        @model_validator(mode="after")
        def _check_literals(self) -> "CornerVisualizationBlock.Params":
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
        corner_length: int,
    ) -> Dict[str, Any]:
        """Mark the corners of every box of ``predictions`` on a copy of ``image``.

        Args:
            image: RGB image; not modified.
            predictions: A box carrier (see ``rows``).
            color_palette: Palette name.
            palette_size: Matplotlib sample count.
            custom_colors: ``CUSTOM`` palette colors.
            color_axis: ``CLASS``, ``INDEX`` or ``TRACK``.
            thickness: Mark width.
            corner_length: Mark length; negative points outward.

        Returns:
            ``image``: the new annotated image.

        Raises:
            ValueError: On invalid parameters, keypoints without boxes or
                predictions of another workflow input.
        """
        values = dict(thickness=thickness, corner_length=corner_length)
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
        for corners, color in zip(rows.xyxy.astype(int).tolist(), colors):
            segments = corner_segments(corners, corner_length=corner_length)

            def draw(mask: Any) -> None:
                for start, end in segments:
                    cv2.line(mask, start, end, 1, thickness=thickness)

            strokes.add(draw, color=color, scan=_segment_scan(segments, reach=reach))
        strokes.paint(canvas)
        result = annotated(image, canvas)

        return result

def corner_segments(corners: Any, *, corner_length: int) -> List[Segment]:
    """Return the eight ``(start, end)`` mark segments of one box, as V1 draws them.

    Args:
        corners: Integer ``(x1, y1, x2, y2)``.
        corner_length: Mark length; negative points outward.

    Returns:
        Per corner, the horizontal then the vertical segment, corners in the
        order top-left, top-right, bottom-left, bottom-right.
    """
    x1, y1, x2, y2 = (int(value) for value in corners)
    segments = []
    for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        # sv.BoxCornerAnnotator compares with x1/y1, so a zero-width box points
        # both marks of a side the same way.
        x_end = x + corner_length if x == x1 else x - corner_length
        y_end = y + corner_length if y == y1 else y - corner_length
        segments.append(((x, y), (x_end, y)))
        segments.append(((x, y), (x, y_end)))

    return segments

def _segment_scan(segments: List[Segment], *, reach: int) -> List[Rect]:
    # One rectangle per segment, grown by the stroke's reach.
    rects = []
    for (start_x, start_y), (end_x, end_y) in segments:
        rects.append(
            (
                min(start_x, end_x) - reach,
                min(start_y, end_y) - reach,
                max(start_x, end_x) + reach + 1,
                max(start_y, end_y) + reach + 1,
            )
        )

    return rects

def _checked_length(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"corner_length must be an integer, got {value!r}")

    return value

_CHECKS = {
    "thickness": lambda value: checked_int(value, name="thickness", minimum=1),
    "corner_length": _checked_length,
}
