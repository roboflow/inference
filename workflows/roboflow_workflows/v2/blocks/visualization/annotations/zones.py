"""Zone drawings: polygon zones, line counters and reference paths.

Ports of V1 ``roboflow_core/polygon_zone_visualization@v1``,
``roboflow_core/line_counter_visualization@v1`` and
``roboflow_core/reference_path_visualization@v1``. Pixels equal V1::

    polygon zone    one cv2.fillPoly of every polygon (OpenCV's fill rule);
                    covered pixels: pixel + opacity * color, rounded, saturated
    line counter    cv2.line of ``thickness``, tinted the same way; then
                    "in: N, out: M", black on white, centred on the first point
    reference path  open cv2.polylines in the solid color

"pixel + opacity * color" is V1's ``cv2.addWeighted(mask, opacity, image, 1,
0)``: a zone brightens towards the color rather than blending with it.

Polygons and lines are read by the zone blocks' own parsers
(``analytics.zones.parse_polygons`` and ``parse_line``), so one workflow value
such as ``$inputs.zone`` draws exactly when the counters accept it. Points are
image pixels; fractional coordinates are truncated as V1's int32 cast does.
A reference path is read more leniently than ``v2/path_deviation_analytics@v2``
reads it, as V1 drew it (``path_points``).

OpenCV rasterizes on the host into a frame-sized scratch mask, so it clips at
the frame edges as V1. Only the shapes' window (bounds plus the stroke's
reach) goes to the device, as one bool mask; the tint is computed there.
Reference paths and the counter text use the shared ``strokes`` and ``text``
painters.
"""

from numbers import Real
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from pydantic import Field, StrictInt, model_validator
from roboflow_workflows.v2.blocks.analytics.zones import parse_line, parse_polygons
from roboflow_workflows.v2.blocks.visualization.annotations.base import (
    AnnotationBlock,
    annotated,
    canvas_of,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import parse_color
from roboflow_workflows.v2.blocks.visualization.annotations.params import (
    AnnotationParams,
    Number,
    check_literals,
    checked_int,
    checked_number,
    checked_opacity,
)
from roboflow_workflows.v2.blocks.visualization.annotations.strokes import (
    Strokes,
    bounds_scan,
    stroke_reach,
)
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    FONT,
    Label,
    TextStyle,
    paint_labels,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import (
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.engine.declaration import Ref

DEFAULT_ZONE_COLOR = "#5bb573"
COUNTER_TEXT_RGB = (0, 0, 0)
COUNTER_BACKGROUND_RGB = (255, 255, 255)
_INT32 = np.iinfo(np.int32)


class PolygonZoneVisualizationBlock(AnnotationBlock):
    """Tint one or more polygon zones on a copy of the image.

    Example: a parking-bay polygon with the default color and opacity 0.3
    brightens the bay's pixels by ``0.3 * (91, 181, 115)``; others are kept.
    """

    type = "v2/polygon_zone_visualization@v1"
    metadata = {
        "name": "Polygon Zone Visualization",
        "port_of": "roboflow_core/polygon_zone_visualization@v1",
        "section": "visualization",
    }

    class Params(AnnotationParams):
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

        @model_validator(mode="after")
        def _check_literals(self) -> "PolygonZoneVisualizationBlock.Params":
            check_literals(self.__dict__, _POLYGON_ZONE_CHECKS)

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
        checked = {
            name: check(values[name]) for name, check in _POLYGON_ZONE_CHECKS.items()
        }
        canvas = canvas_of(image, block=self.type)

        polygons = checked["zone"]
        if polygons:
            scratch = _scratch(image.size_hw)
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


class LineCounterVisualizationBlock(AnnotationBlock):
    """Tint a counting line and write its in/out counts on a copy of the image.

    Example: line ``[[0, 50], [500, 50]]`` with ``count_in: 3`` and
    ``count_out: 1`` gets a tinted 2-pixel line and "in: 3, out: 1" centred
    on ``(0, 50)``.
    """

    type = "v2/line_counter_visualization@v1"
    metadata = {
        "name": "Line Counter Visualization",
        "port_of": "roboflow_core/line_counter_visualization@v1",
        "section": "visualization",
    }

    class Params(AnnotationParams):
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

        @model_validator(mode="after")
        def _check_literals(self) -> "LineCounterVisualizationBlock.Params":
            check_literals(self.__dict__, _LINE_COUNTER_CHECKS)

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
        checked = {
            name: check(values[name]) for name, check in _LINE_COUNTER_CHECKS.items()
        }
        canvas = canvas_of(image, block=self.type)

        line = checked["zone"]
        start, end = [(int(x), int(y)) for x, y in line]
        scratch = _scratch(image.size_hw)
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


class ReferencePathVisualizationBlock(AnnotationBlock):
    """Draw a reference path as an open polyline on a copy of the image.

    Example: the expected route ``[[10, 90], [50, 40], [90, 40]]`` becomes two
    joined 2-pixel segments in the default color.
    """

    type = "v2/reference_path_visualization@v1"
    metadata = {
        "name": "Reference Path Visualization",
        "port_of": "roboflow_core/reference_path_visualization@v1",
        "section": "visualization",
    }

    class Params(AnnotationParams):
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

        @model_validator(mode="after")
        def _check_literals(self) -> "ReferencePathVisualizationBlock.Params":
            check_literals(self.__dict__, _REFERENCE_PATH_CHECKS)

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
        checked = {
            name: check(values[name]) for name, check in _REFERENCE_PATH_CHECKS.items()
        }
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


def polygon_zones(value: Any, *, name: str = "zone") -> List[np.ndarray]:
    """Read polygons with the zone blocks' ``parse_polygons``, as int32 pixels.

    Args:
        value: ``[[x, y], ...]`` or ``[[[x, y], ...], ...]``; ``[]`` for none.
        name: Parameter name for errors.

    Returns:
        ``(K, 2)`` int32 points per polygon, ``K >= 3``.

    Raises:
        ValueError: When ``parse_polygons`` refuses the value or a coordinate
            is outside the int32 range.
    """
    polygons = [
        _pixels(polygon, name=name) for polygon in parse_polygons(value, name=name)
    ]

    return polygons


def line_points(value: Any, *, name: str = "zone") -> np.ndarray:
    """Read a counting line with the line counter's ``parse_line``, as int32 pixels.

    Args:
        value: ``[[x1, y1], [x2, y2]]`` with two different points.
        name: Parameter name for errors.

    Returns:
        ``(2, 2)`` int32 points.

    Raises:
        ValueError: When ``parse_line`` refuses the value or a coordinate is
            outside the int32 range.
    """
    points = _pixels(parse_line(value, name=name), name=name)

    return points


def path_points(value: Any, *, name: str = "reference_path") -> np.ndarray:
    """Read a reference path as V1 drew it, as int32 pixels.

    More lenient than ``v2/path_deviation_analytics@v2``, on purpose (V1
    parity): any number of points, ``[]`` included, and a point may carry
    more than two numbers; only the first two are read. Path deviation needs
    at least 2 points of exactly ``[x, y]``.

    Args:
        value: ``[[x, y, ...], ...]``.
        name: Parameter name for errors.

    Returns:
        ``(K, 2)`` int32 points; ``K`` may be 0.

    Raises:
        ValueError: On a non-list, a point of fewer than two numbers, a
            non-number, or a coordinate that is not finite int32.
    """
    points = []
    for index, point in enumerate(_sequence(value, name=name)):
        point = _sequence(point, name=f"{name}[{index}]")
        if len(point) < 2 or not all(_is_number(number) for number in point):
            raise ValueError(
                f"{name}[{index}] must be a point [x, y, ...] of numbers, got {point!r}"
            )
        points.append((float(point[0]), float(point[1])))
    pixels = _pixels(points, name=name)

    return pixels


def add_tint(
    canvas: torch.Tensor,
    scratch: np.ndarray,
    *,
    scan: Sequence[Tuple[int, int, int, int]],
    color: Tuple[int, int, int],
    opacity: float,
) -> None:
    """Add ``opacity * color`` to the canvas pixels marked in ``scratch``.

    Equals V1 ``cv2.addWeighted(mask, opacity, image, 1, 0)``: float32 sum,
    rounded half to even, saturated to [0, 255].

    Args:
        canvas: Contiguous ``(3, H, W)`` uint8 image, changed in place.
        scratch: ``(H, W)`` host mask, nonzero where the shape was drawn.
        scan: At most one half-open ``(x0, y0, x1, y1)`` window holding
            every marked pixel; clipped to the frame here.
        color: RGB color.
        opacity: Share of the color added.
    """
    if not scan:
        return

    height, width = scratch.shape
    x0, y0, x1, y1 = scan[0]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
    if x1 <= x0 or y1 <= y0:
        return

    window = torch.from_numpy(scratch[y0:y1, x0:x1] != 0).to(canvas.device)
    region = canvas[:, y0:y1, x0:x1]
    tint = torch.tensor(color, dtype=torch.float32, device=canvas.device)
    tint = (tint * torch.tensor(opacity, dtype=torch.float32)).view(3, 1, 1)
    tinted = (region.float() + tint).round_().clamp_(0, 255).to(torch.uint8)
    region.copy_(torch.where(window, tinted, region))


def counter_label(text: str, *, anchor: Tuple[int, int], style: TextStyle) -> Label:
    """Lay out text as V1 ``sv.draw_text`` with padding 0 and a white background.

    Args:
        text: The text.
        anchor: Centre ``(x, y)`` of the text box.
        style: Scale and thickness; padding and radius are 0.

    Returns:
        The label to paint.
    """
    (text_width, text_height), _ = cv2.getTextSize(
        text, FONT, style.scale, style.thickness
    )
    x, y = anchor[0] - text_width // 2, anchor[1] - text_height // 2
    label = Label(
        background=(x, y, x + text_width, y + text_height),
        background_rgb=COUNTER_BACKGROUND_RGB,
        text_rgb=COUNTER_TEXT_RGB,
        lines=((text, x, anchor[1] + text_height // 2),),
        style=style,
    )

    return label


def _sequence(value: Any, *, name: str) -> list:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of points, got {value!r}")

    items = list(value)

    return items


def _is_number(value: Any) -> bool:
    number = isinstance(value, (Real, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )

    return number


def _pixels(points: Sequence[Tuple[float, float]], *, name: str) -> np.ndarray:
    # Truncate toward zero as V1's int32 cast; refuse what int32 cannot hold.
    coordinates = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    fits = np.isfinite(coordinates) & (coordinates >= _INT32.min)
    fits &= coordinates <= _INT32.max
    if not fits.all():
        raise ValueError(f"{name} must be finite int32 pixels, got {list(points)!r}")

    pixels = coordinates.astype(np.int32)

    return pixels


def _scratch(size_hw: Tuple[int, int]) -> np.ndarray:
    # Zero pages until OpenCV touches them; frame-sized so clipping equals V1.
    scratch = np.zeros((int(size_hw[0]), int(size_hw[1])), dtype=np.uint8)

    return scratch


def _color(value: Any) -> Tuple[int, int, int]:
    rgb = parse_color(value, field="color")

    return rgb


_POLYGON_ZONE_CHECKS = {
    "zone": polygon_zones,
    "color": _color,
    "opacity": checked_opacity,
}
_LINE_COUNTER_CHECKS = {
    "zone": line_points,
    "color": _color,
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
_REFERENCE_PATH_CHECKS = {
    "reference_path": path_points,
    "color": _color,
    "thickness": lambda value: checked_int(value, name="thickness", minimum=1),
}
BLOCKS = (
    PolygonZoneVisualizationBlock,
    LineCounterVisualizationBlock,
    ReferencePathVisualizationBlock,
)
