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
A reference path is read more leniently than ``roboflow_core/path_deviation_analytics@v2``
reads it, as V1 drew it (``path_points``).

OpenCV rasterizes on the host into a frame-sized scratch mask, so it clips at
the frame edges as V1. Only the shapes' window (bounds plus the stroke's
reach) goes to the device, as one bool mask; the tint is computed there.
Reference paths and the counter text use the shared ``strokes`` and ``text``
painters.
"""

from numbers import Real
from typing import Any, List, Sequence, Tuple
import cv2
import numpy as np
import torch
from roboflow_workflows.v2.blocks.analytics.zones import parse_line, parse_polygons
from roboflow_workflows.v2.blocks.visualization.annotations.text import (
    FONT,
    Label,
    TextStyle,
)


DEFAULT_ZONE_COLOR = "#5bb573"

COUNTER_TEXT_RGB = (0, 0, 0)

COUNTER_BACKGROUND_RGB = (255, 255, 255)

_INT32 = np.iinfo(np.int32)

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

    More lenient than ``roboflow_core/path_deviation_analytics@v2``, on purpose (V1
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

def frame_scratch(size_hw: Tuple[int, int]) -> np.ndarray:
    """Return a zeroed host mask of the frame for OpenCV to draw shapes into.

    Zero pages stay untouched until OpenCV writes them; frame-sized so the
    shapes clip at the frame edges as in V1.

    Args:
        size_hw: Frame ``(height, width)``.

    Returns:
        ``(H, W)`` uint8 zeros.
    """
    scratch = np.zeros((int(size_hw[0]), int(size_hw[1])), dtype=np.uint8)

    return scratch
