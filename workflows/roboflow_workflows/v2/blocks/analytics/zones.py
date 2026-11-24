"""Zone geometry of the zone counters: parsing and box-anchor predicates.

Pure functions; no state, no blocks. Coordinates are pixels of the frame the
boxes belong to, ``x`` right and ``y`` down::

    parse_line([[x1, y1], [x2, y2]])            -> ((x1, y1), (x2, y2))
    parse_polygons([[x, y], ...])               -> (polygon,)       one zone
    parse_polygons([[[x, y], ...], ...])        -> (polygon, ...)   any zone
    parse_polygons([])                          -> ()               no zone
    anchor_points(xyxy (N, 4), anchors)         -> (A, N, 2)
    line_sides(points (A, N, 2), line)          -> (N,) int8  +1 / -1 / 0
    polygon_mask(polygon)                       -> (H, W) bool, built once
    points_in_mask(points (A, N, 2), mask)      -> (N,) bool

Predicates keep the device and dtype of their inputs. Both follow latest V1
(``core_steps/analytics/_zone_geometry.py``):

* A row is on a side of a line only when every anchor is on that side and
  projects onto the segment, ends included. ``+1`` is the left of the
  ``start -> end`` direction as drawn on screen (``cross < 0``): above a line
  drawn left to right. A row straddling the line or beside the segment is
  ``0``.
* A polygon is rasterized once with ``cv2.fillPoly`` (vertices truncated to
  integers, boundary pixels inside, as V1). An anchor is inside when its
  ``ceil``-ed pixel is set. Every anchor must be inside.
"""

import math
from typing import Any, Sequence, Tuple

import cv2
import numpy as np
import torch

__all__ = [
    "ANCHORS",
    "DEFAULT_LINE_ANCHORS",
    "Line",
    "Point",
    "Polygon",
    "anchor_points",
    "line_sides",
    "parse_anchor",
    "parse_line",
    "parse_polygons",
    "points_in_mask",
    "polygon_mask",
]

Point = Tuple[float, float]
Line = Tuple[Point, Point]
Polygon = Tuple[Point, ...]

# Anchor name -> (x column, y column) of the stacked min / centre / max values.
_ANCHOR_COLUMNS = {
    "CENTER": (1, 1),
    "CENTER_LEFT": (0, 1),
    "CENTER_RIGHT": (2, 1),
    "TOP_CENTER": (1, 0),
    "TOP_LEFT": (0, 0),
    "TOP_RIGHT": (2, 0),
    "BOTTOM_LEFT": (0, 2),
    "BOTTOM_CENTER": (1, 2),
    "BOTTOM_RIGHT": (2, 2),
}
ANCHORS: Tuple[str, ...] = tuple(_ANCHOR_COLUMNS)
"""Box anchor names, the ``supervision.Position`` values without CENTER_OF_MASS."""

DEFAULT_LINE_ANCHORS: Tuple[str, ...] = (
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_RIGHT",
)
"""Anchors of a line crossing without ``triggering_anchor``: the whole box."""


def parse_anchor(value: Any, *, name: str = "triggering_anchor") -> str:
    """Check one anchor name.

    Args:
        value: Anchor name, e.g. ``"BOTTOM_CENTER"``.
        name: Parameter name used in the error message.

    Returns:
        The anchor name.

    Raises:
        ValueError: When ``value`` is not one of ``ANCHORS``.
    """
    if value not in _ANCHOR_COLUMNS:
        raise ValueError(
            f"{name} must be one of {', '.join(ANCHORS)}; got {value!r}. "
            "CENTER_OF_MASS needs masks and is not supported for zones."
        )

    return value


def parse_line(value: Any, *, name: str = "line_segment") -> Line:
    """Read a line segment given as two ``[x, y]`` points.

    Args:
        value: ``[[x1, y1], [x2, y2]]``; lists, tuples or a NumPy array.
        name: Parameter name used in error messages.

    Returns:
        ``((x1, y1), (x2, y2))`` as floats.

    Raises:
        ValueError: When ``value`` is not two finite points or both points
            are equal.
    """
    points = _sequence(value, name=name)
    if len(points) != 2:
        raise ValueError(
            f"{name} must be exactly 2 points [[x1, y1], [x2, y2]]; got "
            f"{len(points)} items"
        )

    start = _point(points[0], name=f"{name}[0]")
    end = _point(points[1], name=f"{name}[1]")
    if start == end:
        raise ValueError(f"{name} needs two different points; both are {start}")

    return start, end


def parse_polygons(value: Any, *, name: str = "zone") -> Tuple[Polygon, ...]:
    """Read one polygon or a list of polygons.

    A list whose first item is a point is one polygon; otherwise every item
    is a polygon. An empty list means no polygon.

    Args:
        value: ``[[x, y], ...]`` or ``[[[x, y], ...], ...]``; lists, tuples
            or NumPy arrays.
        name: Parameter name used in error messages.

    Returns:
        Polygons of at least 3 finite float points each.

    Raises:
        ValueError: When an item is not a point or a polygon has fewer than
            3 points.
    """
    items = _sequence(value, name=name)
    if not items:
        return ()

    first = _as_list(items[0])
    first_is_point = (
        isinstance(first, (list, tuple))
        and len(first) == 2
        and not isinstance(_as_list(first[0]), (list, tuple))
    )
    raw_polygons = [items] if first_is_point else items
    polygons = tuple(
        _polygon(raw, name=name if first_is_point else f"{name}[{index}]")
        for index, raw in enumerate(raw_polygons)
    )

    return polygons


def anchor_points(xyxy: torch.Tensor, anchors: Sequence[str]) -> torch.Tensor:
    """Compute anchor points of boxes.

    Centres are ``(min + max) / 2`` in the boxes' dtype, as V1 computes them.

    Args:
        xyxy: ``(N, 4)`` floating boxes ``x_min, y_min, x_max, y_max``.
        anchors: Names from ``ANCHORS``.

    Returns:
        ``(len(anchors), N, 2)`` points on the boxes' device and dtype.
    """
    x_values = torch.stack(
        [xyxy[:, 0], (xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 2]], dim=0
    )
    y_values = torch.stack(
        [xyxy[:, 1], (xyxy[:, 1] + xyxy[:, 3]) / 2, xyxy[:, 3]], dim=0
    )
    columns = [_ANCHOR_COLUMNS[anchor] for anchor in anchors]
    points = torch.stack(
        [torch.stack([x_values[x], y_values[y]], dim=-1) for x, y in columns],
        dim=0,
    )

    return points


def line_sides(points: torch.Tensor, line: Line) -> torch.Tensor:
    """Classify rows against a line segment.

    Args:
        points: ``(A, N, 2)`` anchor points of ``N`` rows.
        line: Segment from ``parse_line``.

    Returns:
        ``(N,)`` int8: ``+1`` every anchor left of ``start -> end`` on screen,
        ``-1`` every anchor right of it or on it, ``0`` straddling the line
        or projecting outside the segment.
    """
    (start_x, start_y), (end_x, end_y) = line
    delta_x, delta_y = end_x - start_x, end_y - start_y
    offset_x = points[..., 0] - start_x
    offset_y = points[..., 1] - start_y

    # Projection onto the segment, ends included.
    along = delta_x * offset_x + delta_y * offset_y
    within_extent = (along >= 0) & (along <= delta_x**2 + delta_y**2)
    on_segment = within_extent.all(dim=0)

    left = (delta_x * offset_y - delta_y * offset_x) < 0
    all_left = left.all(dim=0)
    all_right = (~left).all(dim=0)
    sides = all_left.to(torch.int8) - all_right.to(torch.int8)
    sides = torch.where(on_segment, sides, torch.zeros_like(sides))

    return sides


def polygon_mask(polygon: Polygon) -> torch.Tensor:
    """Rasterize a polygon as V1 does, on the CPU.

    The mask spans ``x`` in ``[0, max x + 2)`` and ``y`` in ``[0, max y + 2)``;
    vertices are truncated to integers and boundary pixels are inside.

    Args:
        polygon: Polygon from ``parse_polygons``.

    Returns:
        ``(H, W)`` bool CPU tensor.
    """
    vertices = np.asarray(polygon, dtype=np.float64)
    width = max(int(vertices[:, 0].max() + 2), 1)
    height = max(int(vertices[:, 1].max() + 2), 1)
    raster = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(raster, [vertices.astype(np.int32)], color=(1,))
    mask = torch.from_numpy(raster).bool()

    return mask


def points_in_mask(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Tell which rows have every anchor inside a polygon mask.

    Args:
        points: ``(A, N, 2)`` anchor points of ``N`` rows.
        mask: ``(H, W)`` bool mask from ``polygon_mask``, on the points'
            device.

    Returns:
        ``(N,)`` bool on the points' device.
    """
    height, width = mask.shape
    pixels = torch.ceil(points).long()
    x, y = pixels[..., 0], pixels[..., 1]
    in_bounds = (x >= 0) & (y >= 0) & (x < width) & (y < height)
    inside = mask[y.clamp(0, height - 1), x.clamp(0, width - 1)] & in_bounds
    rows_inside = inside.all(dim=0)

    return rows_inside


def _polygon(value: Any, *, name: str) -> Polygon:
    points = _sequence(value, name=name)
    if len(points) < 3:
        raise ValueError(
            f"{name} must be a polygon of at least 3 [x, y] points; got "
            f"{len(points)} items"
        )

    polygon = tuple(
        _point(point, name=f"{name}[{index}]") for index, point in enumerate(points)
    )

    return polygon


def _point(value: Any, *, name: str) -> Point:
    coordinates = _as_list(value)
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
        raise ValueError(f"{name} must be a point [x, y]; got {value!r}")

    for coordinate in coordinates:
        is_number = isinstance(coordinate, (int, float, np.integer, np.floating))
        if isinstance(coordinate, (bool, np.bool_)) or not is_number:
            raise ValueError(f"{name} coordinates must be numbers; got {value!r}")
        if not math.isfinite(coordinate):
            raise ValueError(f"{name} coordinates must be finite; got {value!r}")

    point = (float(coordinates[0]), float(coordinates[1]))

    return point


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    items = _as_list(value)
    if not isinstance(items, (list, tuple)):
        raise ValueError(f"{name} must be a list; got {type(value).__name__}")

    return items


def _as_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()

    return value
