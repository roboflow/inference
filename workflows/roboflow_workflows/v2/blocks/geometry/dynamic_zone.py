"""Convex zone with a fixed vertex count per instance mask: ``v2/dynamic_zone@v1``.

Port of V1 ``roboflow_core/dynamic_zone@v1``. Per instance, as V1::

    contours = OpenCV findContours(mask, RETR_TREE, CHAIN_APPROX_SIMPLE),
               outer contours only (no parent), >= 3 points
    hull     = convexHull(contour with the most points), clockwise
    zone     = approxPolyDP(hull, epsilon), epsilon bisected until the zone has
               ``required_number_of_vertices`` vertices (at most 1000 steps)
    [apply_least_squares: each vertex <- intersection of lines fitted to the
     contour points of its two edges]
    pad with the last vertex / truncate to the required count
    zone    <- scaled by ``scale_ratio`` about its centroid
    mask    <- filled zone

Rows stay one to one, so they keep their row keys, boxes and metadata; the
metadata gains ``polygon`` (the zone as ``[x, y]`` integer vertices). A
``SegmentationView`` input gives its reference masks first
(``geometry.mask_inputs``): zone vertices follow mask boundaries, so an
approximate mask would move them. The result is ``InstanceDetections``.

CPU boundary: contours, hulls and polygon fills are OpenCV host algorithms.
Dense masks stay on their device except the window around each instance; the
filled zones come back as windows (``geometry._masks``, shared with
``v2/bounding_rect@v1``). RLE masks are decoded and re-encoded on the host.

Differences from V1:

* Only outer contours compete for "most points". V1 (supervision
  ``mask_to_polygons``, ``RETR_TREE``) also counted hole contours. An
  axis-aligned outline compresses to 4 points and any round hole has more, so
  V1 returned the hull of a hole, e.g. a top-down court with a player inside
  its mask. This is a deliberate correction of an inherited V1 defect; masks
  without holes give V1's zones.
* A prediction without masks raises ``ValueError`` (as
  ``v2/bounding_rect@v1``); V1 returned no zones and ``None`` predictions.
* An instance without a contour of three points (an empty or tiny mask) keeps
  its mask, gets a ``None`` zone and ``polygon``, and clears
  ``simplification_converged``; V1 failed the whole step.
* Least squares fits each edge with an orientation-free line (principal axis
  of its points) and keeps the simplified vertex where two lines do not
  intersect. V1 fitted ``y = a x + b``, which cannot represent vertical
  edges, and failed the step on parallel lines.
* ``required_number_of_vertices`` must be at least 3 and ``scale_ratio``
  positive; V1 accepted any number.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field, StrictBool, StrictFloat, StrictInt
from roboflow_workflows.v2.blocks.geometry._masks import (
    Window,
    mask_windows,
    polygon_masks,
)
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.geometry.mask_inputs import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    instance_predictions_of,
)
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
)
from roboflow_workflows.v2.data.predictions import INSTANCE_SEGMENTATION_PREDICTION_KIND
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

MIN_POLYGON_POINTS = 3
MAX_BISECTION_STEPS = 1000
POLYGON_KEY = "polygon"
# findContours hierarchy rows are (next, previous, first child, parent).
PARENT = 3

Number = StrictFloat | StrictInt


class DynamicZoneBlock(Block):
    """Turn each instance mask into a convex zone with a fixed vertex count.

    Example: a court segmentation with ``required_number_of_vertices=4`` gives
    the court's four corners as a zone for zone counters or drawings.
    """

    type = "v2/dynamic_zone@v1"
    metadata = {
        "name": "Dynamic Zone",
        "port_of": "roboflow_core/dynamic_zone@v1",
        "section": "advanced",
    }
    outputs = {
        "zones": Output(
            LIST_OF_VALUES_KIND,
            source="predictions",
            description=(
                "One zone per row: a list of [x, y] integer vertices, or None "
                "for a row without a contour."
            ),
        ),
        "predictions": Output(
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            source="predictions",
            description="Same rows; zone masks and polygon metadata.",
        ),
        "simplification_converged": Output(
            BOOLEAN_KIND,
            source="predictions",
            description=(
                "True when every row's zone reached the vertex count without "
                "padding or truncation."
            ),
        ),
    }

    class Params(BlockParams):
        predictions: Ref(
            INSTANCE_SEGMENTATION_PREDICTION_KIND, INSTANCE_SEGMENTATION_VIEW_KIND
        ) = Field(description="Instance segmentation prediction with masks.")
        required_number_of_vertices: StrictInt | Ref(INTEGER_KIND) = Field(
            ge=MIN_POLYGON_POINTS,
            description="Vertices of every zone, e.g. 4 for a quadrilateral.",
            examples=[4, "$inputs.vertices"],
        )
        scale_ratio: Number | Ref(FLOAT_KIND) = Field(
            default=1.0,
            gt=0,
            description="Scale of each zone about its centroid; 1 keeps it.",
            examples=[1.0, 1.05],
        )
        apply_least_squares: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False,
            description="Move vertices to intersections of lines fitted to edges.",
        )
        midpoint_fraction: Number | Ref(FLOAT_KIND) = Field(
            default=1.0,
            gt=0,
            le=1,
            description=(
                "Central fraction of each edge's contour points used by the "
                "least squares fit; 1 uses all of them."
            ),
            examples=[1.0, 0.9],
        )

    def run(
        self,
        *,
        predictions: Any,
        required_number_of_vertices: int,
        scale_ratio: float,
        apply_least_squares: bool,
        midpoint_fraction: float,
    ) -> Dict[str, Any]:
        """Replace every instance mask by its simplified convex zone.

        Args:
            predictions: ``InstanceDetections`` or a ``SegmentationView``.
            required_number_of_vertices: Vertices of every zone, at least 3.
            scale_ratio: Positive scale of each zone about its centroid.
            apply_least_squares: Refine vertices from fitted edge lines.
            midpoint_fraction: Central fraction of edge points to fit.

        Returns:
            ``zones``, ``predictions`` and ``simplification_converged``.

        Raises:
            ValueError: When the prediction has no masks.
        """
        predictions = instance_predictions_of(predictions)
        if predictions.mask is None:
            raise ValueError(
                "v2/dynamic_zone@v1 needs instance masks; bind an instance "
                "segmentation prediction"
            )

        zones: List[Optional[np.ndarray]] = []
        all_converged = True
        for window in mask_windows(predictions.mask):
            zone, converged = _zone_of(
                window,
                vertices=required_number_of_vertices,
                apply_least_squares=apply_least_squares,
                midpoint_fraction=midpoint_fraction,
            )
            if zone is not None:
                zone = _scale_polygon(zone, scale=scale_ratio)
            zones.append(zone)
            all_converged = all_converged and converged

        rows = predictions.bboxes_metadata or [None] * len(zones)
        zoned = replace(
            predictions,
            mask=polygon_masks(predictions.mask, polygons=zones),
            bboxes_metadata=[
                {**without_host_mirrors(row), POLYGON_KEY: _as_list(zone)}
                for row, zone in zip(rows, zones)
            ],
        )
        result = {
            "zones": [_as_list(zone) for zone in zones],
            "predictions": zoned,
            "simplification_converged": all_converged,
        }

        return result


def _zone_of(
    window: Optional[Window],
    *,
    vertices: int,
    apply_least_squares: bool,
    midpoint_fraction: float,
) -> Tuple[Optional[np.ndarray], bool]:
    # The zone in frame pixels and whether it had exactly ``vertices`` vertices.
    contour = _largest_contour(window)
    if contour is None:
        return None, False

    zone = _simplified_hull(contour, vertices=vertices)
    if apply_least_squares:
        zone = _least_squares_polygon(
            contour, polygon=zone, midpoint_fraction=midpoint_fraction
        )
    converged = len(zone) == vertices
    if len(zone) < vertices:
        zone = np.concatenate([zone, np.repeat(zone[-1:], vertices - len(zone), 0)])

    return zone[:vertices], converged


def _largest_contour(window: Optional[Window]) -> Optional[np.ndarray]:
    # The outer contour with the most points, as (K, 2) int32 frame pixels.
    # Hole contours never compete: a round hole has more points than a
    # straight outline (module docstring, differences from V1).
    if window is None:
        return None

    (x0, y0), pixels = window
    # One zero pixel of margin: contours of a window see the same edges as
    # contours of the full mask.
    padded = np.pad(pixels, 1)
    contours, hierarchy = cv2.findContours(
        padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = [
        contour
        for contour, links in zip(contours, hierarchy[0] if contours else [])
        if links[PARENT] == -1 and contour.shape[0] >= MIN_POLYGON_POINTS
    ]
    if not contours:
        return None

    largest = max(contours, key=len).reshape(-1, 2) + np.array([x0 - 1, y0 - 1])

    return largest.astype(np.int32)


def _simplified_hull(contour: np.ndarray, *, vertices: int) -> np.ndarray:
    # Bisect the Douglas-Peucker tolerance until the hull has ``vertices``
    # vertices; the last attempt stands when the bisection does not converge.
    hull = cv2.convexHull(contour, clockwise=True, returnPoints=True)
    lower, upper = 1e-7, cv2.arcLength(hull, closed=True)
    epsilon = lower + upper / 2  # V1's first value; the loop then bisects
    polygon = cv2.approxPolyDP(hull, epsilon=epsilon, closed=True)
    for _ in range(MAX_BISECTION_STEPS):
        if len(polygon) == vertices:
            break
        if len(polygon) > vertices:
            lower = epsilon
        else:
            upper = epsilon
        epsilon = lower + (upper - lower) / 2
        polygon = cv2.approxPolyDP(hull, epsilon=epsilon, closed=True)

    return polygon.reshape(-1, 2)


def _least_squares_polygon(
    contour: np.ndarray, *, polygon: np.ndarray, midpoint_fraction: float
) -> np.ndarray:
    # Edge i runs from vertex i - 1 to vertex i, so vertex i is where the
    # lines of edges i and i + 1 meet.
    lines = [
        _fitted_line(_edge_points(contour, start, end, fraction=midpoint_fraction))
        for start, end in zip(np.roll(polygon, 1, axis=0), polygon)
    ]
    refined = []
    for index, vertex in enumerate(polygon):
        crossing = _intersection(lines[index], lines[(index + 1) % len(lines)])
        refined.append(vertex if crossing is None else crossing)

    return np.round(np.array(refined, dtype=np.float64)).astype(np.int32)


def _edge_points(
    contour: np.ndarray, start: np.ndarray, end: np.ndarray, *, fraction: float
) -> np.ndarray:
    # Contour points from the one nearest ``start`` to the one nearest ``end``,
    # wrapping around; then their central ``fraction`` (V1 selection).
    first = int(np.argmin(np.linalg.norm(contour - start, axis=1)))
    last = int(np.argmin(np.linalg.norm(contour - end, axis=1)))
    if first <= last:
        points = contour[first : last + 1]
    else:
        points = np.concatenate([contour[first:], contour[: last + 1]])
    kept = int(round(len(points) * fraction))
    if fraction < 1 and kept > 2:
        cut = (len(points) - kept) // 2
        points = points[cut : len(points) - cut]

    return points


def _fitted_line(points: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    # Total least squares: (point on the line, unit direction); None when
    # the points do not define a direction.
    points = points.astype(np.float64)
    centre = points.mean(axis=0)
    if len(points) < 2 or np.allclose(points, centre):
        return None

    _, _, axes = np.linalg.svd(points - centre)

    return centre, axes[0]


def _intersection(
    first: Optional[Tuple[np.ndarray, np.ndarray]],
    second: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Optional[np.ndarray]:
    if first is None or second is None:
        return None

    (p, d), (q, e) = first, second
    cross = d[0] * e[1] - d[1] * e[0]
    if abs(cross) < 1e-9:
        return None

    offset = q - p
    along_first = (offset[0] * e[1] - offset[1] * e[0]) / cross

    return p + along_first * d


def _scale_polygon(polygon: np.ndarray, *, scale: float) -> np.ndarray:
    # Scale about the area centroid; a zone without area stays as it is.
    if scale == 1:
        return polygon

    moments = cv2.moments(polygon.astype(np.int32))
    if moments["m00"] == 0:
        return polygon

    centroid = np.array([moments["m10"], moments["m01"]]) / moments["m00"]
    scaled = (polygon - centroid) * scale + centroid

    return np.round(scaled).astype(np.int32)


def _as_list(zone: Optional[np.ndarray]) -> Optional[List[List[int]]]:
    return None if zone is None else zone.astype(int).tolist()
