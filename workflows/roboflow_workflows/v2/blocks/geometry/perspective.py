"""Map an image plane to a top-down rectangle: ``v2/perspective_correction@v1``.

Port of V1 ``roboflow_core/perspective_correction@v1`` (tensor variant)::

    perspective_polygons ─▶ largest 4-vertex polygon, ordered from bottom-left
          │                 (optionally grown to hold every detection anchor)
          ▼
    3x3 homography  polygon ─▶ (0, 0) .. (W' - 1, H' - 1)
          │
          ├─▶ warped_image            image warped on its device (optional)
          └─▶ corrected_coordinates   boxes, masks, keypoints in the rectangle

``W'`` and ``H'`` are ``transformed_rect_width`` / ``height`` plus any growth;
they are also returned as ``extended_transformed_rect_width`` / ``height``.

A homography cannot be written as an image provenance mapping (scale and
offset), so the rectangle is a **new root frame** with a fresh id. The warped
image and the corrected predictions both describe that frame, and keep the
input's source and time (``video_metadata``); the warped image also keeps
``parent_row``. Without ``warp_image`` the input image is returned unchanged
and the predictions still describe the rectangle.

Host work is small and explicit: the polygon logic and
``cv2.getPerspectiveTransform`` (a 3x3 matrix); RLE masks are decoded and
encoded on the host. Pixels, boxes, dense masks and keypoint tensors stay on
their device.

Differences from V1:

* No matrix cache across calls. V1 reused the first call's matrix whenever
  no extension anchor was set, so a changed polygon was ignored.
* Masks are warped pixel by pixel (nearest source pixel). V1 joined every
  contour of a mask into one polygon, which distorted masks with several
  parts. A masked row's box is the box of its warped mask; rows whose warped
  mask is empty use their warped box corners, as V1 did for empty masks.
* Image pixels use float bilinear sampling, not OpenCV's fixed-point
  weights: a pixel may differ by 1 level.
* A degenerate polygon (three collinear vertices) raises instead of producing
  an invalid warp.
* ``transformed_rect_width`` and ``transformed_rect_height`` must be at least
  2, whether written or selected. V1 had no lower bound and silently built a
  singular matrix for a side of 1.
* Keypoint covariance (pixel units) moves with its point as
  ``J @ covariance @ J.T``, where ``J`` is the homography's Jacobian at the
  input point. V1's supervision keypoints carried no covariance.
"""

import math
import uuid
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from pydantic import AliasChoices, Field, StrictBool, StrictInt
from roboflow_workflows.execution_engine.constants import (
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
)
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.blocks.image._resample import (
    compute_dtype,
    new_root_image,
    remap_bilinear,
    require_same_image,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rle import decode_rle_masks
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)
ALL_CORNERS = "ALL"
CENTER_OF_MASS = "CENTER_OF_MASS"
ANCHORS = (
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
    CENTER_OF_MASS,
)
ExtensionAnchor = Literal[
    "",
    "CENTER",
    "CENTER_LEFT",
    "CENTER_RIGHT",
    "TOP_CENTER",
    "TOP_LEFT",
    "TOP_RIGHT",
    "BOTTOM_LEFT",
    "BOTTOM_CENTER",
    "BOTTOM_RIGHT",
    "CENTER_OF_MASS",
    "ALL",
]
# Row metadata holding oriented box corners, (4, 2) per row (supervision's key).
ORIENTED_BOX_KEY = "xyxyxyxy"
# Determinant below this marks a polygon whose homography cannot be inverted.
DEGENERATE_DETERMINANT = 1e-12
# A one-pixel side maps two polygon corners to one point: no homography.
MIN_RECT_SIDE = 2
# Most mask pixels a uint8 sum can count without overflow.
COUNT_BLOCK = 255

Vertex = np.ndarray


@dataclass(frozen=True)
class TopDownPlane:
    """The top-down rectangle a perspective polygon maps to.

    Args:
        matrix: ``(3, 3)`` float64 homography from image to rectangle pixels.
        width: Rectangle width in pixels, including growth.
        height: Rectangle height in pixels, including growth.
    """

    matrix: np.ndarray
    width: int
    height: int


class PerspectiveCorrectionBlock(Block):
    """Map an image region to a top-down rectangle; see the module docstring.

    Example: the polygon ``[[100, 399], [100, 100], [499, 100], [499, 399]]``
    with a 400 x 300 rectangle is a pure shift by ``(-100, -100)``: a box
    ``[150, 150, 200, 250]`` becomes ``[50, 50, 100, 150]``.
    """

    type = "v2/perspective_correction@v1"
    outputs = {
        "corrected_coordinates": Output(
            *PREDICTION_KINDS,
            source="predictions",
            description=(
                "The prediction in top-down rectangle pixels; rows, classes and "
                "row metadata kept. None without predictions."
            ),
        ),
        "warped_image": Output(
            IMAGE_KIND,
            source="images",
            description=(
                "The top-down view when warp_image is true (a new root frame); "
                "otherwise the input image."
            ),
        ),
        "extended_transformed_rect_width": Output(
            INTEGER_KIND, description="Rectangle width including any growth."
        ),
        "extended_transformed_rect_height": Output(
            INTEGER_KIND, description="Rectangle height including any growth."
        ),
    }
    # A missing prediction still warps the image, as in V1.
    accepts_empty = True
    prunable = True

    class Params(BlockParams):
        images: Ref(IMAGE_KIND) = Field(
            validation_alias=AliasChoices("images", "image"),
            description="Image the polygon and predictions are drawn on.",
        )
        predictions: Optional[Ref(*PREDICTION_KINDS)] = Field(
            default=None,
            description=(
                "Optional object detection, instance segmentation or keypoint "
                "prediction of that image."
            ),
        )
        perspective_polygons: list | Ref(LIST_OF_VALUES_KIND) = Field(
            description=(
                "One polygon of four [x, y] vertices, or a list of polygons; the "
                "largest four-vertex polygon is used."
            ),
            examples=[[[100, 100], [500, 100], [500, 400], [100, 400]]],
        )
        transformed_rect_width: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1000,
            ge=MIN_RECT_SIDE,
            description="Width of the top-down rectangle; at least 2.",
        )
        transformed_rect_height: StrictInt | Ref(INTEGER_KIND) = Field(
            default=1000,
            ge=MIN_RECT_SIDE,
            description="Height of the top-down rectangle; at least 2.",
        )
        extend_perspective_polygon_by_detections_anchor: ExtensionAnchor | Ref(
            STRING_KIND
        ) = Field(
            default="",
            description=(
                "Grow the polygon until it holds this anchor of every detection "
                "(a supervision Position name), 'ALL' for whole boxes, or '' to "
                "keep it."
            ),
        )
        warp_image: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Also warp the image to the top-down view."
        )

    def run(
        self,
        *,
        images: Optional[ImageData],
        predictions: Any,
        perspective_polygons: Any,
        transformed_rect_width: int,
        transformed_rect_height: int,
        extend_perspective_polygon_by_detections_anchor: str,
        warp_image: bool,
    ) -> Dict[str, Any]:
        """Correct ``predictions`` and optionally warp ``images``.

        Args:
            images: The image, or ``None`` when it is missing upstream.
            predictions: ``Detections``, ``InstanceDetections``, a
                ``(KeyPoints, Optional[Detections])`` tuple or ``None``.
            perspective_polygons: One polygon or a list of polygons.
            transformed_rect_width: Rectangle width before growth.
            transformed_rect_height: Rectangle height before growth.
            extend_perspective_polygon_by_detections_anchor: Anchor name,
                ``"ALL"`` or ``""``.
            warp_image: Whether to warp the image.

        Returns:
            The four declared outputs; all ``None`` without an image.

        Raises:
            ValueError: On a malformed or degenerate polygon, a rectangle
                side below 2, an unknown anchor, or a ``None`` rectangle
                side or ``warp_image``.
            ContractError: When the predictions describe another image,
                ``CENTER_OF_MASS`` is requested without masks, or keypoint
                covariance is not shaped like the points.
        """
        anchor = extend_perspective_polygon_by_detections_anchor
        if anchor not in ("", ALL_CORNERS, *ANCHORS):
            raise ValueError(
                "extend_perspective_polygon_by_detections_anchor must be '', "
                f"'{ALL_CORNERS}' or one of {list(ANCHORS)}, got {anchor!r}"
            )
        if images is None:
            return dict.fromkeys(self.outputs)
        # The block accepts empty inputs for missing predictions; a missing
        # setting must not become a TypeError or a silently skipped warp.
        settings = {
            "transformed_rect_width": transformed_rect_width,
            "transformed_rect_height": transformed_rect_height,
            "warp_image": warp_image,
        }
        for name, value in settings.items():
            if value is None:
                raise ValueError(f"{name} is required, got None from its selector")

        detections = _boxes_of(predictions)
        if predictions is not None:
            require_same_image(predictions, image=images, consumer=self.type)
        polygon = largest_polygon(perspective_polygons)
        plane = top_down_plane(
            polygon,
            width=transformed_rect_width,
            height=transformed_rect_height,
            detections=detections if anchor else None,
            anchor=anchor,
        )

        plane_id = f"perspective-{uuid.uuid4().hex}"
        frame = FrameRef(
            image_id=plane_id,
            size_hw=(plane.height, plane.width),
            parent=FrameMapping.identity(plane_id, (plane.height, plane.width)),
            root=FrameMapping.identity(plane_id, (plane.height, plane.width)),
            video_metadata=images.video_metadata,
        )
        warped = images
        if warp_image:
            pixels = warp_pixels(images.tensor_image, plane=plane)
            warped = new_root_image(pixels, like=images, image_id=plane_id)

        result = {
            "corrected_coordinates": correct_prediction(
                predictions, plane=plane, frame=frame
            ),
            "warped_image": warped,
            "extended_transformed_rect_width": plane.width,
            "extended_transformed_rect_height": plane.height,
        }

        return result


def largest_polygon(perspective_polygons: Any) -> np.ndarray:
    """Pick the polygon to correct, with V1's rules.

    A list of four ``[x, y]`` pairs, or a ``(4, 2)`` array, is one polygon and
    is used as given. Otherwise the value is a list of polygons: those with
    four vertices are rounded to integers and the largest by area wins.

    Args:
        perspective_polygons: One polygon or a list of polygons.

    Returns:
        ``(4, 2)`` vertex array.

    Raises:
        ValueError: When no four-vertex polygon is given.
    """
    if isinstance(perspective_polygons, np.ndarray):
        if perspective_polygons.shape != (4, 2):
            raise ValueError(
                "a perspective polygon array must have shape (4, 2), got "
                f"{perspective_polygons.shape}"
            )
        return perspective_polygons
    if not isinstance(perspective_polygons, (list, tuple)) or not perspective_polygons:
        raise ValueError(
            "perspective_polygons must be one polygon of four [x, y] vertices or a "
            f"non-empty list of polygons, got {perspective_polygons!r}"
        )
    if _is_one_polygon(perspective_polygons):
        polygon = np.array(perspective_polygons)

        return polygon

    candidates = [np.asarray(candidate) for candidate in perspective_polygons]
    candidates = [
        np.around(candidate).astype(np.int32)
        for candidate in candidates
        if candidate.shape == (4, 2)
    ]
    if not candidates:
        raise ValueError(
            "perspective_polygons holds no polygon of exactly four [x, y] vertices"
        )
    largest = max(candidates, key=cv2.contourArea)

    return largest


def top_down_plane(
    polygon: np.ndarray,
    *,
    width: int,
    height: int,
    detections: Any = None,
    anchor: str = "",
) -> TopDownPlane:
    """Build the homography from a polygon to a ``width`` x ``height`` rectangle.

    The vertices are ordered clockwise from the one nearest the bottom-left
    corner of their bounding box, which maps to ``(0, H' - 1)``. With an
    anchor and detections the polygon first grows along its own sides until
    every anchor point lies inside it, and the rectangle grows in proportion.

    Args:
        polygon: ``(4, 2)`` vertices in image pixels.
        width: Rectangle width before growth.
        height: Rectangle height before growth.
        detections: ``Detections`` / ``InstanceDetections`` whose anchors
            the polygon must hold, or ``None``.
        anchor: Anchor name, ``"ALL"`` for both corners on each side, or
            ``""`` for no growth.

    Returns:
        The homography and the grown rectangle size.

    Raises:
        ValueError: When ``width`` or ``height`` is below 2 (a one-pixel side
            collapses the rectangle), or the vertices are degenerate.
        ContractError: On ``CENTER_OF_MASS`` without masks.
    """
    if width < MIN_RECT_SIDE or height < MIN_RECT_SIDE:
        raise ValueError(
            f"the top-down rectangle must be at least {MIN_RECT_SIDE} x "
            f"{MIN_RECT_SIDE} pixels, got transformed_rect_width={width} and "
            f"transformed_rect_height={height}"
        )
    ordered = _start_from_bottom_left(_sort_clockwise(polygon))
    extended_width, extended_height = 0.0, 0.0
    if anchor and detections is not None and len(detections.xyxy) > 0:
        points = _anchor_points(detections, anchor=anchor)
        ordered, original_wh, extension_wh = _extend_polygon(ordered, points=points)
        extended_width = extension_wh[0] * width / max(original_wh[0], 1)
        extended_height = extension_wh[1] * height / max(original_wh[1], 1)

    plane_width = width + int(round(extended_width))
    plane_height = height + int(round(extended_height))
    destination = np.array(
        [
            [0, plane_height - 1],
            [0, 0],
            [plane_width - 1, 0],
            [plane_width - 1, plane_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < (
        DEGENERATE_DETERMINANT
    ):
        raise ValueError(
            f"perspective polygon {polygon.tolist()} is degenerate (three vertices "
            "on one line); give four corners of a quadrilateral"
        )
    plane = TopDownPlane(matrix=matrix, width=plane_width, height=plane_height)

    return plane


def warp_pixels(pixels: torch.Tensor, *, plane: TopDownPlane) -> torch.Tensor:
    """Warp an image into the rectangle, on its device.

    Args:
        pixels: ``(C, H, W)`` uint8 image.
        plane: Target rectangle and homography.

    Returns:
        ``(C, H', W')`` uint8 image; pixels mapping outside the input are 0.
    """
    source_x, source_y = _source_grid(plane, device=pixels.device)
    warped = remap_bilinear(pixels, source_x=source_x, source_y=source_y)

    return warped


def correct_prediction(prediction: Any, *, plane: TopDownPlane, frame: FrameRef) -> Any:
    """Move a native prediction into the top-down rectangle.

    Args:
        prediction: ``Detections``, ``InstanceDetections``, a
            ``(KeyPoints, Optional[Detections])`` tuple or ``None``. Not
            modified.
        plane: Target rectangle and homography.
        frame: Frame the result describes (the rectangle).

    Returns:
        A prediction of the same type, or ``None``. Keypoint covariance is
        moved with the homography's Jacobian at each input point.

    Raises:
        ContractError: When keypoint covariance is not ``(*xy.shape[:-1], 2, 2)``.
    """
    if prediction is None:
        return None
    if isinstance(prediction, tuple):
        key_points, detections = prediction
        corrected = (
            _correct_key_points(key_points, plane=plane, frame=frame),
            _correct_detections(detections, plane=plane, frame=frame),
        )
        return corrected

    corrected = _correct_detections(prediction, plane=plane, frame=frame)

    return corrected


def _is_one_polygon(value: Sequence[Any]) -> bool:
    return len(value) == 4 and all(
        isinstance(vertex, (list, tuple)) and len(vertex) == 2 for vertex in value
    )


def _boxes_of(prediction: Any) -> Any:
    if isinstance(prediction, tuple):
        return prediction[1]
    return prediction


def _sort_clockwise(polygon: np.ndarray) -> np.ndarray:
    # V1 order: by angle around the bounding-box centre, measured from +y.
    x_center = min(polygon[:, 0]) / 2 + max(polygon[:, 0]) / 2
    y_center = min(polygon[:, 1]) / 2 + max(polygon[:, 1]) / 2
    ordered = np.array(
        sorted(
            polygon.tolist(),
            key=lambda vertex: math.atan2(x_center - vertex[0], y_center - vertex[1]),
            reverse=True,
        )
    )

    return ordered


def _start_from_bottom_left(polygon: np.ndarray) -> np.ndarray:
    # The rotation whose vertices are closest to the bounding box corners
    # (bottom-left, top-left, top-right, bottom-right); ties keep the lower x.
    x_min, x_max = min(polygon[:, 0]), max(polygon[:, 0])
    y_min, y_max = min(polygon[:, 1]), max(polygon[:, 1])
    corners = [[x_min, y_max], [x_min, y_min], [x_max, y_min], [x_max, y_max]]

    def distance(candidate: np.ndarray) -> float:
        return sum(
            ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            for (x1, y1), (x2, y2) in zip(corners, candidate)
        )

    closest, closest_distance = polygon, distance(polygon)
    for shift in range(4):
        rolled = np.roll(polygon, shift, axis=0)
        rolled_distance = distance(rolled)
        if rolled_distance < closest_distance or (
            rolled_distance == closest_distance and rolled[0][0] < closest[0][0]
        ):
            closest, closest_distance = rolled, rolled_distance

    return closest


def _anchor_points(detections: Any, *, anchor: str) -> List[List[Tuple[float, float]]]:
    """Per row, the points each of the four sides must hold, V1 order."""
    xyxy = detections.xyxy.detach().cpu().numpy().astype(np.float64)
    if anchor == CENTER_OF_MASS:
        centroids = _mask_centroids(detections)
    rows = []
    for index, (x1, y1, x2, y2) in enumerate(xyxy.tolist()):
        if anchor == ALL_CORNERS:
            # left, right, bottom, top sides: their two box corners.
            rows.append(
                [
                    ((x1, y2), (x1, y1)),
                    ((x2, y2), (x2, y1)),
                    ((x2, y2), (x1, y2)),
                    ((x2, y1), (x1, y1)),
                ]
            )
            continue
        if anchor == CENTER_OF_MASS:
            point = tuple(float(value) for value in centroids[index])
        else:
            point = _box_anchor((x1, y1, x2, y2), anchor=anchor)
        rows.append([(point,)] * 4)

    return rows


def _box_anchor(
    box: Tuple[float, float, float, float], *, anchor: str
) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    xs = {"LEFT": x1, "CENTER": (x1 + x2) / 2, "RIGHT": x2}
    ys = {"TOP": y1, "CENTER": (y1 + y2) / 2, "BOTTOM": y2}
    vertical, _, horizontal = anchor.partition("_")
    if anchor == "CENTER":
        vertical, horizontal = "CENTER", "CENTER"
    point = (xs[horizontal], ys[vertical])

    return point


def _mask_centroids(detections: Any) -> np.ndarray:
    # supervision's CENTER_OF_MASS: mean pixel centre (index + 0.5), truncated.
    if not isinstance(detections, InstanceDetections):
        raise ContractError(
            "v2/perspective_correction@v1: anchor CENTER_OF_MASS needs instance "
            "segmentation masks; these predictions have only boxes"
        )
    # Only the (N, W) and (N, H) pixel counts become floats, never the masks.
    dense = _dense_masks(detections)
    dtype = compute_dtype(dense.device)
    height, width = dense.shape[-2:]
    column_counts = _pixel_counts(dense, dim=1)
    row_counts = _pixel_counts(dense, dim=2)
    area = column_counts.sum(dim=1).clamp(min=1).to(dtype)
    columns = torch.arange(width, dtype=dtype, device=dense.device) + 0.5
    rows = torch.arange(height, dtype=dtype, device=dense.device) + 0.5
    x = (column_counts.to(dtype) * columns).sum(dim=1) / area
    y = (row_counts.to(dtype) * rows).sum(dim=1) / area
    centroids = torch.stack((x, y), dim=1).cpu().numpy().astype(int)

    return centroids


def _pixel_counts(masks: torch.Tensor, *, dim: int) -> torch.Tensor:
    # A torch sum casts its whole input to the result type first, so summing
    # (N, H, W) bool masks as int64 allocates 8 bytes per pixel. Summing a block
    # of at most 255 pixels as uint8 casts only that block, at 1 byte per pixel.
    # The cast turns every true value into 1, even when its stored byte is not
    # 1, so the uint8 sum cannot overflow. The partial sums are added as int64.
    length = masks.shape[dim]
    shape = masks.shape[:dim] + masks.shape[dim + 1 :]
    counts = torch.zeros(shape, dtype=torch.int64, device=masks.device)
    for start in range(0, length, COUNT_BLOCK):
        block = masks.narrow(dim, start, min(COUNT_BLOCK, length - start))
        counts += block.sum(dim=dim, dtype=torch.uint8)

    return counts


def _extend_polygon(
    polygon: np.ndarray, *, points: List[List[Tuple[Tuple[float, float], ...]]]
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """Grow the polygon along its sides to hold every point (V1 algorithm).

    Returns the grown polygon, the original (width, height) as the longer of
    each pair of opposite sides, and the accumulated growth (width, height).
    """
    vertices = dict(
        zip(("bottom_left", "top_left", "top_right", "bottom_right"), polygon)
    )
    original_width = max(
        _distance(vertices["bottom_left"], vertices["bottom_right"]),
        _distance(vertices["top_left"], vertices["top_right"]),
    )
    original_height = max(
        _distance(vertices["bottom_left"], vertices["top_left"]),
        _distance(vertices["bottom_right"], vertices["top_right"]),
    )
    # Each side: the edge that moves outward (start, end), the vertices its
    # ends slide away from, the vertex whose travel counts as growth, and the
    # growing dimension. Order and choices follow V1 exactly.
    sides = (
        (("bottom_left", "top_left"), ("bottom_right", "top_right"), "bottom_left", 0),
        (("top_right", "bottom_right"), ("top_left", "bottom_left"), "bottom_right", 0),
        (("bottom_right", "bottom_left"), ("top_right", "top_left"), "bottom_left", 1),
        (("top_left", "top_right"), ("bottom_left", "bottom_right"), "top_left", 1),
    )
    growth = [0.0, 0.0]
    for row in points:
        for side_points, ((start, end), (start_from, end_from), measured, axis) in zip(
            row, sides
        ):
            for x, y in side_points:
                contour = np.array(
                    [vertices[name] for name in vertices], dtype=np.float32
                )
                if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                    continue
                if _is_counter_clockwise(vertices[start], vertices[end], (x, y)):
                    continue
                before = vertices[measured]
                vertices[start], vertices[end] = _slide_edge_to_point(
                    vertices[start],
                    vertices[end],
                    start_from=vertices[start_from],
                    end_from=vertices[end_from],
                    point=(x, y),
                )
                growth[axis] += _distance(vertices[measured], before)
    grown = np.array(list(vertices.values()))

    return grown, (original_width, original_height), (growth[0], growth[1])


def _distance(first: Vertex, second: Vertex) -> float:
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _is_counter_clockwise(
    start: Vertex, end: Vertex, point: Tuple[float, float]
) -> bool:
    turn = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )

    return turn > 0


def _line(first: Vertex, second: Vertex) -> Tuple[Optional[float], float]:
    # (slope, intercept); a vertical line is (None, x).
    if first[0] == second[0]:
        return None, first[0]
    slope = (second[1] - first[1]) / (second[0] - first[0])
    intercept = (first[1] * second[0] - second[1] * first[0]) / (second[0] - first[0])

    return slope, intercept


def _intersection(
    first: Tuple[Optional[float], float], second: Tuple[Optional[float], float]
) -> np.ndarray:
    (slope_1, intercept_1), (slope_2, intercept_2) = first, second
    if slope_1 is None and slope_2 is None:
        raise ValueError("cannot grow the perspective polygon: both lines are vertical")
    if slope_1 is None:
        x = intercept_1
        y = slope_2 * x + intercept_2
    elif slope_2 is None:
        x = intercept_2
        y = slope_1 * x + intercept_1
    else:
        x = (intercept_2 - intercept_1) / (slope_1 - slope_2)
        y = slope_1 * x + intercept_1
    vertex = np.array((x, y)).round().astype(int)

    return vertex


def _slide_edge_to_point(
    start: Vertex,
    end: Vertex,
    *,
    start_from: Vertex,
    end_from: Vertex,
    point: Tuple[float, float],
) -> Tuple[Vertex, Vertex]:
    # The edge moves parallel to itself through ``point``; its ends follow
    # the lines towards ``start_from`` / ``end_from``.
    slope, _ = _line(start, end)
    shifted = (slope, point[0] if slope is None else point[1] - slope * point[0])
    new_start = _intersection(_line(start, start_from), shifted)
    new_end = _intersection(_line(end, end_from), shifted)

    return new_start, new_end


def _map_points(points: torch.Tensor, *, matrix: np.ndarray) -> torch.Tensor:
    """Apply the homography to ``(..., 2)`` points, in float64 where possible."""
    dtype = compute_dtype(points.device)
    homography = torch.as_tensor(matrix, dtype=dtype, device=points.device)
    xy = points.to(dtype)
    projected = xy @ homography[:2, :2].T + homography[:2, 2]
    scale = xy @ homography[2, :2] + homography[2, 2]
    mapped = projected / scale.unsqueeze(-1)

    return mapped


def _source_grid(
    plane: TopDownPlane, *, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Input-image position of every rectangle pixel centre."""
    dtype = compute_dtype(device)
    ys, xs = torch.meshgrid(
        torch.arange(plane.height, dtype=dtype, device=device),
        torch.arange(plane.width, dtype=dtype, device=device),
        indexing="ij",
    )
    source = _map_points(
        torch.stack((xs, ys), dim=-1), matrix=np.linalg.inv(plane.matrix)
    )

    return source[..., 0], source[..., 1]


def _correct_detections(
    detections: Any, *, plane: TopDownPlane, frame: FrameRef
) -> Any:
    if detections is None:
        return None

    xyxy = detections.xyxy
    corners = torch.round(xyxy).reshape(-1, 2, 2)
    x_min, y_min = corners[:, 0, 0], corners[:, 0, 1]
    x_max, y_max = corners[:, 1, 0], corners[:, 1, 1]
    quad = torch.stack(
        (
            torch.stack((x_min, y_min), dim=-1),
            torch.stack((x_max, y_min), dim=-1),
            torch.stack((x_max, y_max), dim=-1),
            torch.stack((x_min, y_max), dim=-1),
        ),
        dim=1,
    )
    # cv2.perspectiveTransform computes in float64 and stores float32; V1
    # rounded that float32 result.
    mapped = _map_points(quad, matrix=plane.matrix).to(torch.float32)
    boxes = torch.cat((mapped.amin(dim=1), mapped.amax(dim=1)), dim=1).round()

    changes: Dict[str, Any] = {
        "image_metadata": _frame_metadata(detections.image_metadata, frame=frame),
        "bboxes_metadata": _corrected_rows(detections.bboxes_metadata, plane=plane),
    }
    if isinstance(detections, InstanceDetections):
        dense = _warp_masks(_dense_masks(detections), plane=plane)
        has_pixels = dense.flatten(1).any(dim=1).to(boxes.device)
        boxes = torch.where(
            has_pixels.unsqueeze(1), _mask_boxes(dense, like=boxes), boxes
        )
        changes["mask"] = (
            _encode_rle(dense)
            if isinstance(detections.mask, InstancesRLEMasks)
            else dense.to(detections.mask.device)
        )
    changes["xyxy"] = boxes.to(device=xyxy.device, dtype=xyxy.dtype)
    corrected = replace(detections, **changes)

    return corrected


def _correct_key_points(
    key_points: KeyPoints, *, plane: TopDownPlane, frame: FrameRef
) -> KeyPoints:
    xy = key_points.xy
    mapped = _map_points(xy, matrix=plane.matrix)
    changes: Dict[str, Any] = {
        "xy": mapped.to(dtype=xy.dtype),
        "image_metadata": _frame_metadata(key_points.image_metadata, frame=frame),
    }
    covariance = key_points.covariance
    if covariance is not None:
        if covariance.shape != (*xy.shape[:-1], 2, 2):
            raise ContractError(
                "keypoint covariance must have shape (instances, key points, 2, 2) "
                f"matching xy {tuple(xy.shape)}, got {tuple(covariance.shape)}"
            )
        jacobian = _point_jacobians(xy, mapped=mapped, matrix=plane.matrix)
        moved = jacobian @ covariance.to(jacobian) @ jacobian.transpose(-1, -2)
        changes["covariance"] = moved.to(
            device=covariance.device, dtype=covariance.dtype
        )
    corrected = replace(key_points, **changes)

    return corrected


def _point_jacobians(
    points: torch.Tensor, *, mapped: torch.Tensor, matrix: np.ndarray
) -> torch.Tensor:
    """Jacobian ``d mapped / d points`` of the homography, ``(..., 2, 2)``.

    With ``w = h20 x + h21 y + h22`` and ``mapped = (u, v)``, row ``i`` is
    ``(H[i, :2] - mapped_i * H[2, :2]) / w``.
    """
    homography = torch.as_tensor(matrix, dtype=mapped.dtype, device=mapped.device)
    scale = points.to(mapped) @ homography[2, :2] + homography[2, 2]
    numerator = homography[:2, :2] - mapped.unsqueeze(-1) * homography[2, :2]
    jacobian = numerator / scale[..., None, None]

    return jacobian


def _frame_metadata(
    image_metadata: Optional[Dict[str, Any]], *, frame: FrameRef
) -> Dict[str, Any]:
    # Other keys (class names, row identity) are kept; frame keys describe
    # the rectangle. Stale composite keys of the input go.
    metadata = dict(image_metadata or {})
    for key in ("is_composite", "composite_sources"):
        metadata.pop(key, None)
    metadata.update(frame.prediction_metadata())

    return metadata


def _corrected_rows(
    rows: Optional[List[Dict[str, Any]]], *, plane: TopDownPlane
) -> Optional[List[Dict[str, Any]]]:
    # Point metadata per row, rounded as V1 did; host mirrors are stale.
    if rows is None:
        return None

    corrected = []
    for row in rows:
        entry = without_host_mirrors(row)
        key_points = entry.get(KEYPOINTS_XY_KEY_IN_SV_DETECTIONS)
        if key_points is not None and np.asarray(key_points).size > 0:
            mapped = _map_host_points(key_points, matrix=plane.matrix)
            entry[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = (
                np.around(mapped).astype(np.int32).tolist()
            )
        oriented = entry.get(ORIENTED_BOX_KEY)
        if oriented is not None:
            mapped = _map_host_points(oriented, matrix=plane.matrix)
            entry[ORIENTED_BOX_KEY] = np.around(mapped).astype(np.float32)
        corrected.append(entry)

    return corrected


def _map_host_points(points: Any, *, matrix: np.ndarray) -> np.ndarray:
    xy = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    mapped = cv2.perspectiveTransform(xy, matrix).reshape(-1, 2)

    return mapped


def _dense_masks(detections: InstanceDetections) -> torch.Tensor:
    if isinstance(detections.mask, InstancesRLEMasks):
        decoded = torch.from_numpy(decode_rle_masks(detections.mask))
        return decoded
    return detections.mask.to(torch.bool)


def _warp_masks(masks: torch.Tensor, *, plane: TopDownPlane) -> torch.Tensor:
    # Each rectangle pixel takes the nearest input mask pixel; outside is empty.
    height, width = masks.shape[-2:]
    source_x, source_y = _source_grid(plane, device=masks.device)
    column = torch.nan_to_num(torch.floor(source_x + 0.5), nan=-1.0).clamp(-1, width)
    row = torch.nan_to_num(torch.floor(source_y + 0.5), nan=-1.0).clamp(-1, height)
    inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    column = column.clamp(0, width - 1).long()
    row = row.clamp(0, height - 1).long()
    warped = masks[:, row, column] & inside

    return warped


def _mask_boxes(masks: torch.Tensor, *, like: torch.Tensor) -> torch.Tensor:
    # Inclusive pixel extent of each mask, as supervision's mask_to_xyxy.
    height, width = masks.shape[-2:]
    columns = torch.arange(width, device=masks.device)
    rows = torch.arange(height, device=masks.device)
    in_column = masks.any(dim=1)
    in_row = masks.any(dim=2)
    x_min = torch.where(in_column, columns, width).amin(dim=1)
    x_max = torch.where(in_column, columns, -1).amax(dim=1)
    y_min = torch.where(in_row, rows, height).amin(dim=1)
    y_max = torch.where(in_row, rows, -1).amax(dim=1)
    boxes = torch.stack((x_min, y_min, x_max, y_max), dim=1).to(
        device=like.device, dtype=like.dtype
    )

    return boxes


def _encode_rle(masks: torch.Tensor) -> InstancesRLEMasks:
    height, width = masks.shape[-2:]
    encoded = InstancesRLEMasks(
        image_size=(height, width),
        masks=[torch_mask_to_coco_rle(mask.cpu())["counts"] for mask in masks],
    )

    return encoded
