"""Geometry for the MediaPipe hands pipeline, re-implemented around plain ONNX graphs.

MediaPipe ships the two models (palm detector and hand landmarks) as TFLite
graphs inside ``hand_landmarker.task``; the work between them lives in
MediaPipe's C++ calculators. This module reproduces those calculators in pixel
space so the converted ONNX graphs give the same answers without the library:

* SSD anchors and raw-tensor decoding of the palm detector
  (``SsdAnchorsCalculator``, ``TensorsToDetectionsCalculator``),
* weighted non-maximum suppression (``NonMaxSuppressionCalculator``),
* the rotated hand region of interest built from a palm detection or from the
  previous landmarks (``DetectionsToRectsCalculator``,
  ``RectTransformationCalculator``, ``HandLandmarksToRectCalculator``),
* the affine crop the landmark model sees and the projection of its output
  back into the image (``ImageToTensorCalculator``,
  ``LandmarkProjectionCalculator``, ``WorldLandmarkProjectionCalculator``).

A rotated rectangle is ``(center_x, center_y, width, height, rotation)`` in
image pixels, rotation in radians, clockwise in image coordinates.
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

PALM_INPUT_SIZE = 192
LANDMARKS_INPUT_SIZE = 224
PALM_KEY_POINTS = 7
HAND_LANDMARKS = 21

# Palm detector anchors (full model). Anchors are unit-size, so the
# calculator's min/max scales do not affect decoding.
_ANCHOR_STRIDES = (8, 16, 16, 16)
_ANCHOR_OFFSET = 0.5

_SCORE_CLIPPING_THRESHOLD = 100.0
DEFAULT_PALM_NMS_IOU_THRESHOLD = 0.3

# Palm detection -> hand region: wrist centre to middle finger base, pointing up.
_PALM_ROTATION_START_KEY_POINT = 0
_PALM_ROTATION_END_KEY_POINT = 2
_PALM_RECT_SCALE = 2.6
_PALM_RECT_SHIFT_Y = -0.5

# Landmarks -> hand region (tracking and refinement).
_LANDMARKS_RECT_SUBSET = (0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18)
_WRIST = 0
_INDEX_FINGER_MCP = 5
_MIDDLE_FINGER_MCP = 9
_RING_FINGER_MCP = 13
_LANDMARKS_RECT_SCALE = 2.0
_LANDMARKS_RECT_SHIFT_Y = -0.1

_TARGET_ANGLE = math.pi / 2


@dataclass(frozen=True)
class RotatedRect:
    center_x: float
    center_y: float
    width: float
    height: float
    rotation: float


@dataclass(frozen=True)
class PalmDetection:
    """One palm in image pixels: the box, its seven key points and the score."""

    xyxy: np.ndarray  # (4,)
    key_points: np.ndarray  # (7, 2)
    score: float


def generate_palm_anchors(input_size: int = PALM_INPUT_SIZE) -> np.ndarray:
    """Anchor centres of the palm detector, normalized to the model input, shape (2016, 2).

    Every anchor has unit width and height (``fixed_anchor_size``), so only the
    centres are needed. Layers that share a stride are emitted together, two
    anchors per cell for each layer, in the order the graph's output uses.
    """
    num_layers = len(_ANCHOR_STRIDES)
    centres = []
    layer = 0
    while layer < num_layers:
        stride = _ANCHOR_STRIDES[layer]
        anchors_per_cell = 0
        last_same_stride = layer
        while (
            last_same_stride < num_layers
            and _ANCHOR_STRIDES[last_same_stride] == stride
        ):
            # One anchor for aspect ratio 1.0 plus the interpolated-scale anchor.
            anchors_per_cell += 2
            last_same_stride += 1
        feature_map_size = math.ceil(input_size / stride)
        for y in range(feature_map_size):
            for x in range(feature_map_size):
                centre = (
                    (x + _ANCHOR_OFFSET) / feature_map_size,
                    (y + _ANCHOR_OFFSET) / feature_map_size,
                )
                centres.extend([centre] * anchors_per_cell)
        layer = last_same_stride
    return np.asarray(centres, dtype=np.float32)


def letterbox_matrix(
    image_width: int, image_height: int, input_size: int = PALM_INPUT_SIZE
) -> np.ndarray:
    """Affine (2, 3) that fits the image into a square input, centred, aspect kept."""
    scale = min(input_size / image_width, input_size / image_height)
    offset_x = (input_size - image_width * scale) / 2
    offset_y = (input_size - image_height * scale) / 2
    return np.asarray(
        [[scale, 0.0, offset_x], [0.0, scale, offset_y]], dtype=np.float64
    )


def decode_palm_detections(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    anchors: np.ndarray,
    image_to_input: np.ndarray,
    score_threshold: float,
    iou_threshold: float = DEFAULT_PALM_NMS_IOU_THRESHOLD,
    max_detections: int = 0,
    input_size: int = PALM_INPUT_SIZE,
) -> List[PalmDetection]:
    """Decode one image's palm-detector output into palms in image pixels.

    ``raw_boxes`` is (2016, 18): box centre and size, then seven key points,
    all as offsets in input pixels from the anchor centre. ``raw_scores`` is
    (2016,) or (2016, 1) logits. ``max_detections`` of 0 keeps every palm.
    """
    logits = np.clip(
        raw_scores.reshape(-1).astype(np.float32),
        -_SCORE_CLIPPING_THRESHOLD,
        _SCORE_CLIPPING_THRESHOLD,
    )
    scores = 1.0 / (1.0 + np.exp(-logits))
    keep = np.flatnonzero(scores >= score_threshold)
    if keep.size == 0:
        return []
    raw = raw_boxes[keep].astype(np.float32) / input_size
    anchor_centres = anchors[keep]
    centre = raw[:, 0:2] + anchor_centres
    half_size = raw[:, 2:4] / 2
    boxes = np.concatenate([centre - half_size, centre + half_size], axis=1)
    key_points = raw[:, 4:].reshape(-1, PALM_KEY_POINTS, 2) + anchor_centres[:, None]
    boxes, key_points, kept_scores = weighted_non_max_suppression(
        boxes=boxes,
        key_points=key_points,
        scores=scores[keep],
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    input_to_image = cv2.invertAffineTransform(image_to_input)
    detections = []
    for box, points, score in zip(boxes, key_points, kept_scores):
        corners = _apply_affine(box.reshape(2, 2) * input_size, input_to_image)
        image_points = _apply_affine(points * input_size, input_to_image)
        detections.append(
            PalmDetection(
                xyxy=corners.reshape(4).astype(np.float32),
                key_points=image_points.astype(np.float32),
                score=float(score),
            )
        )
    return detections


def weighted_non_max_suppression(
    boxes: np.ndarray,
    key_points: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MediaPipe's WEIGHTED non-maximum suppression.

    Unlike plain NMS, the boxes overlapping the best remaining box are not
    discarded: the kept box and key points are their score-weighted average,
    which steadies the palm between frames. The kept score is the best score.
    """
    order = np.argsort(-scores, kind="stable")
    remaining = list(order)
    kept_boxes, kept_points, kept_scores = [], [], []
    while remaining and (max_detections <= 0 or len(kept_boxes) < max_detections):
        best = remaining[0]
        ious = _iou(boxes[best], boxes[remaining])
        overlapping = [
            index for index, iou in zip(remaining, ious) if iou > iou_threshold
        ]
        remaining = [
            index for index, iou in zip(remaining, ious) if iou <= iou_threshold
        ]
        weights = scores[overlapping][:, None]
        total = float(weights.sum())
        kept_boxes.append((boxes[overlapping] * weights).sum(axis=0) / total)
        kept_points.append(
            (key_points[overlapping] * weights[:, :, None]).sum(axis=0) / total
        )
        kept_scores.append(scores[best])
    if not kept_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, PALM_KEY_POINTS, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.stack(kept_boxes).astype(np.float32),
        np.stack(kept_points).astype(np.float32),
        np.asarray(kept_scores, dtype=np.float32),
    )


def palm_to_hand_rect(palm: PalmDetection) -> RotatedRect:
    """The region the landmark model should see for a detected palm."""
    start = palm.key_points[_PALM_ROTATION_START_KEY_POINT]
    end = palm.key_points[_PALM_ROTATION_END_KEY_POINT]
    rotation = _rotation_towards(start=start, end=end)
    x_min, y_min, x_max, y_max = (float(v) for v in palm.xyxy)
    return transform_rect(
        RotatedRect(
            center_x=(x_min + x_max) / 2,
            center_y=(y_min + y_max) / 2,
            width=x_max - x_min,
            height=y_max - y_min,
            rotation=rotation,
        ),
        scale=_PALM_RECT_SCALE,
        shift_y=_PALM_RECT_SHIFT_Y,
    )


def landmarks_to_hand_rect(landmarks_xy: np.ndarray) -> RotatedRect:
    """The region for the next landmark pass, from the current landmarks in image pixels.

    MediaPipe uses this to follow a hand between video frames without running
    the palm detector again; here it also turns an upright detector box into a
    rotated region after a first landmark pass.
    """
    wrist = landmarks_xy[_WRIST]
    knuckles = (
        (landmarks_xy[_INDEX_FINGER_MCP] + landmarks_xy[_RING_FINGER_MCP]) / 2
        + landmarks_xy[_MIDDLE_FINGER_MCP]
    ) / 2
    rotation = _rotation_towards(start=wrist, end=knuckles)
    subset = landmarks_xy[list(_LANDMARKS_RECT_SUBSET)].astype(np.float64)
    axis_aligned_centre = (subset.min(axis=0) + subset.max(axis=0)) / 2
    upright = _rotate(subset - axis_aligned_centre, -rotation)
    upright_min, upright_max = upright.min(axis=0), upright.max(axis=0)
    centre = _rotate(((upright_min + upright_max) / 2)[None], rotation)[0]
    centre = centre + axis_aligned_centre
    size = upright_max - upright_min
    return transform_rect(
        RotatedRect(
            center_x=float(centre[0]),
            center_y=float(centre[1]),
            width=float(size[0]),
            height=float(size[1]),
            rotation=rotation,
        ),
        scale=_LANDMARKS_RECT_SCALE,
        shift_y=_LANDMARKS_RECT_SHIFT_Y,
    )


def box_to_hand_rect(xyxy: Sequence[float], scale: float) -> RotatedRect:
    """An upright square region around a hand box from any detector."""
    x_min, y_min, x_max, y_max = (float(v) for v in xyxy)
    side = max(x_max - x_min, y_max - y_min) * scale
    return RotatedRect(
        center_x=(x_min + x_max) / 2,
        center_y=(y_min + y_max) / 2,
        width=side,
        height=side,
        rotation=0.0,
    )


def transform_rect(rect: RotatedRect, scale: float, shift_y: float) -> RotatedRect:
    """Shift along the rect's own axis, square it on its long side, then scale."""
    sin, cos = math.sin(rect.rotation), math.cos(rect.rotation)
    centre_x = rect.center_x - rect.height * shift_y * sin
    centre_y = rect.center_y + rect.height * shift_y * cos
    side = max(rect.width, rect.height)
    return RotatedRect(
        center_x=centre_x,
        center_y=centre_y,
        width=side * scale,
        height=side * scale,
        rotation=rect.rotation,
    )


def crop_to_image_matrix(rect: RotatedRect, crop_size: int) -> np.ndarray:
    """Affine (2, 3) from crop pixels to image pixels for a rotated rect."""
    sin, cos = math.sin(rect.rotation), math.cos(rect.rotation)
    scale_x = rect.width / crop_size
    scale_y = rect.height / crop_size
    linear = np.asarray(
        [[cos * scale_x, -sin * scale_y], [sin * scale_x, cos * scale_y]],
        dtype=np.float64,
    )
    half = crop_size / 2
    translation = np.asarray([rect.center_x, rect.center_y]) - linear @ np.asarray(
        [half, half]
    )
    return np.concatenate([linear, translation[:, None]], axis=1)


def crop_rect(image: np.ndarray, rect: RotatedRect, crop_size: int) -> np.ndarray:
    """Cut the rotated rect out of the image as a square crop, zero outside the image."""
    image_to_crop = cv2.invertAffineTransform(crop_to_image_matrix(rect, crop_size))
    return cv2.warpAffine(
        image,
        image_to_crop,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def project_landmarks(
    landmarks: np.ndarray, rect: RotatedRect, crop_size: int
) -> np.ndarray:
    """Map (21, 3) landmarks from crop pixels to image pixels.

    ``z`` is relative depth in the crop's pixel scale; it is rescaled with the
    rect so it stays comparable to ``x`` in image pixels.
    """
    matrix = crop_to_image_matrix(rect, crop_size)
    xy = _apply_affine(landmarks[:, :2].astype(np.float64), matrix)
    z = landmarks[:, 2:3].astype(np.float64) * rect.width / crop_size
    return np.concatenate([xy, z], axis=1).astype(np.float32)


def project_world_landmarks(world: np.ndarray, rect: RotatedRect) -> np.ndarray:
    """Rotate (21, 3) metric landmarks by the region's rotation; depth is unchanged."""
    xy = _rotate(world[:, :2].astype(np.float64), rect.rotation)
    return np.concatenate([xy, world[:, 2:3]], axis=1).astype(np.float32)


def landmarks_box(landmarks_xy: np.ndarray) -> np.ndarray:
    """Axis-aligned (4,) box around the landmarks in image pixels."""
    return np.concatenate([landmarks_xy.min(axis=0), landmarks_xy.max(axis=0)]).astype(
        np.float32
    )


def _rotation_towards(start: np.ndarray, end: np.ndarray) -> float:
    """Rotation that turns the start->end vector to point straight up in the image."""
    angle = _TARGET_ANGLE - math.atan2(
        -(float(end[1]) - float(start[1])), float(end[0]) - float(start[0])
    )
    return _normalize_radians(angle)


def _normalize_radians(angle: float) -> float:
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    sin, cos = math.sin(angle), math.cos(angle)
    rotation = np.asarray([[cos, -sin], [sin, cos]], dtype=np.float64)
    return points @ rotation.T


def _apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x_min = np.maximum(box[0], boxes[:, 0])
    y_min = np.maximum(box[1], boxes[:, 1])
    x_max = np.minimum(box[2], boxes[:, 2])
    y_max = np.minimum(box[3], boxes[:, 3])
    intersection = np.clip(x_max - x_min, 0, None) * np.clip(y_max - y_min, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)
