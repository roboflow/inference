"""Hand region reuse across video frames, as MediaPipe's hand landmarker does it.

On video, MediaPipe does not run the palm detector on every frame. The
landmarks found on one frame give the hand regions for the next
(``landmarks_to_hand_rect``), and the detector only runs while fewer hands are
tracked than asked for. New detections join the tracked regions only where
they do not overlap one (``HandAssociationCalculator``, tracked regions
first), and hands the landmark model finds twice are dropped
(``HandLandmarksDeduplicationCalculator``).

These functions hold the policy; the caller keeps the per-video state.
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from inference_models.models.mediapipe_hands.geometry import RotatedRect, rect_iou

DEFAULT_MIN_SIMILARITY = 0.5
DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.6


@dataclass(frozen=True)
class TrackedHand:
    """A hand region and the detection score that first found it."""

    rect: RotatedRect
    score: float


def needs_detection(tracked: Sequence[TrackedHand], *, max_num_hands: int) -> bool:
    """MediaPipe skips detection only once it already tracks ``max_num_hands`` hands."""
    return len(tracked) < max_num_hands


def merge_tracked_and_detected(
    tracked: Sequence[TrackedHand],
    detected: Sequence[TrackedHand],
    *,
    max_num_hands: int,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> List[TrackedHand]:
    """Tracked regions, then each detection that overlaps none of the kept regions.

    Args:
        tracked: Regions carried over from the previous frame's landmarks.
        detected: Regions from this frame's detector, best first.
        max_num_hands: Most regions returned.
        min_similarity: A detection whose IoU with a kept region is above this
            is the same hand and is dropped.

    Returns:
        At most ``max_num_hands`` regions, tracked ones first.
    """
    merged = list(tracked)
    for candidate in detected:
        if all(
            rect_iou(candidate.rect, kept.rect) <= min_similarity for kept in merged
        ):
            merged.append(candidate)
    return merged[:max_num_hands]


def deduplicate_hands(
    boxes_xyxy: np.ndarray,
    presences: Sequence[float],
    *,
    iou_threshold: float = DEFAULT_DUPLICATE_IOU_THRESHOLD,
) -> List[int]:
    """Indices of the hands to keep when two regions converge on one hand.

    Two regions can drift onto the same hand while tracking. Of hands whose
    landmark boxes overlap by more than ``iou_threshold``, the one with the
    higher presence is kept. Returned indices are in input order.
    """
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(presences, dtype=np.float64), kind="stable")
    kept: List[int] = []
    for index in order:
        if all(_box_iou(boxes[index], boxes[other]) <= iou_threshold for other in kept):
            kept.append(int(index))
    return sorted(kept)


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x_min, y_min = np.maximum(first[:2], second[:2])
    x_max, y_max = np.minimum(first[2:], second[2:])
    intersection = max(x_max - x_min, 0.0) * max(y_max - y_min, 0.0)
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return float(intersection / union) if union > 0 else 0.0
