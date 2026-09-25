"""Helpers for reading keypoint metadata stored on ``sv.Detections``.

Keypoints are inherently ragged, because different detections carry different
numbers of keypoints (multi-class skeletons, or per-keypoint confidence
filtering). To keep
them as a proper rectangular numpy array (object-dtype ragged arrays break
supervision's ``is_data_equal`` used during indexing/comparison),
``add_inference_keypoints_to_sv_detections`` right-pads every detection to the
batch-wide maximum with placeholder slots.

Padding slots are the single source of ambiguity for every consumer that emits
keypoints (serialization, sinks), so the padding contract lives here in one
place: a padding slot carries the empty class name below, while a real keypoint
always carries a non-empty class name (models label keypoints from a skeleton
map). Padding is always appended after a detection's real keypoints, so the real
keypoints are exactly the leading ``real_keypoints_count(...)`` slots.
"""

from typing import Iterable, Optional, Sequence

# Class name written into padding slots by
# ``add_inference_keypoints_to_sv_detections``. Real keypoints never carry it.
KEYPOINT_PADDING_CLASS_NAME = ""

# Bounds the dense keypoint arrays to about 28 MB on 64-bit NumPy (12 MB for
# torch). A limit on real keypoints alone would not bound ragged padding.
MAX_KEYPOINTS_PADDING_CELLS = 1_000_000

# Bounds the slot count of one skeleton when keypoints are placed by class id.
# Class ids arrive unchecked from runtime input and remote responses, and the
# cell limit above bounds memory only: a single keypoint with class id 999,999
# stays under it and would still send supervision's annotators through a
# million-slot Python loop per frame. No real skeleton comes close to this.
MAX_KEYPOINT_SLOTS = 1_024


def validate_keypoints_padding(detections_count: int, max_keypoints: int) -> None:
    padding_cells = detections_count * max_keypoints
    if padding_cells > MAX_KEYPOINTS_PADDING_CELLS:
        raise ValueError(
            f"Keypoint padding requires {padding_cells} slots, exceeding the limit "
            f"of {MAX_KEYPOINTS_PADDING_CELLS}. Reduce the number of detections "
            "or keypoints per detection."
        )


def real_keypoints_count(keypoint_class_names: Optional[Sequence], total: int) -> int:
    """Return how many of a detection's keypoint slots are real (not padding).

    Args:
        keypoint_class_names: The detection's per-keypoint class names (the
            ``keypoints_class_name`` slice for a single detection). ``None`` when
            the keypoints were stored without class names.
        total: Number of keypoint slots present for the detection, used as the
            fallback when ``keypoint_class_names`` is unavailable, so keypoints
            carrying no class-name metadata are emitted unchanged.

    Returns:
        The count of leading, non-padding keypoints for the detection.
    """
    if keypoint_class_names is None:
        return total
    return sum(
        1
        for class_name in keypoint_class_names
        if str(class_name) != KEYPOINT_PADDING_CLASS_NAME
    )


# Keypoint names of the COCO person skeleton, in skeleton order. A keypoint whose
# ``class_id`` is the index of its name in this tuple sits in its COCO slot.
COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def is_coco_skeleton(
    keypoint_class_ids: Iterable[int], keypoint_class_names: Iterable[str]
) -> bool:
    """Return whether every keypoint sits at its COCO skeleton slot.

    True only when there is at least one keypoint and each ``(class_id, name)``
    pair satisfies ``COCO_KEYPOINT_NAMES[class_id] == name``. Matching on names
    alone is not enough: a custom skeleton that reuses COCO names in its own
    order would then be drawn with COCO bones between the wrong joints. Padding
    slots must be excluded by the caller.
    """
    pairs = list(zip(keypoint_class_ids, keypoint_class_names))
    if not pairs:
        return False
    for class_id, class_name in pairs:
        try:
            index = int(class_id)
        except (TypeError, ValueError):
            return False
        if index < 0 or index >= len(COCO_KEYPOINT_NAMES):
            return False
        if COCO_KEYPOINT_NAMES[index] != str(class_name):
            return False
    return True
