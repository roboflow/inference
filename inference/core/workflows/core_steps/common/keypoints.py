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

from typing import Optional, Sequence

# Class name written into padding slots by
# ``add_inference_keypoints_to_sv_detections``. Real keypoints never carry it.
KEYPOINT_PADDING_CLASS_NAME = ""


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
