import numpy as np

from inference.core.workflows.core_steps.common.keypoints import (
    KEYPOINT_PADDING_CLASS_NAME,
    real_keypoints_count,
)


def test_real_keypoints_count_counts_only_named_slots() -> None:
    # given a padded detection: 2 real keypoints, 2 trailing padding slots
    class_names = np.array(
        ["nose", "eye", KEYPOINT_PADDING_CLASS_NAME, KEYPOINT_PADDING_CLASS_NAME],
        dtype=object,
    )

    # when / then
    assert real_keypoints_count(class_names, total=len(class_names)) == 2


def test_real_keypoints_count_when_no_padding() -> None:
    class_names = np.array(["nose", "eye", "ear"], dtype=object)
    assert real_keypoints_count(class_names, total=len(class_names)) == 3


def test_real_keypoints_count_when_all_padding() -> None:
    class_names = np.array([KEYPOINT_PADDING_CLASS_NAME] * 3, dtype=object)
    assert real_keypoints_count(class_names, total=3) == 0


def test_real_keypoints_count_falls_back_to_total_without_names() -> None:
    # No class-name metadata -> keypoints are emitted unchanged (fallback to total).
    assert real_keypoints_count(None, total=4) == 4
