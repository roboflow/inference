from inference.core.models.utils.keypoints import (
    superset_keypoints_count,
)


def test_superset_keypoints_count():
    keypoints_metadata = {
        0: {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle",
        }
    }
    keypoints_count = superset_keypoints_count(keypoints_metadata)
    assert keypoints_count == 17


def test_superset_keypoints_count_with_two_classes():
    keypoints_metadata = {
        0: {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
        },
        1: {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
        },
    }
    keypoints_count = superset_keypoints_count(keypoints_metadata)
    assert keypoints_count == 13
