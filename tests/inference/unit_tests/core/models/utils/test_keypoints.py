from typing import List

from inference.core.entities.responses.inference import Keypoint
from inference.core.models.utils.keypoints import (
    model_keypoints_to_response,
    superset_keypoints_count,
)


def test_superset_keypoints_count() -> None:
    # given
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
    # when
    keypoints_count = superset_keypoints_count(keypoints_metadata)
    # then
    assert keypoints_count == 17


def test_superset_keypoints_count_with_two_classes() -> None:
    # given
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
    # when
    keypoints_count = superset_keypoints_count(keypoints_metadata)
    # then
    assert keypoints_count == 13


def test_superset_keypoints_count_with_two_non_overlapping_classes() -> None:
    # given
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
            0: "nose1",
            1: "left_eye1",
            2: "right_eye2",
            3: "left_ear3",
            4: "right_ear4",
            5: "left_shoulder5",
            6: "right_shoulder6",
            7: "left_elbow7",
            8: "right_elbow8",
            9: "left_wrist9",
        },
    }
    # when
    keypoints_count = superset_keypoints_count(keypoints_metadata)
    # then
    assert keypoints_count == 10


def test_model_keypoints_to_response() -> None:
    # given
    keypoints_metadata = {
        0: {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
        }
    }
    keypoints = [100, 100, 0.5, 200, 200, 0.5, 300, 300, 0.5, 400, 400, 0.5]
    result = (
        model_keypoints_to_response(
            keypoints_metadata=keypoints_metadata,
            keypoints=keypoints,
            predicted_object_class_id=0,
            keypoint_confidence_threshold=0,
        ),
    )
    # List of keypoints
    assert result == (
        [
            Keypoint(x=100, y=100, confidence=0.5, class_id=0, **{"class": "nose"}),
            Keypoint(x=200, y=200, confidence=0.5, class_id=1, **{"class": "left_eye"}),
            Keypoint(
                x=300, y=300, confidence=0.5, class_id=2, **{"class": "right_eye"}
            ),
            Keypoint(x=400, y=400, confidence=0.5, class_id=3, **{"class": "left_ear"}),
        ],
    )


def test_model_keypoints_to_response_padded_points() -> None:
    # given
    keypoints_metadata = {
        0: {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
        },
        1: {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
        },
    }
    keypoints = [
        100,
        100,
        0.5,
        200,
        200,
        0.5,
        300,
        300,
        0.5,
        400,
        400,
        0.5,
        500,
        500,
        0.5,
    ]
    result = (
        model_keypoints_to_response(
            keypoints_metadata=keypoints_metadata,
            keypoints=keypoints,
            predicted_object_class_id=0,
            keypoint_confidence_threshold=0,
        ),
    )
    # List of keypoints
    assert result == (
        [
            Keypoint(x=100, y=100, confidence=0.5, class_id=0, **{"class": "nose"}),
            Keypoint(x=200, y=200, confidence=0.5, class_id=1, **{"class": "left_eye"}),
            Keypoint(
                x=300, y=300, confidence=0.5, class_id=2, **{"class": "right_eye"}
            ),
            Keypoint(x=400, y=400, confidence=0.5, class_id=3, **{"class": "left_ear"}),
        ],
    )
