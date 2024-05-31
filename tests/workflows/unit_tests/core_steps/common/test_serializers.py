import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
)


def test_serialise_sv_detections() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        tracker_id=np.array([1, 2]),
        mask=np.array(
            [
                sv.polygon_to_mask(
                    np.array([[1, 1], [1, 10], [10, 10], [10, 1]]),
                    resolution_wh=(15, 15),
                ),
                sv.polygon_to_mask(
                    np.array([[1, 1], [1, 10], [10, 10], [10, 1]]),
                    resolution_wh=(15, 15),
                ),
            ],
            dtype=bool,
        ),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
            "keypoints_xy": np.array(
                [
                    np.array([[11, 11], [12, 13], [14, 15]], dtype=np.float64),
                    np.array(
                        [[16, 16], [17, 17], [18, 18], [19, 19]], dtype=np.float64
                    ),
                ],
                dtype="object",
            ),
            "keypoints_class_id": np.array(
                [
                    np.array([1, 2, 3]),
                    np.array([1, 2, 3, 4]),
                ],
                dtype="object",
            ),
            "keypoints_class_name": np.array(
                [
                    np.array(["nose", "ear", "eye"]),
                    np.array(["nose", "ear", "eye", "tail"]),
                ],
                dtype="object",
            ),
            "keypoints_confidence": np.array(
                [
                    np.array([0.1, 0.2, 0.3], dtype=np.float64),
                    np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
                ],
                dtype="object",
            ),
            "parent_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
            "image_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
        },
    )

    # when
    result = serialise_sv_detections(detections=detections)

    # then
    assert result == {
        "image": {
            "width": 168,
            "height": 192,
        },
        "predictions": [
            {
                "width": 1.0,
                "height": 1.0,
                "x": 1.5,
                "y": 1.5,
                "confidence": 0.1,
                "class_id": 1,
                "points": [
                    {"x": 1.0, "y": 1.0},
                    {"x": 1.0, "y": 10.0},
                    {"x": 10.0, "y": 10.0},
                    {"x": 10.0, "y": 1.0},
                ],
                "tracker_id": 1,
                "class": "cat",
                "detection_id": "first",
                "parent_id": "image",
                "keypoints": [
                    {
                        "class_id": 1,
                        "class": "nose",
                        "confidence": 0.1,
                        "x": 11.0,
                        "y": 11.0,
                    },
                    {
                        "class_id": 2,
                        "class": "ear",
                        "confidence": 0.2,
                        "x": 12.0,
                        "y": 13.0,
                    },
                    {
                        "class_id": 3,
                        "class": "eye",
                        "confidence": 0.3,
                        "x": 14.0,
                        "y": 15.0,
                    },
                ],
            },
            {
                "width": 1.0,
                "height": 1.0,
                "x": 3.5,
                "y": 3.5,
                "confidence": 0.9,
                "class_id": 2,
                "points": [
                    {"x": 1.0, "y": 1.0},
                    {"x": 1.0, "y": 10.0},
                    {"x": 10.0, "y": 10.0},
                    {"x": 10.0, "y": 1.0},
                ],
                "tracker_id": 2,
                "class": "dog",
                "detection_id": "second",
                "parent_id": "image",
                "keypoints": [
                    {
                        "class_id": 1,
                        "class": "nose",
                        "confidence": 0.1,
                        "x": 16.0,
                        "y": 16.0,
                    },
                    {
                        "class_id": 2,
                        "class": "ear",
                        "confidence": 0.2,
                        "x": 17.0,
                        "y": 17.0,
                    },
                    {
                        "class_id": 3,
                        "class": "eye",
                        "confidence": 0.3,
                        "x": 18.0,
                        "y": 18.0,
                    },
                    {
                        "class_id": 4,
                        "class": "tail",
                        "confidence": 0.4,
                        "x": 19.0,
                        "y": 19.0,
                    },
                ],
            },
        ],
    }
