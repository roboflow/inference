import base64
from datetime import datetime

import cv2
import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.common.serializers import (
    serialise_image,
    serialise_sv_detections,
    serialize_wildcard_kind,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
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
            "data": np.array(["some", "other"]),
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
                "data": "some",
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
                "data": "other",
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


def test_serialise_image() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np_image,
    )

    # when
    result = serialise_image(image=image)

    # then
    assert result["type"] == "base64", "Type of image must point base64"
    decoded = base64.b64decode(result["value"])
    recovered_image = cv2.imdecode(
        np.fromstring(decoded, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"


def test_serialize_wildcard_kind_when_workflow_image_data_is_given() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    value = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np_image,
    )

    # when
    result = serialize_wildcard_kind(value=value)

    # then
    assert (
        result["type"] == "base64"
    ), "Type of third element must be changed into base64"
    decoded = base64.b64decode(result["value"])
    recovered_image = cv2.imdecode(
        np.fromstring(decoded, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"


def test_serialize_wildcard_kind_when_dictionary_is_given() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    elements = {
        "a": 3,
        "b": "some",
        "c": WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np_image,
        ),
    }

    # when
    result = serialize_wildcard_kind(value=elements)

    # then
    assert len(result) == 3, "The same number of elements must be returned"
    assert result["a"] == 3, "First element of list must be untouched"
    assert result["b"] == "some", "Second element of list must be untouched"
    assert (
        result["c"]["type"] == "base64"
    ), "Type of third element must be changed into base64"
    decoded = base64.b64decode(result["c"]["value"])
    recovered_image = cv2.imdecode(
        np.fromstring(decoded, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"


def test_serializer_serialize_wildcard_kind_when_timestamps_are_present() -> None:
    # given
    timestamp = datetime.now()
    expected_result = timestamp.isoformat()
    input_value = {
        "some": [1, 2, timestamp],
        "other": [1, None, {"value": timestamp}],
        "yet-another": timestamp,
    }

    # when
    result = serialize_wildcard_kind(value=input_value)

    # then
    assert result == {
        "some": [1, 2, expected_result],
        "other": [1, None, {"value": expected_result}],
        "yet-another": expected_result,
    }


def test_serialize_wildcard_kind_when_list_is_given() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    elements = [
        3,
        "some",
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np_image,
        ),
    ]

    # when
    result = serialize_wildcard_kind(value=elements)

    # then
    assert len(result) == 3, "The same number of elements must be returned"
    assert result[0] == 3, "First element of list must be untouched"
    assert result[1] == "some", "Second element of list must be untouched"
    assert (
        result[2]["type"] == "base64"
    ), "Type of third element must be changed into base64"
    decoded = base64.b64decode(result[2]["value"])
    recovered_image = cv2.imdecode(
        np.fromstring(decoded, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"


def test_serialize_wildcard_kind_when_compound_input_is_given() -> None:
    # given
    np_image = np.zeros((192, 168, 3), dtype=np.uint8)
    elements = [
        3,
        "some",
        WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="some"),
            numpy_image=np_image,
        ),
        {
            "nested": [
                WorkflowImageData(
                    parent_metadata=ImageParentMetadata(parent_id="other"),
                    numpy_image=np_image,
                )
            ]
        },
    ]

    # when
    result = serialize_wildcard_kind(value=elements)

    # then
    assert len(result) == 4, "The same number of elements must be returned"
    assert result[0] == 3, "First element of list must be untouched"
    assert result[1] == "some", "Second element of list must be untouched"
    assert (
        result[2]["type"] == "base64"
    ), "Type of third element must be changed into base64"
    decoded = base64.b64decode(result[2]["value"])
    recovered_image = cv2.imdecode(
        np.fromstring(decoded, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"
    nested_dict = result[3]
    assert len(nested_dict["nested"]) == 1, "Expected one element in nested list"
    assert (
        nested_dict["nested"][0]["type"] == "base64"
    ), "Expected image serialized to base64"
    assert (
        "video_metadata" in nested_dict["nested"][0]
    ), "Expected video metadata attached"
    decoded = base64.b64decode(nested_dict["nested"][0]["value"])
    recovered_image = cv2.imdecode(
        np.fromstring(decoded, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"
