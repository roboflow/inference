import base64
from datetime import datetime

import cv2
import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.serializers import (
    mask_to_polygon,
    serialise_image,
    serialise_rle_sv_detections,
    serialise_sv_detections,
    serialize_wildcard_kind,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
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
    try:
        recovered_image = cv2.imdecode(
            np.frombuffer(decoded, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except Exception:
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
    try:
        recovered_image = cv2.imdecode(
            np.frombuffer(decoded, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except Exception:
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
    try:
        recovered_image = cv2.imdecode(
            np.frombuffer(decoded, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except Exception:
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
    try:
        recovered_image = cv2.imdecode(
            np.frombuffer(decoded, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except Exception:
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
    try:
        recovered_image = cv2.imdecode(
            np.frombuffer(decoded, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except Exception:
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
    try:
        recovered_image = cv2.imdecode(
            np.frombuffer(decoded, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except Exception:
        recovered_image = cv2.imdecode(
            np.fromstring(decoded, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    assert (
        recovered_image == np_image
    ).all(), "Recovered image should be equal to input image"


def test_mask_to_polygon_when_no_contours_to_be_found() -> None:
    # given
    mask = np.zeros((128, 128), dtype=np.uint8)

    # when
    result = mask_to_polygon(mask=mask)

    # then
    assert result is None, "No polygons should be manifested as None"


def test_mask_to_polygon_when_mask_contains_point() -> None:
    # given
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:41, 50:51] = 255

    # when
    result = mask_to_polygon(mask=mask)

    # then
    assert np.allclose(
        result, np.array([[50, 40]] * 3)
    ), "Expected single point to be duplicated"


def test_mask_to_polygon_when_mask_contains_line() -> None:
    # given
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:41, 50:60] = 255

    # when
    result = mask_to_polygon(mask=mask)

    # then
    assert np.allclose(
        result, np.array([[50, 40], [59, 40], [59, 40]])
    ), "Expected last point of the shape to be duplicated"


def test_mask_to_polygon_when_mask_contains_standard_shape() -> None:
    # given
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:50, 50:60] = 255

    # when
    result = mask_to_polygon(mask=mask)

    # then
    assert np.allclose(result, np.array([[50, 40], [50, 49], [59, 49], [59, 40]]))


def test_mask_to_polygon_when_mask_contains_multiple_shapes() -> None:
    # given
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:50, 50:60] = 255
    mask[90:100, 100:110] = 255

    # when
    result = mask_to_polygon(mask=mask)

    # then
    assert np.allclose(
        result,
        np.array(
            [
                [100, 90],
                [100, 99],
                [109, 99],
                [109, 90],
            ]
        ),
    ) or np.allclose(result, np.array([[50, 40], [50, 49], [59, 49], [59, 40]]))


def test_serialise_image_with_parent_origin_when_crop() -> None:
    # given
    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(
            parent_id="crop_id",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=50,
                left_top_y=75,
                origin_width=800,
                origin_height=600,
            ),
        ),
        workflow_root_ancestor_metadata=ImageParentMetadata(
            parent_id="original_image",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=150,
                left_top_y=200,
                origin_width=1920,
                origin_height=1080,
            ),
        ),
        numpy_image=np_image,
    )

    # when
    result = serialise_image(image=image)

    # then
    assert result["type"] == "base64"
    assert "parent_id" in result
    assert result["parent_id"] == "crop_id"
    assert "parent_origin" in result
    assert result["parent_origin"] == {
        "offset_x": 50,
        "offset_y": 75,
        "width": 800,
        "height": 600,
    }
    assert "root_parent_id" in result
    assert result["root_parent_id"] == "original_image"
    assert "root_parent_origin" in result
    assert result["root_parent_origin"] == {
        "offset_x": 150,
        "offset_y": 200,
        "width": 1920,
        "height": 1080,
    }


def test_serialise_image_without_parent_origin_when_not_crop() -> None:
    # given
    np_image = np.zeros((100, 100, 3), dtype=np.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="original_image"),
        numpy_image=np_image,
    )

    # when
    result = serialise_image(image=image)

    # then
    assert result["type"] == "base64"
    assert "parent_id" not in result
    assert "parent_origin" not in result
    assert "root_parent_id" not in result
    assert "root_parent_origin" not in result


def test_mask_to_polygon_output_reconstruction_when_output_was_padded() -> None:
    # given
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[40:41, 50:60] = 1

    # when
    serialisation_result = mask_to_polygon(mask=mask)
    de_serialisation_result = sv.polygon_to_mask(
        polygon=serialisation_result, resolution_wh=(128, 128)
    )

    # then
    assert np.allclose(
        serialisation_result, np.array([[50, 40], [59, 40], [59, 40]])
    ), "Expected last point of the shape to be duplicated"
    assert np.allclose(
        mask, de_serialisation_result
    ), "Expected reconstruction to be exact"


def test_serialise_sv_detections_when_mask_with_single_point_detected_present() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        tracker_id=np.array([1, 2]),
        mask=np.array(
            [
                sv.polygon_to_mask(
                    np.array([[1, 1]]),
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
                    {"x": 1.0, "y": 1.0},  # POINT IS DUPLICATED HERE
                    {"x": 1.0, "y": 1.0},  # POINT IS DUPLICATED HERE
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


def test_serialise_sv_detections_with_parent_origin_when_crop() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.1], dtype=np.float64),
        data={
            "class_name": np.array(["cat"]),
            "detection_id": np.array(["first"]),
            "parent_id": np.array(["crop_id"]),
            "root_parent_id": np.array(["original_image"]),
            "parent_coordinates": np.array([[50, 75]]),
            "parent_dimensions": np.array([[600, 800]]),
            "root_parent_coordinates": np.array([[150, 200]]),
            "root_parent_dimensions": np.array([[1080, 1920]]),
            "image_dimensions": np.array([[192, 168]]),
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
                "class": "cat",
                "detection_id": "first",
                "parent_id": "crop_id",
                "parent_origin": {
                    "offset_x": 50,
                    "offset_y": 75,
                    "width": 800,
                    "height": 600,
                },
                "root_parent_id": "original_image",
                "root_parent_origin": {
                    "offset_x": 150,
                    "offset_y": 200,
                    "width": 1920,
                    "height": 1080,
                },
            },
        ],
    }


def test_serialise_sv_detections_without_parent_origin_when_not_crop() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.1], dtype=np.float64),
        data={
            "class_name": np.array(["cat"]),
            "detection_id": np.array(["first"]),
            "parent_id": np.array(["original_image"]),
            "root_parent_id": np.array(["original_image"]),
            "parent_coordinates": np.array([[50, 75]]),
            "parent_dimensions": np.array([[600, 800]]),
            "root_parent_coordinates": np.array([[50, 75]]),
            "root_parent_dimensions": np.array([[600, 800]]),
            "image_dimensions": np.array([[192, 168]]),
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
                "class": "cat",
                "detection_id": "first",
                "parent_id": "original_image",
            },
        ],
    }


def test_serialise_sv_detections_when_empty_mask_detected() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        tracker_id=np.array([1, 2]),
        mask=np.array(
            [
                np.zeros((15, 15), dtype=np.uint8),
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
            {  # Expected only second prediction as first to be filtered by empty mask
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


def test_serialise_rle_sv_detections() -> None:
    # given
    rle_mask_1 = {"size": [192, 168], "counts": "abc123"}
    rle_mask_2 = {"size": [192, 168], "counts": "def456"}
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
            "image_dimensions": np.array(
                [
                    [192, 168],
                    [192, 168],
                ]
            ),
            "rle_mask": np.array([rle_mask_1, rle_mask_2], dtype=object),
        },
    )

    # when
    result = serialise_rle_sv_detections(detections=detections)

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
                "rle_mask": {"size": [192, 168], "counts": "abc123"},
                "tracker_id": 1,
                "class": "cat",
                "detection_id": "first",
                "parent_id": "image",
            },
            {
                "width": 1.0,
                "height": 1.0,
                "x": 3.5,
                "y": 3.5,
                "confidence": 0.9,
                "class_id": 2,
                "rle_mask": {"size": [192, 168], "counts": "def456"},
                "tracker_id": 2,
                "class": "dog",
                "detection_id": "second",
                "parent_id": "image",
            },
        ],
    }


def test_serialise_rle_sv_detections_with_bytes_counts() -> None:
    # given - RLE mask with bytes counts (as returned by pycocotools)
    rle_mask = {"size": [192, 168], "counts": b"abc123"}
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.1], dtype=np.float64),
        data={
            "class_name": np.array(["cat"]),
            "detection_id": np.array(["first"]),
            "parent_id": np.array(["image"]),
            "image_dimensions": np.array([[192, 168]]),
            "rle_mask": np.array([rle_mask], dtype=object),
        },
    )

    # when
    result = serialise_rle_sv_detections(detections=detections)

    # then - counts should be converted to string
    assert result["predictions"][0]["rle_mask"] == {
        "size": [192, 168],
        "counts": "abc123",
    }


def test_serialise_rle_sv_detections_does_not_extract_polygons(monkeypatch) -> None:
    # given - both a dense mask and the RLE representation are present
    from inference.core.workflows.core_steps.common import serializers

    detections = sv.Detections(
        xyxy=np.array([[1, 1, 4, 4]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.9], dtype=np.float64),
        mask=np.ones((1, 5, 5), dtype=bool),
        data={
            "class_name": np.array(["cat"]),
            "detection_id": np.array(["first"]),
            "image_dimensions": np.array([[5, 5]]),
            "rle_mask": np.array([{"size": [5, 5], "counts": "encoded"}], dtype=object),
        },
    )
    original_mask = detections.mask

    def fail_on_polygon_extraction(*args, **kwargs):
        pytest.fail("RLE serialization must not extract a polygon")

    monkeypatch.setattr(serializers, "mask_to_polygon", fail_on_polygon_extraction)

    # when
    result = serializers.serialise_rle_sv_detections(detections=detections)

    # then
    assert detections.mask is original_mask
    assert result["predictions"][0]["rle_mask"] == {
        "size": [5, 5],
        "counts": "encoded",
    }
    assert "points" not in result["predictions"][0]


def test_tensor_rle_serializer_does_not_decode_or_extract_polygons(monkeypatch) -> None:
    # given
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.common import serializers_tensor
    from inference_models.models.base.instance_segmentation import InstanceDetections
    from inference_models.models.base.types import InstancesRLEMasks
    from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

    dense_mask = torch.zeros((5, 5), dtype=torch.bool)
    dense_mask[1:4, 1:4] = True
    rle = torch_mask_to_coco_rle(dense_mask)
    detections = InstanceDetections(
        xyxy=torch.tensor([[1.0, 1.0, 4.0, 4.0]]),
        class_id=torch.tensor([1]),
        confidence=torch.tensor([0.9]),
        mask=InstancesRLEMasks(image_size=(5, 5), masks=[rle["counts"]]),
        image_metadata={"class_names": {1: "cat"}, "image_dimensions": [5, 5]},
        bboxes_metadata=[{"detection_id": "first"}],
    )

    def fail_on_mask_decode(*args, **kwargs):
        pytest.fail("RLE serialization must not decode its mask")

    def fail_on_polygon_extraction(*args, **kwargs):
        pytest.fail("RLE serialization must not extract a polygon")

    monkeypatch.setattr(
        serializers_tensor, "coco_rle_masks_to_numpy_mask", fail_on_mask_decode
    )
    monkeypatch.setattr(
        serializers_tensor, "mask_to_polygon", fail_on_polygon_extraction
    )

    # when
    result = serializers_tensor.serialise_native_rle_detections(detections)

    # then
    assert len(result["predictions"]) == 1
    assert "points" not in result["predictions"][0]
    assert result["predictions"][0]["rle_mask"]["size"] == [5, 5]


def test_serialise_rle_sv_detections_raises_when_no_rle_masks() -> None:
    # given - detections without RLE masks
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.1], dtype=np.float64),
        data={
            "class_name": np.array(["cat"]),
            "detection_id": np.array(["first"]),
            "parent_id": np.array(["image"]),
            "image_dimensions": np.array([[192, 168]]),
        },
    )

    # when / then
    try:
        serialise_rle_sv_detections(detections=detections)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "No RLE masks found" in str(e)


def test_serialise_rle_sv_detections_with_parent_origin() -> None:
    # given
    rle_mask = {"size": [192, 168], "counts": "abc123"}
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.1], dtype=np.float64),
        data={
            "class_name": np.array(["cat"]),
            "detection_id": np.array(["first"]),
            "parent_id": np.array(["crop_id"]),
            "root_parent_id": np.array(["original_image"]),
            "parent_coordinates": np.array([[50, 75]]),
            "parent_dimensions": np.array([[600, 800]]),
            "root_parent_coordinates": np.array([[150, 200]]),
            "root_parent_dimensions": np.array([[1080, 1920]]),
            "image_dimensions": np.array([[192, 168]]),
            "rle_mask": np.array([rle_mask], dtype=object),
        },
    )

    # when
    result = serialise_rle_sv_detections(detections=detections)

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
                "rle_mask": {"size": [192, 168], "counts": "abc123"},
                "class": "cat",
                "detection_id": "first",
                "parent_id": "crop_id",
                "parent_origin": {
                    "offset_x": 50,
                    "offset_y": 75,
                    "width": 800,
                    "height": 600,
                },
                "root_parent_id": "original_image",
                "root_parent_origin": {
                    "offset_x": 150,
                    "offset_y": 200,
                    "width": 1920,
                    "height": 1080,
                },
            },
        ],
    }


def test_serialise_native_classification_key_ordering_matches_numpy_path() -> None:
    # given
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.common.serializers_tensor import (
        serialise_native_classification,
    )
    from inference_models.models.base.classification import (
        ClassificationPrediction,
        MultiLabelClassificationPrediction,
    )

    base_metadata = {
        "class_names": {0: "cat", 1: "dog"},
        "prediction_type": "classification",
        "image_dimensions": [480, 640],
        "inference_id": "iid",
        "parent_id": "p1",
        "root_parent_id": "r1",
    }

    def single_label(metadata: dict) -> "ClassificationPrediction":
        return ClassificationPrediction(
            class_id=torch.tensor([1]),
            confidence=torch.tensor([[0.25, 0.5]], dtype=torch.float32),
            images_metadata=[metadata],
        )

    def multi_label(metadata: dict) -> "MultiLabelClassificationPrediction":
        return MultiLabelClassificationPrediction(
            class_ids=torch.tensor([0, 1]),
            confidence=torch.tensor([0.5, 0.25], dtype=torch.float32),
            image_metadata=metadata,
        )

    # when
    single_label_result = serialise_native_classification(
        single_label(dict(base_metadata))
    )
    multi_label_result = serialise_native_classification(
        multi_label(dict(base_metadata))
    )
    single_label_timed_result = serialise_native_classification(
        single_label({**base_metadata, "time": 0.0123})
    )
    multi_label_timed_result = serialise_native_classification(
        multi_label({**base_metadata, "time": 0.0123})
    )

    # then
    # top-level and per-entry key ORDER mirror the numpy flag-off path
    # (response model_dump(by_alias=True, exclude_none=True) + the block's
    # in-place prediction_type/parent_id/root_parent_id writes); orjson
    # byte-parity depends on it. When the producer block stamps `time`
    # (model-call elapsed, as `Model.infer_from_request` does on the numpy
    # path) it must land between `inference_id` and `image`; when absent,
    # the key is omitted entirely.
    assert list(single_label_result.keys()) == [
        "inference_id",
        "image",
        "predictions",
        "top",
        "confidence",
        "prediction_type",
        "parent_id",
        "root_parent_id",
    ]
    assert [list(e.keys()) for e in single_label_result["predictions"]] == [
        ["class", "class_id", "confidence"],
        ["class", "class_id", "confidence"],
    ]
    assert single_label_result["predictions"][0] == {
        "class": "dog",
        "class_id": 1,
        "confidence": 0.5,
    }
    assert list(multi_label_result.keys()) == [
        "inference_id",
        "image",
        "predictions",
        "predicted_classes",
        "prediction_type",
        "parent_id",
        "root_parent_id",
    ]
    assert [list(e.keys()) for e in multi_label_result["predictions"].values()] == [
        ["confidence", "class_id"],
        ["confidence", "class_id"],
    ]
    assert list(single_label_timed_result.keys()) == [
        "inference_id",
        "time",
        "image",
        "predictions",
        "top",
        "confidence",
        "prediction_type",
        "parent_id",
        "root_parent_id",
    ]
    assert single_label_timed_result["time"] == 0.0123
    assert list(multi_label_timed_result.keys()) == [
        "inference_id",
        "time",
        "image",
        "predictions",
        "predicted_classes",
        "prediction_type",
        "parent_id",
        "root_parent_id",
    ]
    assert multi_label_timed_result["time"] == 0.0123


def test_tensor_wildcard_serializer_dispatches_native_values_like_kind_serializers() -> (
    None
):
    # given
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.common import serializers_tensor
    from inference_models.models.base.classification import ClassificationPrediction
    from inference_models.models.base.instance_segmentation import InstanceDetections
    from inference_models.models.base.keypoints_detection import KeyPoints
    from inference_models.models.base.object_detection import Detections
    from inference_models.models.base.types import InstancesRLEMasks
    from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

    od = Detections(
        xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        class_id=torch.tensor([1]),
        confidence=torch.tensor([0.5]),
        image_metadata={"class_names": {1: "dog"}, "image_dimensions": [100, 200]},
        bboxes_metadata=[{"detection_id": "det-1"}],
    )
    dense_mask = torch.zeros((1, 15, 15), dtype=torch.bool)
    dense_mask[0, 2:6, 3:9] = True
    instance_dense = InstanceDetections(
        xyxy=torch.tensor([[3.0, 2.0, 9.0, 6.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=dense_mask,
        image_metadata={"class_names": {0: "cat"}, "image_dimensions": [15, 15]},
        bboxes_metadata=[{"detection_id": "det-2"}],
    )
    rle = torch_mask_to_coco_rle(dense_mask[0])
    instance_rle = InstanceDetections(
        xyxy=torch.tensor([[3.0, 2.0, 9.0, 6.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
        mask=InstancesRLEMasks(image_size=(15, 15), masks=[rle["counts"]]),
        image_metadata={"class_names": {0: "cat"}, "image_dimensions": [15, 15]},
        bboxes_metadata=[{"detection_id": "det-3"}],
    )
    key_points = KeyPoints(
        xy=torch.tensor([[[11.0, 11.0], [12.0, 13.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9, 0.8]]),
    )
    kp_tuple = (key_points, od)
    classification = ClassificationPrediction(
        class_id=torch.tensor([1]),
        confidence=torch.tensor([[0.25, 0.5]], dtype=torch.float32),
        images_metadata=[
            {
                "class_names": {0: "cat", 1: "dog"},
                "prediction_type": "classification",
                "image_dimensions": [480, 640],
                "inference_id": "iid",
            }
        ],
    )
    bare_tensor = torch.tensor([[0.25, 0.5], [0.75, 1.0]])

    # when
    result = serializers_tensor.serialize_wildcard_kind(
        value={
            "od": od,
            "nested": [instance_dense, {"deeper": instance_rle}],
            "kp": kp_tuple,
            "cls": classification,
            "tensor": bare_tensor,
            "untouched": "text",
            "number": 42,
            "none": None,
            "plain_tuple": (1, 2),
        }
    )

    # then - every native arm must equal its kind serialiser's output
    assert result["od"] == serializers_tensor.serialise_sv_detections(od)
    assert result["nested"][0] == serializers_tensor.serialise_sv_detections(
        instance_dense
    )
    assert result["nested"][1]["deeper"] == serializers_tensor.serialise_sv_detections(
        instance_rle
    )
    assert result["kp"] == serializers_tensor.serialise_native_keypoint_detection(
        prediction=kp_tuple
    )
    assert result["cls"] == serializers_tensor.serialise_native_classification(
        prediction=classification
    )
    assert result["tensor"] == [[0.25, 0.5], [0.75, 1.0]]
    assert result["untouched"] == "text"
    assert result["number"] == 42
    assert result["none"] is None
    assert result["plain_tuple"] == (1, 2), "non-KP tuples pass through like numpy"


def test_tensor_wildcard_serializer_matches_numpy_wildcard_for_equivalent_prediction() -> (
    None
):
    # given - the same logical OD prediction as sv (numpy path) and native (tensor path)
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.common import serializers_tensor
    from inference_models.models.base.object_detection import Detections

    sv_detections = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "class_name": np.array(["dog"]),
            "detection_id": np.array(["det-1"]),
            "image_dimensions": np.array([[100, 200]]),
        },
    )
    native_detections = Detections(
        xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        class_id=torch.tensor([1]),
        confidence=torch.tensor([0.5]),
        image_metadata={"class_names": {1: "dog"}, "image_dimensions": [100, 200]},
        bboxes_metadata=[{"detection_id": "det-1"}],
    )

    # when
    numpy_result = serialize_wildcard_kind(value={"predictions": sv_detections})
    tensor_result = serializers_tensor.serialize_wildcard_kind(
        value={"predictions": native_detections}
    )

    # then
    assert numpy_result == tensor_result


def test_tensor_wildcard_serializer_keeps_numpy_behavior_for_legacy_values() -> None:
    # given - sv.Detections + datetime + image reaching the tensor wildcard
    pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.common import serializers_tensor
    from inference.core.workflows.execution_engine.entities.base import VideoMetadata

    sv_detections = sv.Detections(
        xyxy=np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float64),
        class_id=np.array([1]),
        confidence=np.array([0.5], dtype=np.float64),
        data={
            "class_name": np.array(["dog"]),
            "detection_id": np.array(["det-1"]),
        },
    )
    timestamp = datetime.now()
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="origin"),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
        # explicit metadata: without it, serialisation mints frame_timestamp
        # per call, breaking the two-call comparison below
        video_metadata=VideoMetadata(
            video_identifier="vid",
            frame_number=0,
            frame_timestamp=timestamp,
        ),
    )

    # when
    result = serializers_tensor.serialize_wildcard_kind(
        value={"sv": sv_detections, "ts": timestamp, "img": image}
    )

    # then - byte-for-byte the numpy wildcard behavior for legacy values
    expected = serialize_wildcard_kind(
        value={"sv": sv_detections, "ts": timestamp, "img": image}
    )
    assert result == expected
