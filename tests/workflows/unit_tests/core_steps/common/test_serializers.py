import base64
from datetime import datetime

import cv2
import numpy as np
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
