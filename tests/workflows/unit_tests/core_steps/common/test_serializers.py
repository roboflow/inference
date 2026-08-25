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


def test_serialise_sv_detections_skips_padded_keypoint_slots() -> None:
    # given the padded, rectangular (n, max_kps, 2) keypoint layout that
    # add_inference_keypoints_to_sv_detections produces when detections carry
    # unequal numbers of keypoints (e.g. after keypoint-confidence filtering or
    # multi-class skeletons). Detection 0 has 2 real keypoints, detection 1 has
    # only 1; the remaining slot of detection 1 is empty-named padding.
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64),
        class_id=np.array([0, 0]),
        confidence=np.array([0.9, 0.8], dtype=np.float64),
        data={
            "class_name": np.array(["obj", "obj"]),
            "detection_id": np.array(["first", "second"]),
            "keypoints_xy": np.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[21.0, 22.0], [0.0, 0.0]]],
                dtype=np.float32,
            ),
            "keypoints_confidence": np.array(
                [[0.9, 0.8], [0.95, 0.0]], dtype=np.float32
            ),
            "keypoints_class_id": np.array([[0, 1], [0, 0]], dtype=int),
            "keypoints_class_name": np.array(
                [["nose", "eye"], ["nose", ""]], dtype=object
            ),
        },
    )

    # when
    result = serialise_sv_detections(detections=detections)

    # then the padding slot must not surface as a fabricated keypoint
    assert len(result["predictions"][0]["keypoints"]) == 2
    assert len(result["predictions"][1]["keypoints"]) == 1
    assert result["predictions"][1]["keypoints"][0]["class"] == "nose"
    for prediction in result["predictions"]:
        for keypoint in prediction["keypoints"]:
            assert keypoint["class"] != "", "No empty-named padding keypoint may leak"


def test_serialise_sv_detections_with_nearest_target_distance() -> None:
    # given: one detection with a real match distance, one unmatched (None)
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "nearest_target_distance": np.array([12.5, None], dtype=object),
        },
    )

    # when
    result = serialise_sv_detections(detections=detections)

    # then
    predictions = result["predictions"]
    assert predictions[0]["nearest_target_distance"] == 12.5
    assert isinstance(predictions[0]["nearest_target_distance"], float)
    assert predictions[1]["nearest_target_distance"] is None


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
    # given — small speck first in image order, large instance second
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:14, 10:14] = 255  # 4x4 speck
    mask[40:80, 50:100] = 255  # large instance

    # when
    result = mask_to_polygon(mask=mask)

    # then — must prefer largest contour, not findContours order (Dataset Upload bug)
    xs, ys = result[:, 0], result[:, 1]
    assert xs.min() >= 50 and xs.max() <= 99
    assert ys.min() >= 40 and ys.max() <= 79
    assert (xs.max() - xs.min()) >= 40
    assert (ys.max() - ys.min()) >= 30


def test_mask_to_polygon_when_mask_contains_hole() -> None:
    # given
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:110, 20:100] = 255
    mask[20:100, 30:90] = 0

    # when
    result = mask_to_polygon(mask=mask)

    # then — RETR_TREE returns both contours; the exterior must be selected
    xs, ys = result[:, 0], result[:, 1]
    assert xs.min() == 20
    assert xs.max() == 99
    assert ys.min() == 10
    assert ys.max() == 109


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


def test_serialise_sv_detections_prefers_largest_contour_after_downscale() -> None:
    """Regression for Dataset Upload: after 4096→2080 scale, masks can contain a
    tiny speck + the real instance. Serialising contours[0] produced annotations
    that looked cut away at the edge (tiny polygon, full-size bbox)."""
    from inference.core.workflows.core_steps.common.utils import scale_sv_detections

    orig = 4096
    target = 2080
    scale = target / orig
    mask = np.zeros((orig, orig), dtype=bool)
    mask[400:1600, 3100:orig] = True
    for y in range(400, 1600):
        jag = int(80 * np.sin(y / 40.0) + 40)
        mask[y, 3100 : 3100 + jag] = False
    mask[820:828, 3088:3096] = True

    detections = sv.Detections(
        xyxy=np.array([[3088, 400, orig - 1, 1600]], dtype=np.float64),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={
            "class_name": np.array(["sill"]),
            "detection_id": np.array(["d1"]),
            "image_dimensions": np.array([[orig, orig]]),
        },
    )
    scaled = scale_sv_detections(
        detections=detections,
        scale=(scale, scale),
        target_size_wh=(target, target),
    )

    result = serialise_sv_detections(detections=scaled)

    assert len(result["predictions"]) == 1
    pred = result["predictions"][0]
    points = pred["points"]
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    poly_w = max(xs) - min(xs)
    poly_h = max(ys) - min(ys)
    # Must not collapse to a ~4x8 speck while bbox stays ~500x600
    assert poly_w > pred["width"] * 0.5
    assert poly_h > pred["height"] * 0.5
    assert max(xs) >= 2070  # still reaches the right edge
    assert result["image"] == {"width": target, "height": target}


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


def test_serialise_rle_sv_detections_preserves_detection_with_empty_dense_mask() -> (
    None
):
    # given
    masks = np.zeros((2, 4, 4), dtype=bool)
    masks[1, 0:2, 0:2] = True
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [0, 0, 2, 2]], dtype=np.float64),
        mask=masks,
        class_id=np.array([1, 2]),
        confidence=np.array([0.1, 0.9], dtype=np.float64),
        data={
            "class_name": np.array(["empty", "body"]),
            "detection_id": np.array(["empty-id", "body-id"]),
            "image_dimensions": np.array([[4, 4], [4, 4]]),
            "rle_mask": np.array(
                [
                    {"size": [4, 4], "counts": "a"},
                    {"size": [4, 4], "counts": "b"},
                ],
                dtype=object,
            ),
        },
    )

    # when
    result = serialise_rle_sv_detections(detections=detections)

    # then
    assert [prediction["detection_id"] for prediction in result["predictions"]] == [
        "empty-id",
        "body-id",
    ]
    assert [
        prediction["rle_mask"]["counts"] for prediction in result["predictions"]
    ] == [
        "a",
        "b",
    ]
    assert np.array_equal(detections.mask, masks)


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

    # then - exact key order matters: orjson byte-parity with the numpy path depends on it
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

    # then
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
    assert result["plain_tuple"] == (
        1,
        2,
    ), "non-KP tuples are rebuilt element-wise with values intact"


def test_tensor_wildcard_serializer_converts_tuples_element_wise() -> None:
    # given - tuples must not smuggle a live tensor across the stream-manager
    # process boundary (CUDA IPC is unsupported on Jetson/Tegra), so the
    # wildcard serialiser converts tuple elements like list elements while
    # preserving the container type (namedtuples rebuilt field-wise).
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from collections import namedtuple

    from inference.core.workflows.core_steps.common import serializers_tensor
    from inference_models.models.base.keypoints_detection import KeyPoints
    from inference_models.models.base.object_detection import Detections

    def assert_no_torch_tensor(value) -> None:
        assert not isinstance(value, torch.Tensor), f"live tensor survived: {value!r}"
        if isinstance(value, dict):
            for item in value.values():
                assert_no_torch_tensor(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                assert_no_torch_tensor(item)

    TensorRecord = namedtuple("TensorRecord", ["label", "payload"])
    plain_tuple = (torch.tensor([1.0, 2.0]), torch.tensor([[3.0]]))
    named = TensorRecord(label="foo", payload=torch.tensor([4.0, 5.0]))
    nested = [{"inner": (torch.tensor([6.0]), "text", 7)}]
    scalars = ("a", 1, 2.5, None)
    key_points = KeyPoints(
        xy=torch.tensor([[[11.0, 11.0], [12.0, 13.0]]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([[0.9, 0.8]]),
    )
    kp_tuple = (
        key_points,
        Detections(
            xyxy=torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
            class_id=torch.tensor([1]),
            confidence=torch.tensor([0.5]),
            image_metadata={"class_names": {1: "dog"}, "image_dimensions": [100, 200]},
            bboxes_metadata=[{"detection_id": "det-1"}],
        ),
    )

    # when
    plain_result = serializers_tensor.serialize_wildcard_kind(value=plain_tuple)
    named_result = serializers_tensor.serialize_wildcard_kind(value=named)
    nested_result = serializers_tensor.serialize_wildcard_kind(value=nested)
    scalars_result = serializers_tensor.serialize_wildcard_kind(value=scalars)
    kp_result = serializers_tensor.serialize_wildcard_kind(value=kp_tuple)

    # then
    assert type(plain_result) is tuple
    assert plain_result == ([1.0, 2.0], [[3.0]])
    assert_no_torch_tensor(plain_result)
    assert type(named_result) is TensorRecord, "namedtuple type is preserved"
    assert named_result.label == "foo"
    assert named_result.payload == [4.0, 5.0]
    assert_no_torch_tensor(named_result)
    assert nested_result == [{"inner": ([6.0], "text", 7)}]
    assert type(nested_result[0]["inner"]) is tuple
    assert_no_torch_tensor(nested_result)
    assert scalars_result == ("a", 1, 2.5, None), "scalar tuples keep their values"
    assert kp_result == serializers_tensor.serialise_native_keypoint_detection(
        prediction=kp_tuple
    ), "the keypoint pair still routes to the kind serialiser, not element-wise"


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

    # then
    expected = serialize_wildcard_kind(
        value={"sv": sv_detections, "ts": timestamp, "img": image}
    )
    assert result == expected


def test_tensor_serialise_sv_detections_skips_padded_keypoint_slots() -> None:
    # given the padded per-box keypoint rows the sv -> native conversion carries
    # (detection 0 has 2 real keypoints, detection 1 has 1 real + 1 padding slot)
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.common import serializers_tensor
    from inference_models.models.base.object_detection import Detections

    native_detections = Detections(
        xyxy=torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        class_id=torch.tensor([0, 0]),
        confidence=torch.tensor([0.5, 0.25]),
        image_metadata={"class_names": {0: "obj"}, "image_dimensions": [192, 168]},
        bboxes_metadata=[
            {
                "detection_id": "first",
                "keypoints_class_id": np.array([0, 1], dtype=int),
                "keypoints_class_name": np.array(["nose", "eye"], dtype=object),
                "keypoints_confidence": np.array([0.5, 0.25], dtype=np.float32),
                "keypoints_xy": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            },
            {
                "detection_id": "second",
                "keypoints_class_id": np.array([0, 0], dtype=int),
                "keypoints_class_name": np.array(["nose", ""], dtype=object),
                "keypoints_confidence": np.array([0.75, 0.0], dtype=np.float32),
                "keypoints_xy": np.array([[21.0, 22.0], [0.0, 0.0]], dtype=np.float32),
            },
        ],
    )
    sv_detections = sv.Detections(
        xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float64),
        class_id=np.array([0, 0]),
        confidence=np.array([0.5, 0.25], dtype=np.float64),
        data={
            "class_name": np.array(["obj", "obj"]),
            "detection_id": np.array(["first", "second"]),
            "image_dimensions": np.array([[192, 168], [192, 168]]),
            "keypoints_xy": np.array(
                [[[1.0, 2.0], [3.0, 4.0]], [[21.0, 22.0], [0.0, 0.0]]],
                dtype=np.float32,
            ),
            "keypoints_confidence": np.array(
                [[0.5, 0.25], [0.75, 0.0]], dtype=np.float32
            ),
            "keypoints_class_id": np.array([[0, 1], [0, 0]], dtype=int),
            "keypoints_class_name": np.array(
                [["nose", "eye"], ["nose", ""]], dtype=object
            ),
        },
    )

    # when
    result = serializers_tensor.serialise_sv_detections(native_detections)

    # then the padding slot must not surface as a fabricated keypoint
    assert len(result["predictions"][0]["keypoints"]) == 2
    assert len(result["predictions"][1]["keypoints"]) == 1
    assert result["predictions"][1]["keypoints"][0]["class"] == "nose"
    for prediction in result["predictions"]:
        for keypoint in prediction["keypoints"]:
            assert keypoint["class"] != "", "No empty-named padding keypoint may leak"
    assert result == serialise_sv_detections(detections=sv_detections)


def test_tensor_serialise_sv_detections_with_nearest_target_distance() -> None:
    # given: one detection with a real match distance, one unmatched (None)
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference_models")
    from inference.core.workflows.core_steps.common import serializers_tensor
    from inference_models.models.base.object_detection import Detections

    native_detections = Detections(
        xyxy=torch.tensor([[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]),
        class_id=torch.tensor([1, 2]),
        confidence=torch.tensor([0.25, 0.5]),
        image_metadata={"class_names": {1: "cat", 2: "dog"}},
        bboxes_metadata=[
            {"detection_id": "first", "nearest_target_distance": 12.5},
            {"detection_id": "second", "nearest_target_distance": None},
        ],
    )
    sv_detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [3, 3, 4, 4]], dtype=np.float64),
        class_id=np.array([1, 2]),
        confidence=np.array([0.25, 0.5], dtype=np.float64),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["first", "second"]),
            "nearest_target_distance": np.array([12.5, None], dtype=object),
        },
    )

    # when
    result = serializers_tensor.serialise_sv_detections(native_detections)

    # then
    predictions = result["predictions"]
    assert predictions[0]["nearest_target_distance"] == 12.5
    assert isinstance(predictions[0]["nearest_target_distance"], float)
    assert predictions[1]["nearest_target_distance"] is None
    assert result == serialise_sv_detections(detections=sv_detections)
