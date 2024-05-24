from copy import deepcopy

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.utils import (
    add_inference_keypoints_to_sv_detections,
    attach_parents_coordinates_to_sv_detections,
    attach_prediction_type_info,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    sv_detections_to_root_coordinates, filter_out_unwanted_classes_from_sv_detections_batch,
)
from inference.core.workflows.entities.base import (
    OriginCoordinatesSystem,
    ParentImageMetadata,
    WorkflowImageData,
)


def test_attach_prediction_type_info_for_non_empty_predictions() -> None:
    # given
    predictions = [
        {"top": "car", "confidence": 0.3},
        {"top": "bike", "confidence": 0.7},
    ]

    # when
    result = attach_prediction_type_info(
        predictions=predictions,
        prediction_type="classification",
    )

    # then
    assert result == [
        {"top": "car", "confidence": 0.3, "prediction_type": "classification"},
        {"top": "bike", "confidence": 0.7, "prediction_type": "classification"},
    ]


def test_attach_prediction_type_info_for_empty_predictions() -> None:
    # given
    predictions = []

    # when
    result = attach_prediction_type_info(
        predictions=predictions,
        prediction_type="classification",
    )

    # then
    assert result == []


def test_attach_prediction_type_info_to_sv_detections_batch_when_batch_is_not_empty() -> (
    None
):
    # given
    empty_detection = sv.Detections.empty()
    empty_detection["class_name"] = np.array([])
    predictions = [
        sv.Detections(
            xyxy=np.array([[0, 10, 10, 100], [0, 20, 20, 100]]),
            mask=None,
            confidence=np.array([0.3, 0.5]),
            class_id=np.array([0, 1]),
            tracker_id=None,
            data={"class_name": np.array(["cat", "dog"])},
        ),
        empty_detection,
    ]

    # when
    result = attach_prediction_type_info_to_sv_detections_batch(
        predictions=predictions,
        prediction_type="object-detection",
    )

    # then
    assert len(result) == 2, "Expected number of batch element not to change"
    assert result[0] is predictions[0], "Function expected to operate in-place"
    assert result[1] is predictions[1], "Function expected to operate in-place"
    assert np.allclose(
        result[0].xyxy, np.array([[0, 10, 10, 100], [0, 20, 20, 100]])
    ), "Expected xyxy not to be modified"
    assert result[0].mask is None, "Expected mask not to be modified"
    assert np.allclose(
        result[0].confidence, np.array([0.3, 0.5])
    ), "Expected confidence not to be modified"
    assert np.allclose(
        result[0].class_id, np.array([0, 1])
    ), "Expected class_id not to be modified"
    assert result[0].tracker_id is None, "Expected tracker_id not to be modified"
    assert (
        result[0].data["class_name"] == np.array(["cat", "dog"])
    ).all(), "Expected class_name not to be modified"
    assert (
        result[0].data["prediction_type"]
        == np.array(["object-detection", "object-detection"])
    ).all(), "Expected prediction_type to be added as object-detection for each element"
    expected_empty_detection = sv.Detections.empty()
    expected_empty_detection["class_name"] = np.array([])
    expected_empty_detection["prediction_type"] = np.array([])
    assert (
        result[1] == expected_empty_detection
    ), "Expected empty detections not to be modified"


def test_attach_prediction_type_info_to_sv_detections_batch_when_batch_empty() -> None:
    # given
    predictions = []

    # when
    result = attach_prediction_type_info_to_sv_detections_batch(
        predictions=predictions,
        prediction_type="object-detection",
    )

    # then
    assert result == []


def test_convert_inference_detections_batch_to_sv_detections() -> None:
    # given
    predictions = [
        {
            "image": {"height": 200, "width": 100},
            "predictions": [
                {
                    "width": 50,
                    "height": 100,
                    "x": 50,
                    "y": 100,
                    "confidence": 0.1,
                    "class_id": 1,
                    "points": [
                        {"x": 30, "y": 80},
                        {"x": 30, "y": 120},
                        {"x": 70, "y": 120},
                        {"x": 70, "y": 80},
                    ],
                    "tracker_id": 1,
                    "class": "dog",
                    "detection_id": "first",
                    "parent_id": "image",
                },
                {
                    "width": 50,
                    "height": 100,
                    "x": 75,
                    "y": 175,
                    "confidence": 0.2,
                    "class_id": 0,
                    "points": [
                        {"x": 90, "y": 170},
                        {"x": 90, "y": 190},
                        {"x": 70, "y": 190},
                        {"x": 70, "y": 170},
                    ],
                    "tracker_id": 2,
                    "class": "cat",
                    "detection_id": "second",
                    "parent_id": "image",
                },
            ],
        }
    ]

    # when
    result = convert_inference_detections_batch_to_sv_detections(
        predictions=predictions,
    )

    # then
    assert (
        len(result) == 1
    ), "Expected only single element in output batch, as input batch was of size one"
    expected_mask = np.zeros((2, 200, 100), dtype=np.bool_)
    expected_mask[0, 80:121, 30:71] = True
    expected_mask[1, 170:191, 70:91] = True
    assert np.allclose(result[0].mask, expected_mask)
    assert result[0] == sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=expected_mask,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
        },
    )


def test_add_inference_keypoints_to_sv_detections() -> None:
    # given
    mask = np.zeros((2, 200, 100), dtype=np.bool_)
    mask[0, 80:121, 30:71] = True
    mask[1, 170:191, 70:91] = True
    expected_mask = mask.copy()
    detections = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=mask,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
        },
    )
    inference_prediction = [
        {
            "keypoints": [
                {"x": 10, "y": 20, "class": "a", "confidence": 0.3, "class_id": 1},
                {"x": 20, "y": 30, "class": "b", "confidence": 0.4, "class_id": 0},
            ]
        },
        {
            "keypoints": [],
        },
    ]

    # when
    result = add_inference_keypoints_to_sv_detections(
        detections=detections,
        inference_prediction=inference_prediction,
    )

    # then
    assert result is detections, "Operation is expected to be performed in-place"
    assert np.allclose(
        result.xyxy, np.array([[25, 50, 75, 150], [50, 125, 100, 225]])
    ), "Expected coordinates not to be touched"
    assert np.allclose(result.mask, expected_mask), "Expected mask not to be touched"
    assert np.allclose(
        result.confidence, np.array([0.1, 0.2])
    ), "Expected confidence not to be touched"
    assert np.allclose(
        result.class_id, np.array([1, 0])
    ), "Expected class_id not to be touched"
    assert np.allclose(
        result.tracker_id, np.array([1, 2])
    ), "Expected tracker_id not to be touched"
    assert (
        result["class_name"] == np.array(["dog", "cat"])
    ).all(), "Expected class_name not to be touched"
    assert (
        result["detection_id"] == np.array(["first", "second"])
    ).all(), "Expected detection_id not to be touched"
    assert (
        result["parent_id"] == np.array(["image", "image"])
    ).all(), "Expected detection_id not to be touched"
    assert (
        result["keypoints_class_name"][0] == np.array(["a", "b"])
    ).all(), "There are two keypoints for first object, with classes a and b"
    assert (
        result["keypoints_class_name"][1] == np.array([])
    ).all(), "There are no keypoints for second object"
    assert (
        result["keypoints_class_id"][0] == np.array([1, 0])
    ).all(), "There are two keypoints for first object, with ids 1 and 0"
    assert (
        result["keypoints_class_id"][1] == np.array([])
    ).all(), "There are no keypoints for second object"
    assert (
        result["keypoints_confidence"][0] == np.array([0.3, 0.4], dtype=np.float32)
    ).all(), "There are two keypoints for first object, with confidences 0.3 and 0.4"
    assert (
        result["keypoints_confidence"][1] == np.array([])
    ).all(), "There are no keypoints for second object"
    assert (
        result["keypoints_xy"][0] == np.array([[10, 20], [20, 30]])
    ).all(), "There are two keypoints for first object, with specific coordinates"
    assert (
        result["keypoints_xy"][1] == np.array([])
    ).all(), "There are no keypoints for second object"


def test_add_inference_keypoints_to_sv_detections_when_mismatched_data_provided() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=None,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
        },
    )
    inference_prediction = []

    # when
    with pytest.raises(ValueError):
        _ = add_inference_keypoints_to_sv_detections(
            inference_prediction=inference_prediction,
            detections=detections,
        )


def test_attach_parents_coordinates_to_sv_detections() -> None:
    # given
    mask = np.zeros((2, 200, 100), dtype=np.bool_)
    mask[0, 80:121, 30:71] = True
    mask[1, 170:191, 70:91] = True
    expected_mask = mask.copy()
    detections = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=mask,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
        },
    )
    image = WorkflowImageData(
        parent_metadata=ParentImageMetadata(
            parent_id="crop_1",
        ),
        workflow_root_ancestor_metadata=ParentImageMetadata(
            parent_id="image",
            origin_coordinates=OriginCoordinatesSystem(
                left_top_x=50,
                left_top_y=100,
                origin_width=512,
                origin_height=1024,
            ),
        ),
        numpy_image=np.zeros((200, 100, 3), dtype=np.uint8),
    )

    # when
    result = attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[25, 50, 75, 150], [50, 125, 100, 225]])
    ), "Expected coordinates not to be touched"
    assert np.allclose(result.mask, expected_mask), "Expected mask not to be touched"
    assert np.allclose(
        result.confidence, np.array([0.1, 0.2])
    ), "Expected confidence not to be touched"
    assert np.allclose(
        result.class_id, np.array([1, 0])
    ), "Expected class_id not to be touched"
    assert np.allclose(
        result.tracker_id, np.array([1, 2])
    ), "Expected tracker_id not to be touched"
    assert (
        result["class_name"] == np.array(["dog", "cat"])
    ).all(), "Expected class_name not to be touched"
    assert (
        result["detection_id"] == np.array(["first", "second"])
    ).all(), "Expected detection_id not to be touched"
    assert (
        result["parent_id"] == np.array(["crop_1", "crop_1"])
    ).all(), "Expected parent_id to point into crop_1"
    assert (
        result["parent_coordinates"] == np.array([[0, 0], [0, 0]])
    ).all(), "Detection not shifted compared to parent, hence [0, 0] is shift of coordinates system"
    assert (
        result["parent_dimensions"] == np.array([[200, 100], [200, 100]])
    ).all(), "Expected image size to be denoted"
    assert (
        result["root_parent_id"] == np.array(["image", "image"])
    ).all(), "Expected parent_id to point into crop_1"
    assert (
        result["root_parent_coordinates"] == np.array([[50, 100], [50, 100]])
    ).all(), "Detection shifted compared to root, hence [50, 100] is shift of coordinates system"
    assert (
        result["root_parent_dimensions"] == np.array([[1024, 512], [1024, 512]])
    ).all(), "Expected root size to be denoted"


def test_sv_detections_to_root_coordinates_when_empty_detections_passed() -> None:
    # given
    detections = sv.Detections.empty()

    # when
    result = sv_detections_to_root_coordinates(detections=detections)

    # then
    assert (
        result == sv.Detections.empty()
    ), "Expected empty detections not to be modified"


def test_sv_detections_to_root_coordinates_when_detections_without_root_coordinates_provided() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=None,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
        },
    )
    expected_result = deepcopy(detections)

    # when
    result = sv_detections_to_root_coordinates(detections=detections)

    # then
    assert (
        result == expected_result
    ), "Expected detections not to be mutated when root metadata not provided"


def test_sv_detections_to_root_coordinates_when_shift_is_needed() -> None:
    # given
    mask = np.zeros((2, 200, 100), dtype=np.bool_)
    mask[0, 80:121, 30:71] = True
    mask[1, 170:191, 70:91] = True
    expected_mask = np.zeros((2, 1024, 512), dtype=np.bool_)
    expected_mask[:, 100:300, 50:150] = mask
    detections = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=mask,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["crop_1", "crop_1"]),
            "parent_coordinates": np.array([[10, 20], [10, 20]]),
            "parent_dimensions": np.array([[200, 100], [200, 100]]),
            "root_parent_id": np.array(["root", "root"]),
            "root_parent_coordinates": np.array([[50, 100], [50, 100]]),
            "root_parent_dimensions": np.array([[1024, 512], [1024, 512]]),
            "keypoints_class_name": np.array(
                [np.array(["a", "b"]), np.array([])], dtype="object"
            ),
            "keypoints_class_id": np.array(
                [np.array([1, 0]), np.array([])], dtype="object"
            ),
            "keypoints_confidence": np.array(
                [np.array([0.3, 0.4]), np.array([])], dtype="object"
            ),
            "keypoints_xy": np.array(
                [np.array([[10, 20], [20, 30]]), np.array([])], dtype="object"
            ),
        },
    )

    # when
    result = sv_detections_to_root_coordinates(
        detections=detections,
    )

    # then
    assert np.allclose(
        result.xyxy,
        np.array(
            [
                [50 + 25, 100 + 50, 50 + 75, 100 + 150],
                [50 + 50, 100 + 125, 50 + 100, 100 + 225],
            ]
        ),
    ), "Expected coordinates to be shifted into root coordinates (by [50, 100])"
    assert np.allclose(
        result.mask, expected_mask
    ), "Expected mask to be properly shifted"
    assert np.allclose(
        result.confidence, np.array([0.1, 0.2])
    ), "Expected confidence not to be touched"
    assert np.allclose(
        result.class_id, np.array([1, 0])
    ), "Expected class_id not to be touched"
    assert np.allclose(
        result.tracker_id, np.array([1, 2])
    ), "Expected tracker_id not to be touched"
    assert (
        result["class_name"] == np.array(["dog", "cat"])
    ).all(), "Expected class_name not to be touched"
    assert (
        result["detection_id"] == np.array(["first", "second"])
    ).all(), "Expected detection_id not to be touched"
    assert (
        result["parent_id"] == np.array(["root", "root"])
    ).all(), "root becomes parent, hence we expect it to be marked with parent id"
    assert (
        result["parent_coordinates"] == np.array([[0, 0], [0, 0]])
    ).all(), "root becomes parent, we shifted detection, hence parent coordinates starts in [0, 0]"
    assert (
        result["parent_dimensions"] == np.array([[1024, 512], [1024, 512]])
    ).all(), (
        "root becomes parent, we shifted detection, hence dimensions are [1024, 512]"
    )
    assert (
        result["root_parent_id"] == np.array(["root", "root"])
    ).all(), (
        "root stays root parent, hence we expect it to be marked with root_parent_id"
    )
    assert (
        result["root_parent_coordinates"] == np.array([[0, 0], [0, 0]])
    ).all(), "We shifted predictions"
    assert (
        result["root_parent_dimensions"] == np.array([[1024, 512], [1024, 512]])
    ).all(), "Expected root size to be denoted"


def test_filter_out_unwanted_classes_from_sv_detections_batch_when_no_classes_defined() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=None,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
        },
    )
    expected_result = deepcopy(detections)

    # when
    result = filter_out_unwanted_classes_from_sv_detections_batch(
        predictions=[detections],
        classes_to_accept=None,
    )

    # then
    assert len(result) == 1, "Expected batch dimension not to change"
    assert result[0] == expected_result, "We expect nothing to be filtered out"
    assert result[0] is detections, "We expect operation to be in-place"


def test_filter_out_unwanted_classes_from_sv_detections_batch_when_empty_class_list_defined() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=None,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
        },
    )
    expected_result = deepcopy(detections)

    # when
    result = filter_out_unwanted_classes_from_sv_detections_batch(
        predictions=[detections],
        classes_to_accept=[],
    )

    # then
    assert len(result) == 1, "Expected batch dimension not to change"
    assert result[0] == expected_result, "We expect nothing to be filtered out"
    assert result[0] is detections, "We expect operation to be in-place"


def test_filter_out_unwanted_classes_from_sv_detections_batch_when_filtering_should_be_applied() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150], [50, 125, 100, 225]]),
        mask=None,
        confidence=np.array([0.1, 0.2]),
        class_id=np.array([1, 0]),
        tracker_id=np.array([1, 2]),
        data={
            "class_name": np.array(["dog", "cat"]),
            "detection_id": np.array(["first", "second"]),
            "parent_id": np.array(["image", "image"]),
        },
    )
    expected_result = sv.Detections(
        xyxy=np.array([[25, 50, 75, 150]]),
        mask=None,
        confidence=np.array([0.1]),
        class_id=np.array([1]),
        tracker_id=np.array([1]),
        data={
            "class_name": np.array(["dog"]),
            "detection_id": np.array(["first"]),
            "parent_id": np.array(["image"]),
        },
    )

    # when
    result = filter_out_unwanted_classes_from_sv_detections_batch(
        predictions=[detections],
        classes_to_accept=["dog"],
    )

    # then
    assert len(result) == 1, "Expected batch dimension not to change"
    assert result[0] == expected_result, "We expect result to be filtered"
