from copy import deepcopy

import cv2
import numpy as np
import pytest
import supervision as sv
from supervision.config import ORIENTED_BOX_COORDINATES
from pycocotools import mask as mask_utils

from inference.core.workflows.core_steps.common.serializers import (
    serialise_rle_sv_detections,
)
from inference.core.workflows.core_steps.common.utils import (
    DETECTION_MAX_EDGE_PIXELS,
    add_inference_keypoints_to_sv_detections,
    attach_parents_coordinates_to_sv_detections,
    attach_prediction_type_info,
    attach_prediction_type_info_to_sv_detections_batch,
    convert_inference_detections_batch_to_sv_detections,
    filter_out_unwanted_classes_from_sv_detections_batch,
    grab_batch_parameters,
    grab_non_batch_parameters,
    remove_unexpected_keys_from_dictionary,
    scale_dimensions_to_max_edge,
    scale_sv_detections,
    sv_detections_to_root_coordinates,
)
from inference.core.workflows.execution_engine.constants import (
    POLYGON_KEY_IN_SV_DETECTIONS,
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
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
    assert isinstance(result[0].mask, np.ndarray)
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
            "image_dimensions": np.array([[200, 100], [200, 100]]),
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
        result["keypoints_class_name"][1] == np.array(["", ""])
    ).all(), "No keypoints for second object, padded with empty strings"
    assert (
        result["keypoints_class_id"][0] == np.array([1, 0])
    ).all(), "There are two keypoints for first object, with ids 1 and 0"
    assert (
        result["keypoints_class_id"][1] == np.array([0, 0])
    ).all(), "No keypoints for second object, padded with zeros"
    assert (
        result["keypoints_confidence"][0] == np.array([0.3, 0.4], dtype=np.float32)
    ).all(), "There are two keypoints for first object, with confidences 0.3 and 0.4"
    assert (
        result["keypoints_confidence"][1] == np.array([0.0, 0.0], dtype=np.float32)
    ).all(), "No keypoints for second object, padded with zeros"
    assert (
        result["keypoints_xy"][0] == np.array([[10, 20], [20, 30]])
    ).all(), "There are two keypoints for first object, with specific coordinates"
    assert (
        result["keypoints_xy"][1] == np.array([[0, 0], [0, 0]])
    ).all(), "No keypoints for second object, padded with zeros"


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
        parent_metadata=ImageParentMetadata(
            parent_id="crop_1",
        ),
        workflow_root_ancestor_metadata=ImageParentMetadata(
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
            POLYGON_KEY_IN_SV_DETECTIONS: np.array(
                [
                    [[25, 50], [75, 50], [75, 150], [25, 150]],
                    [[50, 125], [100, 125], [100, 225], [50, 225]],
                ]
            ),
            ORIENTED_BOX_COORDINATES: np.array(
                [
                    [[25.0, 50.0], [75.0, 50.0], [75.0, 150.0], [25.0, 150.0]],
                    [[50.0, 125.0], [100.0, 125.0], [100.0, 225.0], [50.0, 225.0]],
                ]
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
    assert (
        result["keypoints_class_name"][0] == np.array(["a", "b"])
    ).all(), "Expected keypoints classes not to be touched"
    assert (
        result["keypoints_class_name"][1] == np.array([])
    ).all(), "Expected keypoints classes not to be touched"
    assert (
        result["keypoints_confidence"][0] == np.array([0.3, 0.4])
    ).all(), "Expected keypoints confidence not to be touched"
    assert (
        result["keypoints_confidence"][1] == np.array([])
    ).all(), "Expected keypoints confidence not to be touched"
    assert (
        result["keypoints_xy"][0]
        == np.array([[50 + 10, 100 + 20], [50 + 20, 100 + 30]])
    ).all(), "Expected keypoints xy to be shifted"
    assert (
        result["keypoints_xy"][1] == np.array([])
    ).all(), "Expected empty keypoints xy to be left as is"
    assert np.allclose(
        result[POLYGON_KEY_IN_SV_DETECTIONS],
        np.array(
            [
                [
                    [50 + 25, 100 + 50],
                    [50 + 75, 100 + 50],
                    [50 + 75, 100 + 150],
                    [50 + 25, 100 + 150],
                ],
                [
                    [50 + 50, 100 + 125],
                    [50 + 100, 100 + 125],
                    [50 + 100, 100 + 225],
                    [50 + 50, 100 + 225],
                ],
            ]
        ),
    ), "Expected polygon metadata to be shifted into root coordinates"
    assert np.allclose(
        result[ORIENTED_BOX_COORDINATES],
        np.array(
            [
                [
                    [50 + 25.0, 100 + 50.0],
                    [50 + 75.0, 100 + 50.0],
                    [50 + 75.0, 100 + 150.0],
                    [50 + 25.0, 100 + 150.0],
                ],
                [
                    [50 + 50.0, 100 + 125.0],
                    [50 + 100.0, 100 + 125.0],
                    [50 + 100.0, 100 + 225.0],
                    [50 + 50.0, 100 + 225.0],
                ],
            ]
        ),
    ), "Expected oriented-box corners to be shifted into root coordinates"


def test_sv_detections_to_root_coordinates_when_scale_and_shift_is_needed() -> None:
    # given
    mask = np.zeros((2, 200, 100), dtype=np.bool_)
    mask[0, 80:121, 30:71] = True
    mask[1, 170:191, 70:91] = True
    scaled_mask = np.array(
        [
            cv2.resize(
                detection_mask.astype(np.uint8),
                (200, 400),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for detection_mask in mask
        ]
    )
    expected_mask = np.zeros((2, 1024, 512), dtype=np.bool_)
    expected_mask[:, 100:500, 50:250] = scaled_mask
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
            "scaling_relative_to_root_parent": np.array([0.5, 0.5]),
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
            "image_dimensions": np.array([[200, 100], [200, 100]]),
            POLYGON_KEY_IN_SV_DETECTIONS: np.array(
                [
                    [[25, 50], [75, 50], [75, 150], [25, 150]],
                    [[50, 125], [100, 125], [100, 225], [50, 225]],
                ]
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
                [50 + 2 * 25, 100 + 2 * 50, 50 + 2 * 75, 100 + 2 * 150],
                [50 + 2 * 50, 100 + 2 * 125, 50 + 2 * 100, 100 + 2 * 225],
            ]
        ),
    ), "Expected coordinates to be first scaled 2x and then shifted into root coordinates (by [50, 100])"
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
        result["scaling_relative_to_parent"] == np.array([1.0, 1.0])
    ).all(), "Expected parent scaling to be set to 1.0"
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
    assert (
        result["scaling_relative_to_root_parent"] == np.array([1.0, 1.0])
    ).all(), "Expected root parent scaling to be set to 1.0"
    assert (
        result["keypoints_class_name"][0] == np.array(["a", "b"])
    ).all(), "Expected keypoints classes not to be touched"
    assert (
        result["keypoints_class_name"][1] == np.array([])
    ).all(), "Expected keypoints classes not to be touched"
    assert (
        result["keypoints_confidence"][0] == np.array([0.3, 0.4])
    ).all(), "Expected keypoints confidence not to be touched"
    assert (
        result["keypoints_confidence"][1] == np.array([])
    ).all(), "Expected keypoints confidence not to be touched"
    assert (
        result["keypoints_xy"][0]
        == np.array([[50 + 2 * 10, 100 + 2 * 20], [50 + 2 * 20, 100 + 2 * 30]])
    ).all(), "Expected keypoints xy to be scaled x2 and shifted"
    assert (
        result["keypoints_xy"][1] == np.array([])
    ).all(), "Expected empty keypoints xy to be left as is"
    assert np.allclose(
        result[POLYGON_KEY_IN_SV_DETECTIONS],
        np.array(
            [
                [
                    [50 + 2 * 25, 100 + 2 * 50],
                    [50 + 2 * 75, 100 + 2 * 50],
                    [50 + 2 * 75, 100 + 2 * 150],
                    [50 + 2 * 25, 100 + 2 * 150],
                ],
                [
                    [50 + 2 * 50, 100 + 2 * 125],
                    [50 + 2 * 100, 100 + 2 * 125],
                    [50 + 2 * 100, 100 + 2 * 225],
                    [50 + 2 * 50, 100 + 2 * 225],
                ],
            ]
        ),
    ), "Expected polygon metadata to be scaled and shifted into root coordinates"
    assert np.allclose(
        result["image_dimensions"], np.array([[1024, 512], [1024, 512]])
    ), "Expected image dimensions to be root dimensions"


def test_filter_out_unwanted_classes_from_sv_detections_batch_when_no_classes_defined() -> (
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
    expected_result = deepcopy(detections)

    # when
    result = filter_out_unwanted_classes_from_sv_detections_batch(
        predictions=[detections],
        classes_to_accept=None,
    )

    # then
    assert len(result) == 1, "Expected batch dimension not to change"
    assert len(result[0]) == 2, "Expected still to see 2 detections"
    assert result[0] == expected_result, "We expect nothing to be filtered out"
    assert result[0] is detections, "We expect operation to be in-place"


def test_filter_out_unwanted_classes_from_sv_detections_batch_when_empty_class_list_defined() -> (
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
    expected_result = deepcopy(detections)

    # when
    result = filter_out_unwanted_classes_from_sv_detections_batch(
        predictions=[detections],
        classes_to_accept=[],
    )

    # then
    assert len(result) == 1, "Expected batch dimension not to change"
    assert len(result[0]) == 2, "Expected still to see 2 detections"
    assert result[0] == expected_result, "We expect nothing to be filtered out"
    assert result[0] is detections, "We expect operation to be in-place"


def test_filter_out_unwanted_classes_from_sv_detections_batch_when_filtering_should_be_applied() -> (
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


def test_grab_batch_parameters() -> None:
    # given
    operations_parameters = {
        "non_batch": [1, 2, 3, 4],
        "batch_matching_dim": Batch(
            content=["a", "b", "c", "d"], indices=[(0,), (1,), (2,), (3,)]
        ),
        "batch_to_broadcast": Batch(content=["A"], indices=[(0,), (1,)]),
    }

    # when
    result = grab_batch_parameters(
        operations_parameters=operations_parameters,
        main_batch_size=4,
    )

    # then
    assert set(result.keys()) == {
        "batch_matching_dim",
        "batch_to_broadcast",
    }, "Only batch-parameters are supposed to be grabbed"
    assert list(result["batch_matching_dim"]) == [
        "a",
        "b",
        "c",
        "d",
    ], "Expected content of batch for `batch_matching_dim` not to be changed"
    assert list(result["batch_to_broadcast"]) == [
        "A",
        "A",
        "A",
        "A",
    ], "Expected elements of `batch_to_broadcast` to be broadcast"


def test_grab_batch_parameters_when_batch_parameters_not_spotted() -> None:
    # given
    operations_parameters = {
        "non_batch": [1, 2, 3, 4],
    }

    # when
    result = grab_batch_parameters(
        operations_parameters=operations_parameters,
        main_batch_size=4,
    )

    # then
    assert result == {}, "Expected nothing to be extracted"


def test_grab_batch_parameters_when_non_broadcastable_parameter_spotted() -> None:
    # given
    operations_parameters = {
        "non_batch": [1, 2, 3, 4],
        "batch_matching_dim": Batch(
            content=["a", "b", "c", "d"], indices=[(0,), (1,), (2,), (3,)]
        ),
        "batch_to_broadcast": Batch(
            content=["A", "B", "C"], indices=[(0,), (1,), (2,)]
        ),  # cannot be broadcast
    }

    # when

    with pytest.raises(ValueError):
        _ = grab_batch_parameters(
            operations_parameters=operations_parameters,
            main_batch_size=4,
        )


def test_grab_batch_parameters_when_non_empty_parameters_given() -> None:
    # given
    operations_parameters = {}

    # when
    result = grab_batch_parameters(
        operations_parameters=operations_parameters,
        main_batch_size=4,
    )

    # then
    assert result == {}, "Expected nothing to be found"


def test_grab_non_batch_parameters_when_non_batch_parameters_to_be_found() -> None:
    # given
    operations_parameters = {
        "non_batch": [1, 2, 3, 4],
        "batch_matching_dim": Batch(
            content=["a", "b", "c", "d"], indices=[(0,), (1,), (2,), (3,)]
        ),
        "batch_to_broadcast": Batch(content=["A"], indices=[(0,)]),
    }

    # when
    result = grab_non_batch_parameters(operations_parameters=operations_parameters)

    # then
    assert set(result.keys()) == {
        "non_batch"
    }, "Only non-batch-parameters are supposed to be grabbed"
    assert result["non_batch"] == [
        1,
        2,
        3,
        4,
    ], "Content of `non_batch` parameter cannot be changed"


def test_grab_non_batch_parameters_when_non_batch_parameters_not_to_be_found() -> None:
    # given
    operations_parameters = {
        "batch_matching_dim": Batch(
            content=["a", "b", "c", "d"], indices=[(0,), (1,), (2,), (3,)]
        ),
        "batch_to_broadcast": Batch(content=["A"], indices=[(0,)]),
    }

    # when
    result = grab_non_batch_parameters(operations_parameters=operations_parameters)

    # then
    assert result == {}, "Expected nothing to be extracted"


def test_grab_non_batch_parameters_when_empty_input_given() -> None:
    # given
    operations_parameters = {}

    # when
    result = grab_non_batch_parameters(operations_parameters=operations_parameters)

    # then
    assert result == {}, "Expected nothing to be extracted"


def test_scale_sv_detections_when_empty_detections_given() -> None:
    # given
    detections = sv.Detections.empty()

    # when
    result = scale_sv_detections(
        detections=detections,
        scale=1.2,
    )

    # then
    assert (
        result == sv.Detections.empty()
    ), "Expected still to see empty detections at the output"


def test_scale_sv_detections_when_scale_makes_output_bigger() -> None:
    # given
    mask = np.zeros((2, 200, 100), dtype=np.bool_)
    mask[0, 80:121, 30:71] = True
    mask[1, 170:191, 70:91] = True
    expected_mask = np.array(
        [
            cv2.resize(
                detection_mask.astype(np.uint8),
                (200, 400),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for detection_mask in mask
        ]
    )
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
            "scaling_relative_to_root_parent": np.array([0.5, 0.5]),
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
            "image_dimensions": np.array([[200, 100], [200, 100]]),
        },
    )

    # when
    result = scale_sv_detections(
        detections=detections,
        scale=2.0,
    )

    # then
    assert np.allclose(
        result.xyxy,
        np.array(
            [[2 * 25, 2 * 50, 2 * 75, 2 * 150], [2 * 50, 2 * 125, 2 * 100, 2 * 225]]
        ),
    ), "Expected coordinates to be scaled 2x"
    assert np.allclose(
        result.mask, expected_mask
    ), "Expected mask to be properly scaled 2x"
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
        result["parent_id"] == np.array([["crop_1", "crop_1"]])
    ).all(), "perant id should not be touched"
    assert (
        result["parent_coordinates"] == np.array([[10, 20], [10, 20]])
    ).all(), "Parent coordinates should not be touched"
    assert (
        result["parent_dimensions"] == np.array([[200, 100], [200, 100]])
    ).all(), "Parent dimensions should not be touched"
    assert (
        result["scaling_relative_to_parent"] == [2.0, 2.0]
    ).all(), "Parent scale should be denoted"
    assert (
        result["root_parent_id"] == np.array(["root", "root"])
    ).all(), "root stays root parent"
    assert (
        result["root_parent_coordinates"] == np.array([[50, 100], [50, 100]])
    ).all(), "Root coordinates not to be touched"
    assert (
        result["root_parent_dimensions"] == np.array([[1024, 512], [1024, 512]])
    ).all(), "Root dimensions not to be touched"
    assert (
        result["scaling_relative_to_root_parent"] == np.array([1.0, 1.0])
    ).all(), (
        "Root parent scale should be adjusted to the previous content (0.5 * 2.0 = 1.0)"
    )
    assert (
        result["keypoints_class_name"][0] == np.array(["a", "b"])
    ).all(), "Expected keypoints classes not to be touched"
    assert (
        result["keypoints_class_name"][1] == np.array([])
    ).all(), "Expected keypoints classes not to be touched"
    assert (
        result["keypoints_confidence"][0] == np.array([0.3, 0.4])
    ).all(), "Expected keypoints confidence not to be touched"
    assert (
        result["keypoints_confidence"][1] == np.array([])
    ).all(), "Expected keypoints confidence not to be touched"
    assert (
        result["keypoints_xy"][0]
        == np.array([[2.0 * 10, 2.0 * 20], [2.0 * 20, 2.0 * 30]]).round()
    ).all(), "Expected keypoints xy to be scaled"
    assert (
        result["keypoints_xy"][1] == np.array([])
    ).all(), "Expected empty keypoints xy to be left as is"
    assert np.allclose(
        result["image_dimensions"], np.array([[400, 200], [400, 200]])
    ), "Expected image dimensions to increase 2x"


def test_scale_sv_detections_when_scale_makes_output_smaller() -> None:
    # given
    mask = np.zeros((2, 200, 100), dtype=np.bool_)
    mask[0, 80:121, 30:71] = True
    mask[1, 170:191, 70:91] = True
    expected_mask = np.array(
        [
            cv2.resize(
                detection_mask.astype(np.uint8),
                (50, 100),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for detection_mask in mask
        ]
    )
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
            "scaling_relative_to_root_parent": np.array([0.5, 0.5]),
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
            "image_dimensions": np.array([[200, 100], [200, 100]]),
        },
    )

    # when
    result = scale_sv_detections(
        detections=detections,
        scale=0.5,
    )

    # then
    assert np.allclose(
        result.xyxy,
        np.array(
            [
                [0.5 * 25, 0.5 * 50, 0.5 * 75, 0.5 * 150],
                [0.5 * 50, 0.5 * 125, 0.5 * 100, 0.5 * 225],
            ]
        ).round(),
    ), "Expected coordinates to be scaled 0.5x"
    assert np.allclose(
        result.mask, expected_mask
    ), "Expected mask to be properly scaled 0.5x"
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
        result["parent_id"] == np.array([["crop_1", "crop_1"]])
    ).all(), "perant id should not be touched"
    assert (
        result["parent_coordinates"] == np.array([[10, 20], [10, 20]])
    ).all(), "Parent coordinates should not be touched"
    assert (
        result["parent_dimensions"] == np.array([[200, 100], [200, 100]])
    ).all(), "Parent dimensions should not be touched"
    assert (
        result["scaling_relative_to_parent"] == [0.5, 0.5]
    ).all(), "Parent scale should be denoted"
    assert (
        result["root_parent_id"] == np.array(["root", "root"])
    ).all(), "root stays root parent"
    assert (
        result["root_parent_coordinates"] == np.array([[50, 100], [50, 100]])
    ).all(), "Root coordinates not to be touched"
    assert (
        result["root_parent_dimensions"] == np.array([[1024, 512], [1024, 512]])
    ).all(), "Root dimensions not to be touched"
    assert (
        result["scaling_relative_to_root_parent"] == np.array([0.25, 0.25])
    ).all(), (
        "Root parent scale should be adjusted to the previous content (0.5 * 2.0 = 1.0)"
    )
    assert (
        result["keypoints_class_name"][0] == np.array(["a", "b"])
    ).all(), "Expected keypoints classes not to be touched"
    assert (
        result["keypoints_class_name"][1] == np.array([])
    ).all(), "Expected keypoints classes not to be touched"
    assert (
        result["keypoints_confidence"][0] == np.array([0.3, 0.4])
    ).all(), "Expected keypoints confidence not to be touched"
    assert (
        result["keypoints_confidence"][1] == np.array([])
    ).all(), "Expected keypoints confidence not to be touched"
    assert (
        result["keypoints_xy"][0]
        == np.array([[0.5 * 10, 0.5 * 20], [0.5 * 20, 0.5 * 30]]).round()
    ).all(), "Expected keypoints xy to be scaled"
    assert (
        result["keypoints_xy"][1] == np.array([])
    ).all(), "Expected empty keypoints xy to be left as is"
    assert np.allclose(
        result["image_dimensions"], np.array([[100, 50], [100, 50]])
    ), "Expected image dimensions to decrease 2x"


def test_scale_sv_detections_preserves_edge_flush_mask_area_when_downscaling() -> None:
    """Regression: Dataset Upload downscales >max_image_size images; masks flush
    to the image edge must survive scale_sv_detections without collapsing."""
    # given — 4096→2080 style downscale with a right-edge instance + speck
    orig = 4096
    target = 2080
    scale = target / orig
    mask = np.zeros((orig, orig), dtype=bool)
    mask[400:1600, 3100:orig] = True
    mask[820:828, 3088:3096] = True  # detached speck near left of instance
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

    # when
    result = scale_sv_detections(
        detections=detections,
        scale=(scale, scale),
        target_size_wh=(target, target),
    )

    # then
    assert result.mask is not None
    assert result.mask.shape == (1, target, target)
    assert np.allclose(result["image_dimensions"], np.array([[target, target]]))
    # Main body should dominate; speck must not wipe the instance
    assert result.mask[0].sum() > 200_000
    # Right edge of canvas should still contain mask pixels
    assert result.mask[0][:, -1].any()


def test_scale_sv_detections_anisotropic_matches_exact_target_canvas() -> None:
    """Aspect-preserving resize can make scale_x != scale_y after int truncation.
    Annotations must land on the exact uploaded JPEG size, not round(dim * sy)."""
    # 4000x1080 → max 2080x2080 → 2080 x 561 (scale_x=0.52, scale_y≈0.5194)
    orig_w, orig_h = 4000, 1080
    target_w, target_h = 2080, 561
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h
    assert abs(scale_x - scale_y) > 1e-4  # the bug case

    mask = np.zeros((orig_h, orig_w), dtype=bool)
    # Full-height strip flush to the right edge
    mask[:, orig_w - 200 : orig_w] = True
    detections = sv.Detections(
        xyxy=np.array([[orig_w - 200, 0, orig_w - 1, orig_h - 1]], dtype=np.float64),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={
            "class_name": np.array(["edge"]),
            "detection_id": np.array(["d1"]),
            "image_dimensions": np.array([[orig_h, orig_w]]),
            POLYGON_KEY_IN_SV_DETECTIONS: np.array(
                [
                    [
                        [orig_w - 200, 0],
                        [orig_w - 1, 0],
                        [orig_w - 1, orig_h - 1],
                        [orig_w - 200, orig_h - 1],
                    ]
                ],
                dtype=np.int32,
            ),
            SCALING_RELATIVE_TO_PARENT_KEY: np.array([0.5]),
            SCALING_RELATIVE_TO_ROOT_PARENT_KEY: np.array([0.25]),
        },
    )

    result = scale_sv_detections(
        detections=detections,
        scale=(scale_x, scale_y),
        target_size_wh=(target_w, target_h),
        update_scaling_metadata=False,
    )

    assert result.mask.shape == (1, target_h, target_w)
    assert np.allclose(result["image_dimensions"], np.array([[target_h, target_w]]))
    x1, y1, x2, y2 = result.xyxy[0]
    assert x2 <= target_w
    assert y2 <= target_h
    # Right-edge instance must still touch the destination right edge
    assert result.mask[0][:, -1].any()
    assert x2 >= target_w - 2
    # Height-only isotropic scale maps width to the wrong canvas size
    isotropic_w = int(round(orig_w * scale_y))
    assert isotropic_w != target_w
    polygon = result.data[POLYGON_KEY_IN_SV_DETECTIONS][0]
    assert polygon[:, 0].max() >= target_w - 2
    assert np.array_equal(result[SCALING_RELATIVE_TO_PARENT_KEY], np.array([0.5]))
    assert np.array_equal(result[SCALING_RELATIVE_TO_ROOT_PARENT_KEY], np.array([0.25]))


def test_scale_sv_detections_rejects_anisotropic_scalar_metadata() -> None:
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 20, 20]], dtype=np.float64),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={
            "class_name": np.array(["obj"]),
            "detection_id": np.array(["d1"]),
            "image_dimensions": np.array([[100, 100]]),
        },
    )

    with pytest.raises(
        ValueError,
        match="Anisotropic scaling cannot be represented",
    ):
        scale_sv_detections(
            detections=detections,
            scale=(0.5, 0.6),
        )


def test_scale_sv_detections_regenerates_rle_when_scale_changes_mask() -> None:
    # given
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True
    detections = sv.Detections(
        xyxy=np.array([[20, 20, 40, 40]], dtype=np.float64),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={
            "class_name": np.array(["obj"]),
            "detection_id": np.array(["d1"]),
            "image_dimensions": np.array([[100, 100]]),
            "rle_mask": np.array([{"size": [100, 100], "counts": "x"}], dtype=object),
        },
    )

    # when
    result = scale_sv_detections(detections=detections, scale=0.5)

    # then
    assert result.mask.shape == (1, 50, 50)
    assert "rle_mask" in result.data
    resized_rle = result.data["rle_mask"][0]
    assert resized_rle["size"] == [50, 50]
    assert isinstance(resized_rle["counts"], str)
    decoded_mask = mask_utils.decode(
        {
            "size": resized_rle["size"],
            "counts": resized_rle["counts"].encode("utf-8"),
        }
    ).astype(bool)
    assert np.array_equal(decoded_mask, result.mask[0])


def test_scale_sv_detections_passes_rle_only_masks_through_untouched() -> None:
    # RLE-only predictions (mask=None) are intentionally not resized to avoid
    # a decode/resize/re-encode cost on a path no stock workflow exercises -
    # boxes scale, but the RLE stays sized to the source canvas. None entries
    # (see convert_inference_detections_batch_to_sv_detections) must not crash.
    # given
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    rle_mask = mask_utils.encode(np.asfortranarray(mask))
    detections = sv.Detections(
        xyxy=np.array([[20, 20, 40, 40], [50, 50, 70, 70]], dtype=np.float64),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 1]),
        data={
            "class_name": np.array(["obj", "other"]),
            "detection_id": np.array(["d1", "d2"]),
            "image_dimensions": np.array([[100, 100], [100, 100]]),
            "rle_mask": np.array([None, rle_mask], dtype=object),
        },
    )

    # when
    result = scale_sv_detections(detections=detections, scale=0.5)

    # then
    assert result.mask is None
    assert np.allclose(result.xyxy, np.array([[10, 10, 20, 20], [25, 25, 35, 35]]))
    assert result.data["rle_mask"][0] is None, "None RLE entry must stay None"
    # scale_sv_detections deep-copies its input, so compare by value
    assert result.data["rle_mask"][1] == rle_mask, "RLE must be left untouched"


def test_scale_sv_detections_preserves_empty_masks_and_matching_rles() -> None:
    # given
    masks = np.zeros((2, 4, 4), dtype=np.uint8)
    masks[0, 3, 3] = 1
    masks[1, 0:3, 0:3] = 1
    rle_masks = np.array(
        [mask_utils.encode(np.asfortranarray(mask)) for mask in masks],
        dtype=object,
    )
    detections = sv.Detections(
        xyxy=np.array([[3, 3, 4, 4], [0, 0, 3, 3]], dtype=np.float64),
        mask=masks.astype(bool),
        confidence=np.array([0.5, 0.9]),
        class_id=np.array([0, 1]),
        data={
            "class_name": np.array(["thin", "body"]),
            "detection_id": np.array(["thin-id", "body-id"]),
            "image_dimensions": np.array([[4, 4], [4, 4]]),
            "rle_mask": rle_masks,
        },
    )

    # when
    result = scale_sv_detections(detections=detections, scale=0.5)
    serialized_result = serialise_rle_sv_detections(detections=result)

    # then
    assert len(result) == 2
    assert result.data["detection_id"].tolist() == ["thin-id", "body-id"]
    assert len(result.data["rle_mask"]) == 2
    assert [
        prediction["detection_id"] for prediction in serialized_result["predictions"]
    ] == ["thin-id", "body-id"]
    serialized_mask_areas = []
    for prediction in serialized_result["predictions"]:
        serialized_mask = mask_utils.decode(
            {
                "size": prediction["rle_mask"]["size"],
                "counts": prediction["rle_mask"]["counts"].encode("utf-8"),
            }
        )
        serialized_mask_areas.append(int(serialized_mask.sum()))
    assert serialized_mask_areas == [0, 4]


def test_scale_sv_detections_keeps_rle_when_scale_is_noop() -> None:
    """No-op scale (e.g. root-coordinates pass with scaling key == 1.0) must not
    strip `rle_mask` - the RLE-kind output serializer requires it."""
    # given
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 20:40] = True
    detections = sv.Detections(
        xyxy=np.array([[20, 20, 40, 40]], dtype=np.float64),
        mask=np.array([mask]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        data={
            "class_name": np.array(["obj"]),
            "detection_id": np.array(["d1"]),
            "image_dimensions": np.array([[100, 100]]),
            "rle_mask": np.array([{"size": [100, 100], "counts": "x"}], dtype=object),
        },
    )

    # when
    result = scale_sv_detections(detections=detections, scale=1.0)

    # then
    assert "rle_mask" in result.data
    assert result.mask.shape == (1, 100, 100)
    assert np.array_equal(result.mask, detections.mask)


def test_remove_unexpected_keys_from_dictionary_when_empty_dict_given() -> None:
    # when
    result = remove_unexpected_keys_from_dictionary(
        dictionary={}, expected_keys={"some", "other"}
    )

    # then
    assert result == {}


def test_remove_unexpected_keys_from_dictionary_when_non_empty_dict_given_but_no_keys_expected() -> (
    None
):
    # when
    result = remove_unexpected_keys_from_dictionary(
        dictionary={"a": 1, "b": 2}, expected_keys=set()
    )

    # then
    assert result == {}


def test_remove_unexpected_keys_from_dictionary_when_part_of_keys_are_not_expected() -> (
    None
):
    # when
    result = remove_unexpected_keys_from_dictionary(
        dictionary={"a": 1, "b": 2, "c": 3}, expected_keys={"a", "d"}
    )

    # then
    assert result == {"a": 1}


@pytest.mark.parametrize(
    "width, height, max_edge, expected",
    [
        pytest.param(100, 50, 2048, (100, 50), id="below-limit-noop"),
        pytest.param(2048, 1024, 2048, (2048, 1024), id="landscape-at-limit-noop"),
        pytest.param(4096, 2048, 2048, (2048, 1024), id="landscape-downscale"),
        pytest.param(1000, 4000, 2048, (512, 2048), id="portrait-downscale"),
        pytest.param(4096, 4096, 2048, (2048, 2048), id="square-downscale"),
        pytest.param(3000, 2000, 2048, (2048, 1365), id="rounds-down-1365.33"),
        pytest.param(4097, 2048, 2048, (2048, 1024), id="rounds-up-1023.75"),
        pytest.param(4096, 1021, 2048, (2048, 510), id="half-510.5-rounds-to-even-510"),
        pytest.param(10000, 1, 2048, (2048, 1), id="wide-short-edge-clamps-to-1px"),
        pytest.param(1, 10000, 2048, (1, 2048), id="tall-short-edge-clamps-to-1px"),
        pytest.param(300, 200, 1, (1, 1), id="max-edge-of-1"),
    ],
)
def test_scale_dimensions_to_max_edge(
    width: int, height: int, max_edge: int, expected: tuple
) -> None:
    # when
    result = scale_dimensions_to_max_edge(width=width, height=height, max_edge=max_edge)

    # then
    assert result == expected


@pytest.mark.parametrize(
    "width, height",
    [
        (2899, 2841),
        (1920, 2560),
        (2164, 1868),
    ],
)
def test_scale_dimensions_to_max_edge_preserves_invariants(
    width: int, height: int
) -> None:
    # when
    scaled_width, scaled_height = scale_dimensions_to_max_edge(
        width=width, height=height, max_edge=DETECTION_MAX_EDGE_PIXELS
    )

    # then
    assert max(scaled_width, scaled_height) <= DETECTION_MAX_EDGE_PIXELS
    assert scaled_width >= 1 and scaled_height >= 1
    # aspect ratio preserved within the error introduced by rounding one edge
    original_ratio = width / height
    scaled_ratio = scaled_width / scaled_height
    assert abs(scaled_ratio - original_ratio) <= original_ratio / min(
        scaled_width, scaled_height
    )
    assert scale_dimensions_to_max_edge(
        width=scaled_width, height=scaled_height, max_edge=DETECTION_MAX_EDGE_PIXELS
    ) == (scaled_width, scaled_height)
