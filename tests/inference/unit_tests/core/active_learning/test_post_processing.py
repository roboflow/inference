import json

import pytest

from inference.core.active_learning.post_processing import (
    adjust_bbox_coordinates_to_client_scaling_factor,
    adjust_object_detection_predictions_to_client_scaling_factor,
    adjust_points_coordinates_to_client_scaling_factor,
    adjust_prediction_to_client_scaling_factor,
    adjust_prediction_with_bbox_and_points_to_client_scaling_factor,
    encode_prediction,
)
from inference.core.exceptions import PredictionFormatNotSupported


def test_encode_prediction_when_non_classification_task_prediction_is_given() -> None:
    # given
    prediction = {
        "image": {"height": 960, "width": 1280},
        "predictions": [
            {
                "x": 200.0,
                "y": 400.0,
                "width": 400.0,
                "height": 600.0,
                "confidence": 0.9,
                "class": "A",
            }
        ],
    }

    # when
    result = encode_prediction(
        prediction=prediction, prediction_type="object-detection"
    )
    decoded_result = json.loads(result[0])

    # then
    assert result[1] == "json"
    assert decoded_result == prediction


def test_encode_prediction_when_classification_task_prediction_is_given() -> None:
    # given
    prediction = {
        "frame_id": None,
        "time": 0.1833498750002036,
        "image": {"width": 3487, "height": 2444},
        "predictions": [
            {"class": "Ambulance", "class_id": 0, "confidence": 0.6865},
            {"class": "Limousine", "class_id": 16, "confidence": 0.2673},
            {"class": "Truck", "class_id": 7, "confidence": 0.0008},
        ],
        "top": "Ambulance",
        "confidence": 0.6865,
    }

    # when
    result = encode_prediction(prediction=prediction, prediction_type="classification")

    # then
    assert result[1] == "txt"
    assert result[0] == "Ambulance"


def test_encode_prediction_when_multi_label_classification_task_prediction_is_given() -> (
    None
):
    # given
    prediction = {
        "frame_id": None,
        "time": 0.16578604099959193,
        "image": {"width": 499, "height": 417},
        "predictions": {
            "cat": {"confidence": 0.9713834524154663},
            "dog": {"confidence": 0.02873784303665161},
        },
        "predicted_classes": ["cat"],
    }

    # when
    with pytest.raises(PredictionFormatNotSupported):
        _ = encode_prediction(prediction=prediction, prediction_type="classification")


def test_adjust_points_coordinates_to_client_scaling_factor() -> None:
    # given
    points = [{"x": 50, "y": 100}, {"x": 80, "y": 120}]

    # when
    result = adjust_points_coordinates_to_client_scaling_factor(
        points=points,
        scaling_factor=0.5,
    )

    # then
    assert result == [{"x": 100.0, "y": 200.0}, {"x": 160.0, "y": 240.0}]


def test_adjust_bbox_coordinates_to_client_scaling_factor() -> None:
    # given
    bbox = {
        "x": 50.0,
        "y": 100.0,
        "width": 100.0,
        "height": 150.0,
        "confidence": 0.9,
        "class": "A",
    }

    # when
    result = adjust_bbox_coordinates_to_client_scaling_factor(
        bbox=bbox,
        scaling_factor=0.5,
    )

    # then
    assert result == {
        "x": 100.0,
        "y": 200.0,
        "width": 200.0,
        "height": 300.0,
        "confidence": 0.9,
        "class": "A",
    }


def test_adjust_prediction_with_bbox_and_points_to_client_scaling_factor() -> None:
    # given
    predictions = [
        {
            "x": 50.0,
            "y": 100.0,
            "width": 100.0,
            "height": 150.0,
            "confidence": 0.9,
            "class": "A",
            "points": [{"x": 50, "y": 100}, {"x": 80, "y": 120}],
        }
    ]

    # when
    result = adjust_prediction_with_bbox_and_points_to_client_scaling_factor(
        predictions=predictions,
        scaling_factor=0.5,
        points_key="points",
    )

    # then
    assert result == [
        {
            "x": 100.0,
            "y": 200.0,
            "width": 200.0,
            "height": 300.0,
            "confidence": 0.9,
            "class": "A",
            "points": [{"x": 100.0, "y": 200.0}, {"x": 160.0, "y": 240.0}],
        }
    ]


def test_adjust_object_detection_predictions_to_client_scaling_factor() -> None:
    # given
    predictions = [
        {
            "x": 50.0,
            "y": 100.0,
            "width": 100.0,
            "height": 150.0,
            "confidence": 0.9,
            "class": "A",
        }
    ]

    # when
    result = adjust_object_detection_predictions_to_client_scaling_factor(
        predictions=predictions,
        scaling_factor=0.5,
    )

    # then
    assert result == [
        {
            "x": 100.0,
            "y": 200.0,
            "width": 200.0,
            "height": 300.0,
            "confidence": 0.9,
            "class": "A",
        }
    ]


def test_adjust_prediction_to_client_scaling_factor_when_scaling_factor_is_not_enabled() -> (
    None
):
    # given
    prediction = {
        "visualization": None,
        "frame_id": None,
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {"class": "Corn", "class_id": 1, "confidence": 0.4329},
            {"class": "Rice", "class_id": 0, "confidence": 0.3397},
        ],
        "top": "Corn",
        "confidence": 0.4329,
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=1.0, prediction_type="classification"
    )

    # then
    assert result is prediction


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_classification() -> (
    None
):
    # given
    prediction = {
        "visualization": None,
        "frame_id": None,
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {"class": "Corn", "class_id": 1, "confidence": 0.4329},
            {"class": "Rice", "class_id": 0, "confidence": 0.3397},
        ],
        "top": "Corn",
        "confidence": 0.4329,
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=0.5, prediction_type="classification"
    )

    # then
    assert result["image"] == {"width": 448, "height": 300}
    for key in [
        "visualization",
        "frame_id",
        "time",
        "predictions",
        "top",
        "confidence",
    ]:
        assert result[key] == prediction[key]


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_object_detection() -> (
    None
):
    # given
    prediction = {
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {
                "x": 21.0,
                "y": 128.0,
                "width": 42.0,
                "height": 48.0,
                "confidence": 0.9,
                "class": "wood-log",
                "class_id": 0,
            }
        ],
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction,
        scaling_factor=0.5,
        prediction_type="object-detection",
    )

    # then
    assert result["image"] == {"width": 448, "height": 300}
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == {
        "x": 42.0,
        "y": 256.0,
        "width": 84.0,
        "height": 96.0,
        "confidence": 0.9,
        "class": "wood-log",
        "class_id": 0,
    }


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_instance_segmentation() -> (
    None
):
    # given
    prediction = {
        "visualization": None,
        "frame_id": None,
        "time": 0.26239124999847263,
        "image": {"width": 224, "height": 150},
        "predictions": [
            {
                "x": 21.0,
                "y": 128.0,
                "width": 42.0,
                "height": 48.0,
                "confidence": 0.9,
                "class": "wood-log",
                "class_id": 0,
                "points": [{"x": 100.0, "y": 200.0}],
            }
        ],
    }
    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction,
        scaling_factor=0.5,
        prediction_type="instance-segmentation",
    )

    # then
    assert result["image"] == {"width": 448, "height": 300}
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == {
        "x": 42.0,
        "y": 256.0,
        "width": 84.0,
        "height": 96.0,
        "confidence": 0.9,
        "class": "wood-log",
        "class_id": 0,
        "points": [{"x": 200.0, "y": 400.0}],
    }


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_multi_label_classification() -> (
    None
):
    # given
    prediction = {
        "time": 0.14,
        "image": {"width": 224, "height": 224},
        "predictions": {"cat": {"confidence": 0.49}, "dog": {"confidence": 0.92}},
        "predicted_classes": ["cat", "dog"],
    }

    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=0.5, prediction_type="classification"
    )

    # then
    assert result["image"] == {"width": 448, "height": 448}
    assert result["time"] == prediction["time"]
    assert result["predictions"] == prediction["predictions"]
    assert result["predicted_classes"] == prediction["predicted_classes"]


def test_adjust_prediction_to_client_scaling_factor_when_scaling_is_enabled_against_stub_prediction() -> (
    None
):
    # given
    prediction = {
        "time": 0.0002442499971948564,
        "is_stub": True,
        "model_id": "asl-poly-instance-seg/0",
        "task_type": "instance-segmentation",
    }

    # when
    result = adjust_prediction_to_client_scaling_factor(
        prediction=prediction, scaling_factor=0.5, prediction_type="object-detection"
    )

    # then
    assert result == prediction
