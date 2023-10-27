import json

import pytest

from inference.core.active_learning.post_processing import encode_prediction
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
