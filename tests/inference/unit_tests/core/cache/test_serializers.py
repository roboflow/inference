import os
from unittest.mock import MagicMock

import pytest

from inference.core.cache import serializers
from inference.core.cache.serializers import (
    build_condensed_response,
    to_cachable_inference_item,
)
from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    ClassificationPrediction,
    InstanceSegmentationInferenceResponse,
    InstanceSegmentationPrediction,
    Keypoint,
    KeypointsDetectionInferenceResponse,
    KeypointsPrediction,
    MultiLabelClassificationInferenceResponse,
    MultiLabelClassificationPrediction,
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
    Point,
)


@pytest.fixture
def mock_classification_data():
    mock_response = MagicMock(spec=ClassificationInferenceResponse)
    predictions = [
        ClassificationPrediction(**{"class": "cat", "class_id": 1, "confidence": 0.8})
    ]
    mock_response.top = "cat"
    mock_response.predictions = predictions
    mock_response.confidence = 0.8
    mock_response.time = "2023-10-01T12:00:00Z"
    return mock_response


def test_build_condensed_response_single_classification(mock_classification_data):
    mock_response = mock_classification_data
    result = build_condensed_response(mock_response)
    assert len(result) == 1
    assert "predictions" in result[0]
    assert "time" in result[0]


def test_build_condensed_response_multiple_classification(mock_classification_data):
    mock_response = mock_classification_data
    responses = [mock_response, mock_response]
    result = build_condensed_response(responses)
    assert len(result) == 2


def test_build_condensed_response_no_predictions_classification(
    mock_classification_data,
):
    mock_response = mock_classification_data
    mock_response.predictions = None
    result = build_condensed_response(mock_response)
    assert len(result) == 0


@pytest.fixture
def mock_object_detection_data():
    mock_request = MagicMock(spec=ObjectDetectionInferenceRequest)
    mock_request.id = "test_id"
    mock_request.confidence = 0.85
    mock_request.dict.return_value = {
        "api_key": "test_key",
        "confidence": 0.85,
        "model_id": "sharks",
        "model_type": "object_detection",
    }

    mock_response = MagicMock(spec=ObjectDetectionInferenceResponse)
    mock_response.predictions = [
        ObjectDetectionPrediction(
            **{
                "class_name": "tiger-shark",
                "confidence": 0.95,
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "class_confidence": None,
                "class_id": 1,
                "class": "tiger-shark",
            }
        ),
        ObjectDetectionPrediction(
            **{
                "class_name": "hammerhead",
                "confidence": 0.95,
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "class_confidence": None,
                "class_id": 2,
                "class": "hammerhead",
            }
        ),
        ObjectDetectionPrediction(
            **{
                "class_name": "white-shark",
                "confidence": 0.95,
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "class_confidence": None,
                "class_id": 3,
                "class": "white-shark",
            }
        ),
    ]
    mock_response.time = "2023-10-01T12:00:00Z"

    return mock_request, mock_response


def test_to_cachable_inference_item_no_tiny_cache_object_detection(
    mock_object_detection_data,
):
    mock_request, mock_response = mock_object_detection_data
    os.environ["TINY_CACHE"] = "False"
    result = to_cachable_inference_item(mock_request, mock_response)
    assert result["inference_id"] == mock_request.id
    assert result["request"]["api_key"] == mock_request.dict.return_value["api_key"]
    assert (
        result["response"][0]["predictions"][0]["class"]
        == mock_response.predictions[0].class_name
    )
    assert (
        result["response"][0]["predictions"][0]["confidence"]
        == mock_response.predictions[0].confidence
    )


def test_to_cachable_inference_item_with_tiny_cache_object_detection(
    mock_object_detection_data,
):
    mock_request, mock_response = mock_object_detection_data
    os.environ["TINY_CACHE"] = "True"
    result = to_cachable_inference_item(mock_request, mock_response)
    assert result["inference_id"] == mock_request.id
    assert result["request"]["api_key"] == mock_request.dict.return_value["api_key"]
    assert (
        result["response"][0]["predictions"][0]["class"]
        == mock_response.predictions[0].class_name
    )
    assert (
        result["response"][0]["predictions"][0]["confidence"]
        == mock_response.predictions[0].confidence
    )


@pytest.mark.parametrize("tiny_cache", [True, False], ids=["tiny", "full"])
def test_to_cachable_inference_item_fills_model_id_missing_from_request(
    monkeypatch: pytest.MonkeyPatch,
    tiny_cache: bool,
) -> None:
    monkeypatch.setattr(serializers, "TINY_CACHE", tiny_cache)
    request = YOLOWorldInferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"},
        text=["person"],
    )
    response = ObjectDetectionInferenceResponse(
        predictions=[], image={"width": 1, "height": 1}, time=0.1
    )

    result = to_cachable_inference_item(request, response, "yolo_world/l")

    assert result["request"]["model_id"] == "yolo_world/l"


def test_to_cachable_inference_item_keeps_model_id_of_request(
    monkeypatch: pytest.MonkeyPatch,
    mock_object_detection_data,
) -> None:
    monkeypatch.setattr(serializers, "TINY_CACHE", True)
    mock_request, mock_response = mock_object_detection_data

    result = to_cachable_inference_item(
        mock_request, mock_response, model_id="sharks:capabilities=image_embeddings"
    )

    assert result["request"]["model_id"] == "sharks"


@pytest.mark.parametrize("tiny_cache", [True, False], ids=["tiny", "full"])
def test_to_cachable_inference_item_reports_box_threshold_as_grounding_dino_confidence(
    monkeypatch: pytest.MonkeyPatch,
    tiny_cache: bool,
) -> None:
    monkeypatch.setattr(serializers, "TINY_CACHE", tiny_cache)
    request = GroundingDINOInferenceRequest(
        image={"type": "url", "value": "https://some/image.jpg"},
        text=["dog"],
        box_threshold=0.3,
    )
    response = ObjectDetectionInferenceResponse(
        predictions=[], image={"width": 1, "height": 1}, time=0.1
    )

    result = to_cachable_inference_item(
        request, response, model_id="grounding_dino/default"
    )

    assert result["request"]["confidence"] == 0.3


def test_build_condensed_response_no_predictions_object_detection(
    mock_object_detection_data,
):
    _, mock_response = mock_object_detection_data
    mock_response.predictions = None
    result = build_condensed_response(mock_response)
    assert len(result) == 0


@pytest.fixture
def mock_multilabel_classification_data():
    mock_response = MagicMock(spec=MultiLabelClassificationInferenceResponse)
    mock_response.predictions = {
        "cat": MultiLabelClassificationPrediction(confidence=0.8, class_id=1),
        "dog": MultiLabelClassificationPrediction(confidence=0.7, class_id=2),
    }
    mock_response.time = "2023-10-01T12:00:00Z"
    return mock_response


@pytest.fixture
def mock_instance_segmentation_data():
    mock_response = MagicMock(spec=InstanceSegmentationInferenceResponse)
    mock_response.predictions = [
        InstanceSegmentationPrediction(
            **{
                "class": "person",
                "confidence": 0.9,
                "class_confidence": None,
                "detection_id": "1",
                "parent_id": None,
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "points": [Point(x=0, y=0)],
                "class_id": 1,
            }
        )
    ]
    mock_response.time = "2023-10-01T12:00:00Z"
    return mock_response


@pytest.fixture
def mock_keypoints_detection_data():
    mock_response = MagicMock(spec=KeypointsDetectionInferenceResponse)
    mock_response.predictions = [
        KeypointsPrediction(
            **{
                "class": "person",
                "confidence": 0.9,
                "class_confidence": None,
                "detection_id": "1",
                "parent_id": None,
                "x": 0,
                "y": 0,
                "width": 0,
                "height": 0,
                "keypoints": [
                    Keypoint(
                        **{
                            "x": 0,
                            "y": 0,
                            "confidence": 0.8,
                            "class_id": 1,
                            "class": "nose",
                        }
                    )
                ],
                "class_id": 1,
            }
        )
    ]
    mock_response.time = "2023-10-01T12:00:00Z"
    return mock_response


def test_build_condensed_response_instance_segmentation(
    mock_instance_segmentation_data,
):
    mock_response = mock_instance_segmentation_data
    result = build_condensed_response(mock_response)
    assert len(result) == 1
    assert "predictions" in result[0]
    assert "time" in result[0]


def test_build_condensed_response_keypoints_detection(mock_keypoints_detection_data):
    mock_response = mock_keypoints_detection_data
    result = build_condensed_response(mock_response)
    assert len(result) == 1
    assert "predictions" in result[0]
    assert "time" in result[0]


def test_build_condensed_response_object_detection(mock_object_detection_data):
    _, mock_response = mock_object_detection_data
    result = build_condensed_response(mock_response)
    assert len(result) == 1
    assert "predictions" in result[0]
    assert "time" in result[0]
