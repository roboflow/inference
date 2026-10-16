import base64
from io import BytesIO
from typing import Union
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from inference.core.entities.requests.inference import ClassificationInferenceRequest
from inference.core.entities.responses.inference import (
    AnomalyDetectionResponse,
    ClassificationInferenceResponse,
    response_from_type,
)
from inference.core.exceptions import PostProcessingError
from inference.core.models.inference_models_adapters import (
    InferenceModelsAnomalyDetectionAdapter,
    prepare_anomaly_detection_response,
)
from inference.core.utils.anomaly_map_encoding import decode_anomaly_map
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference_models import AnomalyDetectionPrediction

CLASS_NAMES = ["normal", "anomalous"]


def prediction(
    scores=(2.0,), threshold=2.0, anomaly_maps=None
) -> AnomalyDetectionPrediction:
    scores_tensor = torch.tensor(list(scores), dtype=torch.float64)
    return AnomalyDetectionPrediction(
        anomaly_scores=scores_tensor,
        anomaly_threshold=threshold,
        is_anomalous=scores_tensor >= threshold,
        anomaly_maps=(
            torch.from_numpy(np.stack(anomaly_maps)) if anomaly_maps else None
        ),
    )


def request(**kwargs) -> ClassificationInferenceRequest:
    image = BytesIO()
    Image.new("RGB", (20, 10)).save(image, format="PNG")
    return ClassificationInferenceRequest(
        model_id="defects/1",
        image={"type": "base64", "value": base64.b64encode(image.getvalue()).decode()},
        **kwargs,
    )


def test_both_architectures_are_served_by_the_anomaly_adapter():
    assert (
        ROBOFLOW_MODEL_TYPES[("classification", "patchcore")]
        is InferenceModelsAnomalyDetectionAdapter
    )
    assert (
        ROBOFLOW_MODEL_TYPES[("classification", "foundad")]
        is InferenceModelsAnomalyDetectionAdapter
    )
    assert InferenceModelsAnomalyDetectionAdapter.accepts_anomaly_threshold is True


def test_response_is_not_a_classification_response():
    assert not issubclass(AnomalyDetectionResponse, ClassificationInferenceResponse)
    for field in ("predictions", "top", "confidence"):
        assert field not in AnomalyDetectionResponse.model_fields


def test_response_carries_score_threshold_and_verdict_only():
    results = prepare_anomaly_detection_response(
        prediction(scores=(1.0, 2.0, 3.5), threshold=2.0),
        image_sizes=[(10, 20), (30, 40), (50, 60)],
    )

    assert [r.anomaly_score for r in results] == [1.0, 2.0, 3.5]
    assert [r.anomaly_threshold for r in results] == [2.0, 2.0, 2.0]
    assert [r.is_anomalous for r in results] == [False, True, True]
    assert [r.anomaly_map for r in results] == [None, None, None]
    assert (results[0].image.width, results[0].image.height) == (20, 10)
    assert (results[2].image.width, results[2].image.height) == (60, 50)
    dumped = results[1].model_dump(by_alias=True, exclude_none=True)
    assert set(dumped) == {
        "image",
        "anomaly_score",
        "anomaly_threshold",
        "is_anomalous",
    }


def test_response_count_must_match_images():
    with pytest.raises(PostProcessingError):
        prepare_anomaly_detection_response(
            prediction(scores=(1.0,)), image_sizes=[(10, 20), (30, 40)]
        )


def test_http_serialization_keeps_anomaly_fields_and_encodes_the_map():
    app = FastAPI()

    @app.get(
        "/test",
        response_model=Union[AnomalyDetectionResponse, ClassificationInferenceResponse],
    )
    def predict():
        return prepare_anomaly_detection_response(
            prediction(anomaly_maps=[np.ones((10, 20), dtype=np.float32)]),
            image_sizes=[(300, 400)],
        )[0]

    body = TestClient(app).get("/test").json()
    assert body["anomaly_score"] == 2.0
    assert body["is_anomalous"] is True
    assert body["anomaly_threshold"] == 2.0
    assert "predictions" not in body
    assert "top" not in body
    assert "confidence" not in body
    assert body["anomaly_map"]["shape"] == [10, 20]
    assert body["anomaly_map"]["dtype"] == "float32"
    # The map is served at network resolution, never at the image size.
    assert np.array_equal(
        decode_anomaly_map(body["anomaly_map"]), np.ones((10, 20), dtype=np.float32)
    )


def test_request_forwards_map_flag_and_threshold_to_the_model():
    model = object.__new__(InferenceModelsAnomalyDetectionAdapter)
    model.class_names = CLASS_NAMES
    model._model = Mock(resolved_model=None)
    model._model.post_process.return_value = prediction(scores=(2.0,), threshold=1.5)

    response = model.infer_from_request(
        request(include_anomaly_map=True, anomaly_threshold=1.5, confidence=0.99)
    )

    assert isinstance(response, AnomalyDetectionResponse)
    assert response.is_anomalous is True
    assert response.anomaly_threshold == 1.5
    assert response.image.width == 20
    assert response.image.height == 10
    bgr_images = model._model.pre_process.call_args.args[0]
    assert bgr_images[0].shape == (10, 20, 3)
    assert model._model.pre_process.call_args.kwargs["input_color_format"] == "bgr"
    # The request confidence never reaches the model: only the two anomaly knobs.
    assert model._model.post_process.call_args.kwargs == {
        "include_anomaly_map": True,
        "anomaly_threshold": 1.5,
    }


def test_request_without_threshold_lets_the_model_use_the_saved_one():
    model = object.__new__(InferenceModelsAnomalyDetectionAdapter)
    model.class_names = CLASS_NAMES
    model._model = Mock(resolved_model=None)
    model._model.post_process.return_value = prediction()

    model.infer_from_request(request())

    assert model._model.post_process.call_args.kwargs == {
        "include_anomaly_map": False,
        "anomaly_threshold": None,
    }


def test_visualization_draws_the_verdict_without_class_confidences():
    model = object.__new__(InferenceModelsAnomalyDetectionAdapter)
    model.class_names = CLASS_NAMES
    model._model = Mock(resolved_model=None)
    model._model.post_process.return_value = prediction()

    response = model.infer_from_request(request(visualize_predictions=True))

    assert isinstance(response.visualization, bytes)
    assert Image.open(BytesIO(response.visualization)).size == (20, 10)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), "high"])
def test_request_rejects_non_finite_or_non_numeric_threshold(invalid):
    with pytest.raises(ValidationError):
        request(anomaly_threshold=invalid)


def test_request_threshold_defaults_to_none_and_accepts_zero():
    assert request().anomaly_threshold is None
    assert request(anomaly_threshold=0).anomaly_threshold == 0.0


def test_response_from_type_recognises_the_anomaly_shape():
    # The enterprise parallel server rebuilds responses from JSON by task type.
    response = response_from_type(
        "classification",
        {
            "image": {"width": 20, "height": 10},
            "anomaly_score": 1.5,
            "anomaly_threshold": 2.0,
            "is_anomalous": False,
        },
    )

    assert isinstance(response, AnomalyDetectionResponse)
    assert response.is_anomalous is False
    plain = response_from_type(
        "classification",
        {
            "image": {"width": 20, "height": 10},
            "predictions": [{"class": "cat", "class_id": 0, "confidence": 0.9}],
            "top": "cat",
            "confidence": 0.9,
        },
    )
    assert isinstance(plain, ClassificationInferenceResponse)
