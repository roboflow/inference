import base64
from io import BytesIO
from typing import Union
from unittest.mock import Mock

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from inference.core.entities.requests.inference import ClassificationInferenceRequest
from inference.core.entities.responses.inference import (
    AnomalyDetectionResponse,
    ClassificationInferenceResponse,
)
from inference.core.models.inference_models_adapters import (
    InferenceModelsAnomalyDetectionAdapter,
    prepare_anomaly_detection_response,
)
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference_models import ClassificationPrediction
from inference_sdk.http.entities import InferenceConfiguration

CLASS_NAMES = ["normal", "anomalous"]


def prediction(anomalous_confidence=0.5, is_anomalous=True, anomaly_map=None):
    metadata = {
        "anomaly_score": 2.0,
        "anomaly_threshold": 2.0,
        "is_anomalous": is_anomalous,
    }
    if anomaly_map is not None:
        metadata["anomaly_map"] = anomaly_map
    return ClassificationPrediction(
        class_id=torch.tensor([int(is_anomalous)]),
        confidence=torch.tensor([[1 - anomalous_confidence, anomalous_confidence]]),
        images_metadata=[metadata],
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


def test_boundary_decision_leads_the_classification_predictions():
    result = prepare_anomaly_detection_response(
        prediction(), image_sizes=[(10, 20)], class_names=CLASS_NAMES
    )[0]
    assert result.top == "anomalous"
    assert result.predictions[0].class_id == 1
    assert result.predictions[1].class_id == 0
    assert result.confidence == 0.5
    assert result.image.width == 20
    assert result.image.height == 10
    assert result.anomaly_map is None


def test_low_confidence_class_is_retained():
    result = prepare_anomaly_detection_response(
        prediction(anomalous_confidence=0.01, is_anomalous=False),
        image_sizes=[(10, 20)],
        class_names=CLASS_NAMES,
    )[0]
    assert result.top == "normal"
    assert [p.class_name for p in result.predictions] == ["normal", "anomalous"]
    assert result.predictions[1].confidence == 0.01


def test_http_serialization_keeps_anomaly_fields():
    app = FastAPI()

    @app.get(
        "/test",
        response_model=Union[AnomalyDetectionResponse, ClassificationInferenceResponse],
    )
    def predict():
        return prepare_anomaly_detection_response(
            prediction(anomaly_map=np.ones((10, 20), dtype=np.float32)),
            image_sizes=[(10, 20)],
            class_names=CLASS_NAMES,
        )[0]

    body = TestClient(app).get("/test").json()
    assert body["anomaly_score"] == 2.0
    assert body["is_anomalous"] is True
    assert body["anomaly_threshold"] == 2.0
    assert len(body["anomaly_map"]) == 10
    assert len(body["anomaly_map"][0]) == 20


def test_request_and_sdk_forward_optional_map_to_the_model():
    model = object.__new__(InferenceModelsAnomalyDetectionAdapter)
    model.class_names = CLASS_NAMES
    model._model = Mock()
    model._model.post_process.return_value = prediction()
    image = BytesIO()
    Image.new("RGB", (20, 10)).save(image, format="PNG")
    request = ClassificationInferenceRequest(
        model_id="defects/1",
        image={"type": "base64", "value": base64.b64encode(image.getvalue()).decode()},
        include_anomaly_map=True,
    )
    response = model.infer_from_request(request)
    assert response.top == "anomalous"
    assert response.image.width == 20
    assert response.image.height == 10
    bgr_images = model._model.pre_process.call_args.args[0]
    assert bgr_images[0].shape == (10, 20, 3)
    assert model._model.pre_process.call_args.kwargs["input_color_format"] == "bgr"
    assert model._model.post_process.call_args.kwargs == {"include_anomaly_map": True}
    assert (
        InferenceConfiguration(include_anomaly_map=True).to_classification_parameters()[
            "include_anomaly_map"
        ]
        is True
    )
    assert (
        "include_anomaly_map"
        not in InferenceConfiguration().to_classification_parameters()
    )
