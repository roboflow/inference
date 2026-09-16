from unittest.mock import Mock

import base64
from io import BytesIO
from PIL import Image
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typing import Union

from inference.core.entities.requests.inference import ClassificationInferenceRequest
from inference.core.entities.responses.inference import (
    AnomalyDetectionResponse,
    ClassificationInferenceResponse,
)
from inference.models.anomaly_detection.model import (
    PatchCoreAnomalyDetection,
    make_anomaly_response,
)
from inference_sdk.http.entities import InferenceConfiguration


def prediction(score=2.0, threshold=2.0):
    return dict(
        anomaly_score=score,
        anomaly_threshold=threshold,
        is_anomalous=score >= threshold,
        anomalous_confidence=0.5,
    )


def test_boundary_decision_is_consistent_with_classification_predictions():
    result = make_anomaly_response(prediction(), 20, 10, 0.1)
    assert result.top == "anomalous"
    assert result.predictions[0].class_id == 1
    assert result.predictions[1].class_id == 0
    assert result.confidence == 0.5


def test_http_serialization_keeps_anomaly_fields():
    app = FastAPI()

    @app.get(
        "/test",
        response_model=Union[AnomalyDetectionResponse, ClassificationInferenceResponse],
    )
    def predict():
        return make_anomaly_response(prediction(), 20, 10, 0.1)

    body = TestClient(app).get("/test").json()
    assert body["anomaly_score"] == 2.0
    assert body["is_anomalous"] is True
    assert body["anomaly_threshold"] == 2.0


def test_request_and_sdk_forward_optional_map_to_shared_runtime():
    model = object.__new__(PatchCoreAnomalyDetection)
    model.detector = Mock()
    model.detector.predict.return_value = [prediction()]
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
    assert model.detector.predict.call_args.kwargs == {"include_maps": True}
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


def test_loads_trainer_artifact_and_preserves_raw_scores(tmp_path, monkeypatch):
    import torch
    from torch import nn

    pytest.importorskip("roboflow_anomaly")
    from roboflow_anomaly import AnomalyConfig, AnomalyDetector
    from roboflow_anomaly.adapters import patchcore
    from roboflow_anomaly.data import Sample
    from inference.core.models.roboflow import RoboflowInferenceModel
    from inference.models.anomaly_detection.model import AnomalyDetectionModel

    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer2 = nn.Conv2d(3, 8, 3, stride=4, padding=1)
            self.layer3 = nn.Conv2d(8, 16, 3, stride=2, padding=1)

        def forward(self, images):
            return self.layer3(self.layer2(images))

    torch.set_num_threads(1)
    monkeypatch.setattr(patchcore, "wide_resnet50_2", lambda **kwargs: TinyBackbone())
    normal = tmp_path / "normal.png"
    Image.new("RGB", (20, 10), (80, 130, 230)).save(normal)
    sample = Sample(normal, 0)
    trainer_model = AnomalyDetector(
        AnomalyConfig(architecture="patchcore", image_size=32), pretrained=False
    )
    trainer_model.fit([sample])
    trainer_model.calibrate([sample])
    trainer_model.save(tmp_path / "best.pt")

    # Only the existing remote artifact/cache boundary is replaced. Loading, decoding,
    # preprocessing, official scoring, and classification serialization all run normally.
    def initialize_cache(self, *args, **kwargs):
        self.cache_dir = str(tmp_path)

    monkeypatch.setattr(RoboflowInferenceModel, "__init__", initialize_cache)
    monkeypatch.setattr(
        AnomalyDetectionModel, "get_model_artifacts", lambda self, **kwargs: None
    )
    serving_model = PatchCoreAnomalyDetection("defects/1")
    request = ClassificationInferenceRequest(
        model_id="defects/1",
        image={
            "type": "base64",
            "value": base64.b64encode(normal.read_bytes()).decode(),
        },
        include_anomaly_map=True,
    )
    response = serving_model.infer_from_request(request)
    with Image.open(normal) as image:
        expected = trainer_model.predict([image])[0]
    assert response.anomaly_score == pytest.approx(expected["anomaly_score"], abs=1e-6)
    assert response.anomaly_threshold == expected["anomaly_threshold"]
    assert response.is_anomalous == expected["is_anomalous"]
    assert response.top == "normal"
    assert len(response.anomaly_map) == 10
    assert len(response.anomaly_map[0]) == 20
