from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock

import numpy as np
import pytest
from starlette.testclient import TestClient

from inference.core.entities.responses.inference import (
    InferenceResponseImage,
    ObjectDetectionInferenceResponse,
)
from inference.core.interfaces.http import http_api
from inference.core.managers import base as manager_module
from inference.core.managers.base import ModelManager
from inference.core.models import inference_models_adapters as adapters
from inference.core.registries.base import ModelRegistry
from inference_models.entities import ResolvedModelMetadata
from inference_models.utils.content_addressed_artifact_cache import (
    NullContentAddressedArtifactCache,
)
from inference_sdk import InferenceHTTPClient

RESOLVED_MODEL = ResolvedModelMetadata(
    model_id="canonical/1",
    model_package_id="onnxpackage",
    backend="onnx",
    quantization="fp32",
)


class PackageModel(adapters.InferenceModelsAdapter):
    task_type = "object-detection"
    model_id = "test/1"

    def __init__(self, metadata: Optional[ResolvedModelMetadata] = RESOLVED_MODEL):
        self._model = SimpleNamespace()
        if metadata is not None:
            self._model.resolved_model = metadata

    def infer(self, image, **kwargs):
        response = ObjectDetectionInferenceResponse(
            image=InferenceResponseImage(width=640, height=480), predictions=[]
        )
        images = image if isinstance(image, list) else [image]
        return [response.model_copy() for _ in images]


def build_client(monkeypatch, flag=True, model=None, registry=None):
    monkeypatch.setattr(adapters, "USE_INFERENCE_MODELS", flag)
    monkeypatch.setattr(manager_module, "USE_INFERENCE_MODELS", flag)
    monkeypatch.setattr(manager_module, "MODELS_CACHE_AUTH_ENABLED", False)
    monkeypatch.setattr(manager_module, "DISABLE_INFERENCE_CACHE", True)
    monkeypatch.setattr(http_api, "USE_INFERENCE_MODELS", flag)
    monkeypatch.setattr(http_api, "MAX_INFERENCE_MODELS_CACHE_SIZE_MB", 0)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(
        http_api,
        "InferenceInstrumentator",
        lambda *args, **kwargs: SimpleNamespace(
            set_stream_manager_client=lambda client: None
        ),
    )
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    manager = ModelManager(
        model_registry=registry or ModelRegistry({}),
        models={"test/1": model or PackageModel()},
        content_addressed_artifact_cache=NullContentAddressedArtifactCache(),
    )
    return TestClient(http_api.HttpInterface(model_manager=manager).app), manager


def test_http_inference_reports_the_serving_package(monkeypatch):
    client, _ = build_client(monkeypatch)

    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "test/1",
            "image": {"type": "base64", "value": "image"},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["resolved_model"] == {
        "model_id": "canonical/1",
        "model_package_id": "onnxpackage",
        "backend": "onnx",
        "quantization": "fp32",
    }


def test_legacy_http_inference_reports_the_serving_package(monkeypatch):
    client, _ = build_client(monkeypatch)

    response = client.post(
        "/test/1",
        content="image",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["resolved_model"]["model_package_id"] == "onnxpackage"


def test_http_batch_reports_package_for_each_image(monkeypatch):
    client, _ = build_client(monkeypatch)

    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "test/1",
            "image": [
                {"type": "base64", "value": "first"},
                {"type": "base64", "value": "second"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert [item["resolved_model"]["model_package_id"] for item in response.json()] == [
        "onnxpackage",
        "onnxpackage",
    ]


def test_http_inference_omits_metadata_when_flag_is_disabled(monkeypatch):
    client, _ = build_client(monkeypatch, flag=False)

    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "test/1",
            "image": {"type": "base64", "value": "image"},
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.pop("time") >= 0
    payload.pop("inference_id")
    assert payload == {
        "image": {"width": 640, "height": 480},
        "predictions": [],
    }


@pytest.mark.parametrize("api_version", ["v0", "v1"])
def test_sdk_preserves_http_package_metadata(monkeypatch, requests_mock, api_version):
    server, _ = build_client(monkeypatch)
    sdk = InferenceHTTPClient(api_url="http://testserver", api_key="test")
    getattr(sdk, f"select_api_{api_version}")()

    def infer(request, context):
        response = server.post(
            request.path_url,
            content=request.body,
            headers={"Content-Type": request.headers["Content-Type"]},
        )
        context.status_code = response.status_code
        return response.json()

    requests_mock.get(
        "http://testserver/model/registry",
        json=server.get("/model/registry").json(),
    )
    endpoint = "/test/1" if api_version == "v0" else "/infer/object_detection"
    requests_mock.post(f"http://testserver{endpoint}", json=infer)

    response = sdk.infer(np.zeros((480, 640, 3), dtype=np.uint8), model_id="test/1")

    assert isinstance(response, dict)
    assert response["resolved_model"] == {
        "model_id": "canonical/1",
        "model_package_id": "onnxpackage",
        "backend": "onnx",
        "quantization": "fp32",
    }


def test_http_response_keeps_serving_package_when_model_is_replaced(monkeypatch):
    replacement = PackageModel(
        ResolvedModelMetadata(
            model_id="canonical/1",
            model_package_id="trtpackage",
            backend="trt",
            quantization="fp16",
        )
    )
    registry = SimpleNamespace(
        get_model=lambda *args, **kwargs: lambda **kwargs: replacement
    )

    class EvictedModel(PackageModel):
        def infer_from_request(self, request):
            manager.remove("test/1", delete_from_disk=False)
            manager.add_model("test/1", api_key="test")
            return super().infer_from_request(request)

    client, manager = build_client(monkeypatch, model=EvictedModel(), registry=registry)
    request = {
        "model_id": "test/1",
        "image": {"type": "base64", "value": "image"},
    }

    first = client.post("/infer/object_detection", json=request)
    second = client.post("/infer/object_detection", json=request)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["resolved_model"]["model_package_id"] == "onnxpackage"
    assert second.json()["resolved_model"]["model_package_id"] == "trtpackage"


def test_http_inference_omits_metadata_for_a_model_without_package_identity(
    monkeypatch,
):
    client, _ = build_client(monkeypatch, model=PackageModel(metadata=None))

    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "test/1",
            "image": {"type": "base64", "value": "image"},
        },
    )

    assert response.status_code == 200, response.text
    assert "resolved_model" not in response.json()


def test_direct_adapter_response_reports_its_package(monkeypatch):
    from inference.core.entities.requests.inference import (
        InferenceRequestImage,
        ObjectDetectionInferenceRequest,
    )

    monkeypatch.setattr(adapters, "USE_INFERENCE_MODELS", True)
    response = PackageModel().infer_from_request(
        ObjectDetectionInferenceRequest(
            id="direct-request",
            model_id="test/1",
            image=InferenceRequestImage(type="base64", value="image"),
        )
    )

    assert isinstance(response, ObjectDetectionInferenceResponse)
    assert response.resolved_model is not None
    assert response.resolved_model.model_package_id == "onnxpackage"


def test_classification_adapter_reports_package_with_predictions(monkeypatch):
    import base64

    import cv2
    import torch

    from inference.core.entities.requests.inference import (
        ClassificationInferenceRequest,
        InferenceRequestImage,
    )
    from inference.core.entities.responses.inference import (
        ClassificationInferenceResponse,
    )
    from inference_models import ClassificationPrediction

    class ClassificationBackend:
        class_names = ["cat", "dog"]
        resolved_model = RESOLVED_MODEL

        def pre_process(self, images, **kwargs):
            return np.stack(images)

        def forward(self, images, **kwargs):
            return torch.tensor([[0.9, 0.1]] * len(images))

        def post_process(self, predictions, **kwargs):
            return ClassificationPrediction(
                class_id=predictions.argmax(dim=-1), confidence=predictions
            )

    monkeypatch.setattr(adapters, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(
        adapters.AutoModel, "from_pretrained", lambda **kwargs: ClassificationBackend()
    )
    model = adapters.InferenceModelsClassificationAdapter("test/1", api_key="test")
    _, image = cv2.imencode(".png", np.zeros((4, 5, 3), dtype=np.uint8))

    response = model.infer_from_request(
        ClassificationInferenceRequest(
            id="classification-request",
            model_id="test/1",
            image=InferenceRequestImage(
                type="base64", value=base64.b64encode(image).decode()
            ),
        )
    )

    assert isinstance(response, ClassificationInferenceResponse)
    assert response.top == "cat"
    assert response.resolved_model is not None
    assert response.resolved_model.model_package_id == "onnxpackage"
