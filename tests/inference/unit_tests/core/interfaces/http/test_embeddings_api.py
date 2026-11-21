import base64
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch
from starlette.testclient import TestClient

from inference.core.entities.responses.embeddings import ImageEmbeddingResponse
from inference.core.managers.base import ModelManager
from inference.core.models.embeddings import make_embedding_info
from inference.core.registries.lazy import _LazyModelClass, _LazyModelRegistry
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.usage_tracking.model_types import clear_recorded_model_descriptors


@pytest.mark.parametrize("query_auth", [False, True])
@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_embedding_endpoint_negotiates_capability_and_preserves_auth(
    monkeypatch, query_auth, output_type
):
    from inference.core.interfaces.http import http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    manager = MagicMock()
    manager.pingback = None
    manager.num_errors = 0
    manager.infer_from_request_sync.return_value = ImageEmbeddingResponse(
        embeddings=[[2.0, 3.0]],
        embedding_info=make_embedding_info(
            "classifiers/4",
            {
                "feature_definition": "classifier-linear-input@v1",
                "normalization": "none",
            },
            {},
            "onnx",
            "float32",
            2,
        ),
    )
    interface = http_api.HttpInterface(model_manager=manager)
    with TestClient(interface.app) as client:
        payload = {
            "model_id": "resnet101",
            "output_type": output_type,
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
        }
        if not query_auth:
            payload["api_key"] = "key"
        result = client.post(
            "/infer/embeddings",
            params={"api_key": "key"} if query_auth else None,
            json=payload,
        )
    assert result.status_code == 200, result.text
    assert result.json()["embeddings"] == [[2.0, 3.0]]
    assert result.json()["embedding_info"]["model_id"] == "classifiers/4"
    manager.add_model.assert_called_once_with(
        "classifiers/4",
        "key",
        model_id_alias="resnet101",
        countinference=None,
        service_secret=None,
        required_capabilities=["image_embeddings"],
        output_type=output_type,
    )
    assert manager.infer_from_request_sync.call_args.args[
        0
    ] == "resnet101:capabilities=image_embeddings" + (
        ":output_type=logits" if output_type == "logits" else ""
    )

    assert manager.infer_from_request_sync.call_args.args[1].output_type == output_type


@pytest.mark.parametrize(
    "model_id", ["dinov2/small", "dinov3/vits16", "siglip2/base-patch16-224"]
)
def test_embedding_endpoint_negotiates_capability_for_pretrained_models(
    monkeypatch, model_id
):
    from inference.core.interfaces.http import http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    manager = MagicMock()
    manager.pingback = None
    manager.num_errors = 0
    manager.infer_from_request_sync.return_value = ImageEmbeddingResponse(
        embeddings=[[2.0, 3.0]],
        embedding_info=make_embedding_info(
            model_id,
            {"feature_definition": "pretrained@v1", "normalization": "none"},
            {"do_resize": True},
            "PretrainedImageEmbeddingOnnx",
            "torch.float32",
            2,
        ),
    )
    interface = http_api.HttpInterface(model_manager=manager)
    with TestClient(interface.app) as client:
        result = client.post(
            "/infer/embeddings",
            json={
                "model_id": model_id,
                "api_key": "key",
                "image": {"type": "url", "value": "https://example.com/image.jpg"},
            },
        )

    assert result.status_code == 200, result.text
    assert result.json()["embedding_info"]["model_id"] == model_id
    manager.add_model.assert_called_once_with(
        model_id,
        "key",
        model_id_alias=None,
        countinference=None,
        service_secret=None,
        required_capabilities=["image_embeddings"],
        output_type="feature_vector",
    )
    assert (
        manager.infer_from_request_sync.call_args.args[0]
        == f"{model_id}:capabilities=image_embeddings"
    )


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_pretrained_embedding_endpoint_returns_vectors_and_rejects_logits(
    monkeypatch, output_type
):
    from inference.core.interfaces.http import http_api
    from inference.models.pretrained_image_embeddings import (
        pretrained_image_embeddings_inference_models,
    )

    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    backend = MagicMock()
    backend.embed_images.return_value = torch.tensor([[0.5, -1.5, 2.0]])
    backend.get_embedding_info.return_value = {
        "feature_definition": "siglip2-pooler-output@v1",
        "output_type": "feature_vector",
        "normalization": "none",
    }
    backend.embedding_preprocessing = {"do_resize": True}
    load_model = MagicMock(return_value=backend)
    monkeypatch.setattr(
        pretrained_image_embeddings_inference_models.AutoModel,
        "from_pretrained",
        load_model,
    )
    registry = RoboflowModelRegistry(
        _LazyModelRegistry(
            {
                ("embed", "siglip2"): _LazyModelClass(
                    "inference.models.pretrained_image_embeddings."
                    "pretrained_image_embeddings_inference_models:"
                    "InferenceModelsPretrainedImageEmbeddingAdapter"
                )
            }
        )
    )
    manager = ModelManager(registry, content_addressed_artifact_cache=MagicMock())
    _, image = cv2.imencode(".png", np.zeros((8, 8, 3), dtype=np.uint8))
    interface = http_api.HttpInterface(model_manager=manager)
    try:
        with TestClient(interface.app) as client:
            result = client.post(
                "/infer/embeddings",
                json={
                    "model_id": "siglip2/base-patch16-224",
                    "api_key": "key",
                    "output_type": output_type,
                    "image": {
                        "type": "base64",
                        "value": base64.b64encode(image.tobytes()).decode(),
                    },
                },
            )
    finally:
        clear_recorded_model_descriptors()

    if output_type == "logits":
        assert result.status_code == 400, result.text
        assert "feature_vector" in result.json()["message"]
        load_model.assert_not_called()
        return
    assert result.status_code == 200, result.text
    assert result.json()["embeddings"] == [[0.5, -1.5, 2.0]]
    assert result.json()["embedding_info"]["model_id"] == "siglip2/base-patch16-224"
    assert result.json()["embedding_info"]["dimension"] == 3
    assert load_model.call_args.kwargs["required_capabilities"] == ["image_embeddings"]
