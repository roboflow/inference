"""Contract: image embeddings write the usage row hosted billing counts.

Hosted Inference bills per image from `model` rows. Embeddings skip
`BaseInference.infer`, so `infer_embeddings_from_request` records that row
itself. Here the real route, model manager and collector run on a serverless
interface; only the model's computation is replaced, so no weights load.
"""

import base64
import io
import json
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import numpy as np
import pytest
import torch
from PIL import Image
from starlette.testclient import TestClient

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.interfaces.workflows_models_provider import (
    ModelManagerModelsProvider,
)
from inference.core.managers.base import ModelManager
from inference.core.models.embeddings import (
    IMAGE_EMBEDDINGS,
    make_embedding_info,
    model_cache_key,
)
from inference.core.models.inference_models_adapters import (
    InferenceModelsClassificationAdapter,
)
from inference.core.roboflow_api import ServerlessUsageCheckResponse
from inference.models.resnet.resnet_classification import ResNetClassification
from inference.usage_tracking.megapixel_buckets import record_measured_model_hw
from tests.inference.unit_tests.core.interfaces.http.test_workflow_billing_contract import (
    SERVICE_SECRET,
    _billable_by_category,
    _build_test_client,
    _rows_for_api_key,
)

MODEL_ID = "project/1"
MODEL_KEY = model_cache_key(MODEL_ID, [IMAGE_EMBEDDINGS])
LOADER_API_KEY = "key-that-loaded-the-model"
FEATURE_INFO = {
    "feature_definition": "classifier-linear-input@v1",
    "output_type": "feature_vector",
    "normalization": "none",
}


def _inference_models_adapter() -> InferenceModelsClassificationAdapter:
    """The classifier hosted Inference runs, with a fake backend."""
    backend = MagicMock(image_size=None, _image_size=None, img_size=None, _model=None)
    backend.pre_process.side_effect = lambda images, **kwargs: torch.zeros(
        len(images), 3, 224, 224
    )
    backend.forward_embedding.side_effect = lambda inputs, output_type: torch.ones(
        len(inputs), 2
    )
    backend.get_embedding_info.return_value = FEATURE_INFO
    backend._inference_config.image_pre_processing.model_dump.return_value = {}
    backend._inference_config.network_input.model_dump.return_value = {}
    model = InferenceModelsClassificationAdapter.__new__(
        InferenceModelsClassificationAdapter
    )
    model._model = backend
    model._embedding_model_id = MODEL_ID
    model.api_key = LOADER_API_KEY
    model.task_type = "classification"
    model.img_size_h = 224
    model.img_size_w = 224

    return model


def _fake_embedding_arrays(request: ImageEmbeddingRequest) -> tuple:
    image_count = len(request.image) if isinstance(request.image, list) else 1
    info = make_embedding_info(MODEL_ID, FEATURE_INFO, {}, "onnx", "float32", 2)

    return np.ones((image_count, 2), dtype=np.float32), info


def _onnx_classifier() -> ResNetClassification:
    """The legacy classifier, with its ONNX session replaced."""
    model = ResNetClassification.__new__(ResNetClassification)
    model._infer_embedding_arrays = _fake_embedding_arrays
    model.dataset_id = "project"
    model.version_id = "1"
    model.api_key = LOADER_API_KEY
    model.img_size_h = 224
    model.img_size_w = 224

    return model


@pytest.fixture(
    params=[_inference_models_adapter, _onnx_classifier],
    ids=["inference_models", "onnx"],
)
def embedding_model(request: pytest.FixtureRequest) -> Any:
    return request.param()


@pytest.fixture
def api_key() -> str:
    # Rows outlive a test: the client holds the collector's flush, and every test
    # on an xdist worker shares that collector.
    return f"embeddings-billing-contract-{uuid4().hex}"


def _hosted_client(monkeypatch: pytest.MonkeyPatch, model: Any) -> TestClient:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", True)
    monkeypatch.setattr(http_api, "OFFLINE_MODE", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(
        http_api,
        "get_serverless_usage_check_async",
        AsyncMock(
            return_value=ServerlessUsageCheckResponse(
                status_code=200, workspace_id="workspace", under_cap=True
            )
        ),
    )
    monkeypatch.setattr(
        http_api, "get_roboflow_workspace_async", AsyncMock(return_value="workspace")
    )
    manager = ModelManager(MagicMock(), models={MODEL_KEY: model})
    client = _build_test_client(monkeypatch, model_manager=manager)

    return client


def _image() -> dict:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="JPEG")
    value = base64.b64encode(buffer.getvalue()).decode()

    return {"type": "base64", "value": value}


def _model_rows(api_key: str) -> List[dict]:
    rows = [
        row for row in _rows_for_api_key(api_key).values() if row["category"] == "model"
    ]

    return rows


@pytest.mark.parametrize(
    "images, frames",
    [(_image(), 1), ([_image(), _image(), _image()], 3)],
    ids=["single_image", "image_list"],
)
def test_every_image_lands_on_one_billable_model_row(
    monkeypatch, embedding_model, api_key, images, frames
):
    # given
    client = _hosted_client(monkeypatch, embedding_model)

    # when
    response = client.post(
        "/infer/embeddings",
        params={"api_key": api_key},
        json={"model_id": MODEL_ID, "image": images},
    )

    # then
    assert response.status_code == 200, response.text
    assert len(response.json()["embeddings"]) == frames
    assert _billable_by_category(api_key) == {"request": True, "model": True}
    (row,) = _model_rows(api_key)
    assert row["processed_frames"] == frames
    assert row["resource_id"] == MODEL_ID
    details = json.loads(row["resource_details"])
    assert details["task_type"] == "classification"
    assert details["model_input_height"] == 224
    assert details["model_input_width"] == 224
    assert (
        sum(bucket["processed_frames"] for bucket in row["megapixel_buckets"].values())
        == frames
    )


def test_authenticated_opt_out_reaches_the_model_row(
    monkeypatch, embedding_model, api_key
):
    # given
    client = _hosted_client(monkeypatch, embedding_model)

    # when
    response = client.post(
        "/infer/embeddings",
        params={
            "api_key": api_key,
            "countinference": "false",
            "service_secret": SERVICE_SECRET,
        },
        json={"model_id": MODEL_ID, "image": _image()},
    )

    # then
    assert response.status_code == 200, response.text
    assert _billable_by_category(api_key) == {"request": False, "model": False}


def test_header_api_key_owns_the_row_not_the_key_that_loaded_the_model(
    monkeypatch, embedding_model, api_key
):
    # given
    client = _hosted_client(monkeypatch, embedding_model)

    # when
    response = client.post(
        "/infer/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model_id": MODEL_ID, "image": _image()},
    )

    # then
    assert response.status_code == 200, response.text
    assert len(_model_rows(api_key)) == 1
    assert _model_rows(LOADER_API_KEY) == []


def test_a_failed_embedding_call_still_records_its_model_row(monkeypatch, api_key):
    # given
    model = _inference_models_adapter()
    model._model.forward_embedding.side_effect = RuntimeError("out of memory")
    client = _hosted_client(monkeypatch, model)

    # when
    response = client.post(
        "/infer/embeddings",
        params={"api_key": api_key},
        json={"model_id": MODEL_ID, "image": _image()},
    )

    # then
    assert response.status_code == 500
    (row,) = _model_rows(api_key)
    assert json.loads(row["resource_details"])["error_type"] == "RuntimeError"


def test_a_key_without_access_cannot_embed_with_a_cached_model(monkeypatch, api_key):
    # given
    import inference.core.managers.base as model_manager_module

    client = _hosted_client(monkeypatch, _inference_models_adapter())
    monkeypatch.setattr(model_manager_module, "MODELS_CACHE_AUTH_ENABLED", True)
    access_check = MagicMock(return_value=False)
    monkeypatch.setattr(
        model_manager_module, "_check_if_api_key_has_access_to_model", access_check
    )

    # when
    response = client.post(
        "/infer/embeddings",
        params={"api_key": api_key},
        json={"model_id": MODEL_ID, "image": _image()},
    )

    # then
    assert response.status_code == 401
    assert access_check.call_args.kwargs["api_key"] == api_key
    assert access_check.call_args.kwargs["model_id"] == MODEL_ID
    assert _model_rows(api_key) == []


def test_a_list_above_the_serverless_limit_bills_no_model_row(monkeypatch, api_key):
    # given
    import inference.core.interfaces.http.http_api as http_api

    client = _hosted_client(monkeypatch, _inference_models_adapter())
    monkeypatch.setattr(http_api, "SERVERLESS_IMAGE_EMBEDDINGS_MAX_BATCH_SIZE", 2)

    # when
    response = client.post(
        "/infer/embeddings",
        params={"api_key": api_key},
        json={"model_id": MODEL_ID, "image": [_image(), _image(), _image()]},
    )

    # then
    assert response.status_code == 413
    assert _model_rows(api_key) == []


def test_a_local_workflow_step_records_one_model_row(
    monkeypatch, embedding_model, api_key
):
    # given
    _build_test_client(monkeypatch)
    manager = ModelManager(MagicMock(), models={MODEL_KEY: embedding_model})
    provider = ModelManagerModelsProvider(manager)
    image = {"type": "numpy_object", "value": np.zeros((4, 4, 3), dtype=np.uint8)}

    # when
    response = provider.run_image_embeddings(
        model_id=MODEL_ID, images=[image, image], api_key=api_key
    )

    # then
    assert len(response["embeddings"]) == 2
    (row,) = _model_rows(api_key)
    assert row["processed_frames"] == 2
    assert json.loads(row["resource_details"])["source"] == "workflow-execution"


def test_an_earlier_call_input_size_is_not_attributed_to_the_row(
    monkeypatch, embedding_model, api_key
):
    # given
    _build_test_client(monkeypatch)
    del embedding_model.img_size_h
    del embedding_model.img_size_w
    request = ImageEmbeddingRequest(api_key=api_key, model_id=MODEL_ID, image=_image())
    record_measured_model_hw(height=4000, width=4000)

    # when
    embedding_model.infer_embeddings_from_request(request)

    # then
    (row,) = _model_rows(api_key)
    assert list(row["megapixel_buckets"]) == ["unknown"]
