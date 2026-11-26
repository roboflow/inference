"""NPY image input through the real `/infer/embeddings` route.

The route, model manager and image loading are real; only the classifier's
computation is replaced, so no weights load.
"""

import base64
import io
import pickle
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
import torch
from starlette.testclient import TestClient

from inference.core.managers.base import ModelManager
from inference.core.models.embeddings import IMAGE_EMBEDDINGS, model_cache_key
from inference.core.models.inference_models_adapters import (
    InferenceModelsClassificationAdapter,
)
from tests.inference.unit_tests.core.interfaces.http.test_workflow_billing_contract import (
    _build_test_client,
)

MODEL_ID = "project/1"


@pytest.fixture
def api_key() -> str:
    return f"npy-http-{uuid4().hex}"


def _classifier() -> InferenceModelsClassificationAdapter:
    backend = MagicMock(image_size=None, _image_size=None, img_size=None, _model=None)
    backend.pre_process.side_effect = lambda images, **kwargs: torch.zeros(
        len(images), 3, 224, 224
    )
    backend.forward_embedding.side_effect = lambda inputs, output_type: torch.ones(
        len(inputs), 2
    )
    backend.get_embedding_info.return_value = {
        "feature_definition": "classifier-linear-input@v1",
        "output_type": "feature_vector",
        "normalization": "none",
    }
    backend._inference_config.image_pre_processing.model_dump.return_value = {}
    backend._inference_config.network_input.model_dump.return_value = {}
    model = InferenceModelsClassificationAdapter.__new__(
        InferenceModelsClassificationAdapter
    )
    model._model = backend
    model._embedding_model_id = MODEL_ID
    model.api_key = "key-that-loaded-the-model"
    model.task_type = "classification"
    model.img_size_h = 224
    model.img_size_w = 224

    return model


def _client(
    monkeypatch: pytest.MonkeyPatch, model: InferenceModelsClassificationAdapter
) -> TestClient:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    manager = ModelManager(
        MagicMock(), models={model_cache_key(MODEL_ID, [IMAGE_EMBEDDINGS]): model}
    )
    client = _build_test_client(monkeypatch, model_manager=manager)

    return client


def _post_image(client: TestClient, api_key: str, image_type: str, data: bytes):
    response = client.post(
        "/infer/embeddings",
        json={
            "model_id": MODEL_ID,
            "api_key": api_key,
            "image": {
                "type": image_type,
                "value": base64.b64encode(data).decode("ascii"),
            },
        },
    )

    return response


def _npy(image: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, image, allow_pickle=False)
    data = buffer.getvalue()

    return data


@pytest.mark.parametrize("fortran_order", [False, True])
def test_embeddings_accept_npy(monkeypatch, api_key, fortran_order):
    image = np.arange(105, dtype=np.uint8).reshape(5, 7, 3)
    sent = np.asfortranarray(image) if fortran_order else image
    model = _classifier()

    response = _post_image(_client(monkeypatch, model), api_key, "npy", _npy(sent))

    assert response.status_code == 200, response.text
    assert len(response.json()["embeddings"]) == 1
    model._model.pre_process.assert_called_once()
    forwarded = model._model.pre_process.call_args.args[0][0]
    assert forwarded.flags.c_contiguous
    np.testing.assert_array_equal(forwarded, image)


def test_embeddings_refuse_pickled_numpy_without_unpickling(monkeypatch, api_key):
    unpickle = MagicMock(
        side_effect=AssertionError("request data must never be unpickled")
    )
    monkeypatch.setattr(pickle, "loads", unpickle)
    data = pickle.dumps(np.arange(105, dtype=np.uint8).reshape(5, 7, 3))
    model = _classifier()

    response = _post_image(_client(monkeypatch, model), api_key, "numpy", data)

    assert response.status_code == 400, response.text
    assert "'npy'" in response.json()["message"]
    unpickle.assert_not_called()
    model._model.forward_embedding.assert_not_called()


def test_embeddings_refuse_a_pickle_declared_as_npy(monkeypatch, api_key):
    data = pickle.dumps(np.zeros((5, 7, 3), dtype=np.uint8))
    model = _classifier()

    response = _post_image(_client(monkeypatch, model), api_key, "npy", data)

    assert response.status_code == 400, response.text
    model._model.pre_process.assert_not_called()


_MALFORMED_HEADERS = {
    "unterminated-brace": b"{",
    "truncated-dict": b"{'descr': '|u1', 'fortran_order': False, 'shape': (2,2),",
    "deep-nesting": b"[" * 3000,
    "binary": b"\x00\xff",
    "bad-syntax": b"{'descr': '|u1' 'x'}",
    "not-a-dict": b"(1, 2)",
    "empty-tuple-descr": b"{'descr': (), 'fortran_order': False, 'shape': (1, 1)}",
    "incomplete-tuple-descr": (
        b"{'descr': ('|u1',), 'fortran_order': False, 'shape': (1, 1)}"
    ),
    "list-of-garbage-descr": b"{'descr': [()], 'fortran_order': False, 'shape': (1, 1)}",
    "missing-keys": b"{'shape': (1, 1)}",
}


@pytest.mark.parametrize("version", [(1, 0), (2, 0)], ids=["v1", "v2"])
@pytest.mark.parametrize("name", list(_MALFORMED_HEADERS))
def test_embeddings_malformed_npy_header_returns_400(
    monkeypatch, api_key, version, name
):
    header = _MALFORMED_HEADERS[name]
    size = len(header).to_bytes(2 if version == (1, 0) else 4, "little")
    data = b"\x93NUMPY" + bytes(version) + size + header
    model = _classifier()

    response = _post_image(_client(monkeypatch, model), api_key, "npy", data)

    assert response.status_code == 400, response.text
    model._model.forward_embedding.assert_not_called()


@pytest.mark.parametrize("channels_axis", [False, True])
def test_embeddings_grayscale_npy_reaches_the_model_as_bgr(
    monkeypatch, api_key, channels_axis
):
    gray = np.arange(35, dtype=np.uint8).reshape(5, 7)
    image = gray[..., None] if channels_axis else gray
    model = _classifier()

    response = _post_image(_client(monkeypatch, model), api_key, "npy", _npy(image))

    assert response.status_code == 200, response.text
    forwarded = model._model.pre_process.call_args.args[0][0]
    np.testing.assert_array_equal(forwarded, np.repeat(gray[..., None], 3, axis=2))
