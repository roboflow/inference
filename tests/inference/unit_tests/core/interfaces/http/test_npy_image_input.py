import base64
import io
import pickle
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from inference.core.utils import image_utils
from tests.inference.unit_tests.core.interfaces.http.test_embeddings_billing_contract import (
    MODEL_ID,
    _hosted_client,
    _inference_models_adapter,
)


@pytest.fixture
def api_key():
    return f"npy-http-{uuid4().hex}"


@pytest.mark.parametrize("enabled", [False, True])
def test_embeddings_accept_npy(monkeypatch, api_key, enabled):
    monkeypatch.setattr(image_utils, "ALLOW_NUMPY_INPUT", enabled)
    image = np.arange(105, dtype=np.uint8).reshape(5, 7, 3)
    buffer = io.BytesIO()
    np.save(buffer, image, allow_pickle=False)
    model = _inference_models_adapter()
    client = _hosted_client(monkeypatch, model)
    request = {
        "model_id": MODEL_ID,
        "api_key": api_key,
        "image": {"type": "npy", "value": base64.b64encode(buffer.getvalue()).decode("ascii")},
    }

    response = client.post("/infer/embeddings", json=request)

    assert response.status_code == 200, response.text
    assert len(response.json()["embeddings"]) == 1
    model._model.pre_process.assert_called_once()
    np.testing.assert_array_equal(model._model.pre_process.call_args.args[0][0], image)


@pytest.mark.parametrize("enabled", [False, True])
def test_embeddings_refuse_pickled_numpy_without_unpickling(monkeypatch, api_key, enabled):
    monkeypatch.setattr(image_utils, "ALLOW_NUMPY_INPUT", enabled)
    unpickle = MagicMock(side_effect=AssertionError("request data must never be unpickled"))
    monkeypatch.setattr(pickle, "loads", unpickle)
    data = pickle.dumps(np.arange(105, dtype=np.uint8).reshape(5, 7, 3))
    model = _inference_models_adapter()
    client = _hosted_client(monkeypatch, model)

    response = client.post(
        "/infer/embeddings",
        json={
            "model_id": MODEL_ID,
            "api_key": api_key,
            "image": {"type": "numpy", "value": base64.b64encode(data).decode("ascii")},
        },
    )

    assert response.status_code == 400, response.text
    unpickle.assert_not_called()
    model._model.forward_embedding.assert_not_called()


@pytest.mark.parametrize("kind", ["npy", "numpy"])
def test_embeddings_reject_pickle_without_legacy_opt_in(monkeypatch, api_key, kind):
    monkeypatch.setattr(image_utils, "ALLOW_NUMPY_INPUT", False)
    model = _inference_models_adapter()
    client = _hosted_client(monkeypatch, model)
    data = pickle.dumps(np.zeros((5, 7, 3), dtype=np.uint8))

    response = client.post(
        "/infer/embeddings",
        json={
            "model_id": MODEL_ID,
            "api_key": api_key,
            "image": {"type": kind, "value": base64.b64encode(data).decode("ascii")},
        },
    )

    assert response.status_code == 400, response.text
    model._model.pre_process.assert_not_called()
