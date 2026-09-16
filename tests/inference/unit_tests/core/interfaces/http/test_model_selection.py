from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel
from starlette.testclient import TestClient

from inference.core.interfaces.http import http_api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "OFFLINE_MODE", True)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "DEPTH_ESTIMATION_ENABLED", True)
    interface = http_api.HttpInterface(model_manager=MagicMock())
    with TestClient(interface.app) as client:
        yield client


@pytest.mark.parametrize("selector", [{"backend": "trt"}, {"quantization": "fp16"}])
def test_http_rejects_package_id_combined_with_other_selectors(client, selector):
    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "project/1",
            "model_package_id": "package-1",
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
            **selector,
        },
    )
    assert response.status_code == 422
    assert "model_package_id" in response.text


@pytest.mark.parametrize("path", ["/infer/object_detection", "/model/add"])
def test_http_rejects_selection_when_flag_is_disabled(client, monkeypatch, path):
    from inference.core import env

    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", False)
    response = client.post(
        path,
        json={
            "model_id": "project/1",
            "backend": "trt",
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
        },
    )
    assert response.status_code == 422
    assert "USE_INFERENCE_MODELS" in response.text


@pytest.mark.parametrize(
    "path", ["/infer/depth-estimation", "/infer/depth-estimation/project/1"]
)
@pytest.mark.parametrize(
    "selectors",
    [{"backend": "trt"}, {"quantization": "fp16"}, {"model_package_id": "engine-1"}],
)
def test_depth_rejects_model_package_selection(client, monkeypatch, path, selectors):
    from inference.core import env

    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", True)
    response = client.post(
        path,
        json={
            "model_id": "project/1",
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
            **selectors,
        },
    )
    assert response.status_code == 422
    assert "not supported for depth estimation" in response.text


@pytest.fixture
def package_client(monkeypatch):
    from types import SimpleNamespace

    from inference.core import env
    from inference.core.managers import base
    from inference.core.managers.base import ModelManager
    from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache

    class PackageResponse(BaseModel):
        served_package: str

    class PackageModel:
        task_type = "object-detection"
        batch_size = 1
        img_size_h = 32
        img_size_w = 32

        def __init__(
            self, model_id, api_key, backend=None, model_package_id=None, **kwargs
        ):
            if api_key == "other-workspace" and model_package_id == "engine-1":
                from inference.core.exceptions import RoboflowAPINotAuthorizedError

                raise RoboflowAPINotAuthorizedError(
                    "Package is not accessible to this workspace."
                )
            selected_backend = backend or (
                "trt" if model_package_id == "engine-1" else "onnx"
            )
            self.resolved_model = SimpleNamespace(
                model_id=model_id,
                model_package_id="engine-1" if selected_backend == "trt" else "onnx-1",
                backend=selected_backend,
                quantization="fp16" if selected_backend == "trt" else "fp32",
            )

        def infer_from_request(self, request):
            return PackageResponse(served_package=self.resolved_model.model_package_id)

        def clear_cache(self, delete_from_disk=True):
            pass

    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(base, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(base, "OFFLINE_MODE", True)
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "OFFLINE_MODE", True)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    registry = MagicMock()
    monkeypatch.setattr(http_api, "ALLOW_ORIGINS", ["https://client.example"])
    registry.get_model.return_value = PackageModel
    manager = WithFixedSizeCache(
        ModelManager(registry, content_addressed_artifact_cache=MagicMock()), max_size=3
    )
    interface = http_api.HttpInterface(model_manager=manager)
    with TestClient(interface.app) as client:
        yield client


@pytest.mark.parametrize(
    "selectors",
    [{"backend": "trt", "quantization": "fp16"}, {"model_package_id": "engine-1"}],
)
def test_http_selected_package_coexists_with_automatic_package(
    package_client, selectors
):
    payload = {
        "model_id": "project/1",
        "image": {"type": "url", "value": "https://example.com/image.jpg"},
        "disable_model_monitoring": True,
    }
    automatic = package_client.post("/infer/object_detection", json=payload)
    selected = package_client.post(
        "/infer/object_detection", json={**payload, **selectors}
    )
    again = package_client.post("/infer/object_detection", json=payload)
    assert automatic.status_code == selected.status_code == again.status_code == 200
    assert (
        automatic.json()["served_package"] == again.json()["served_package"] == "onnx-1"
    )
    assert selected.json()["served_package"] == "engine-1"
    registered = package_client.get("/model/registry").json()["models"]
    assert len(registered) == 2
    removed = package_client.post(
        "/model/remove", json={"model_id": "project/1", **selectors}
    )
    assert removed.status_code == 200
    assert [model["model_id"] for model in removed.json()["models"]] == ["project/1"]


def test_legacy_http_selection_is_enforced_and_acknowledged(package_client):
    response = package_client.post(
        "/project/1",
        params={
            "backend": "trt",
            "quantization": "fp16",
            "image": "https://example.com/image.jpg",
        },
    )
    assert response.status_code == 200
    assert response.json()["served_package"] == "engine-1"
    assert response.headers["X-Roboflow-Model-Selection"] == "applied"


def test_legacy_http_rejects_conflicting_selectors(package_client):
    response = package_client.post(
        "/project/1",
        params={
            "backend": "trt",
            "model_package_id": "engine-1",
            "image": "https://example.com/image.jpg",
        },
    )
    assert response.status_code == 422
    assert "model_package_id" in response.text


def test_aliased_legacy_selection_shares_v1_entry_and_can_be_removed(
    package_client, monkeypatch
):
    from inference.models.aliases import REGISTERED_ALIASES

    monkeypatch.setitem(REGISTERED_ALIASES, "alias/1", "project/1")
    selectors = {"backend": "trt", "quantization": "fp16"}
    legacy = package_client.post(
        "/alias/1", params={**selectors, "image": "https://example.com/image.jpg"}
    )
    modern = package_client.post(
        "/infer/object_detection",
        json={
            "model_id": "project/1",
            **selectors,
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
        },
    )
    assert legacy.status_code == modern.status_code == 200
    registered = package_client.get("/model/registry").json()["models"]
    assert len(registered) == 1
    loaded = package_client.post(
        "/model/add", json={"model_id": "alias/1", **selectors}
    )
    assert loaded.status_code == 200
    assert loaded.json()["selected_model_id"] == registered[0]["model_id"]
    removed = package_client.post(
        "/model/remove", json={"model_id": "alias/1", **selectors}
    )
    assert removed.status_code == 200
    assert removed.json()["models"] == []


def test_selected_model_handle_depends_on_server_secret(package_client, monkeypatch):
    from inference.core.entities.requests import model_selection

    payload = {"model_id": "project/1", "backend": "trt", "api_key": "owner"}
    first = package_client.post("/model/add", json=payload)
    assert first.status_code == 200
    handle = first.json()["selected_model_id"]
    assert (
        package_client.post("/model/add", json=payload).json()["selected_model_id"]
        == handle
    )
    monkeypatch.setattr(model_selection, "_MODEL_SELECTION_SECRET", b"different-server")
    second = package_client.post("/model/add", json=payload)
    assert second.status_code == 200
    assert second.json()["selected_model_id"] != handle


def test_browser_can_read_package_selection_acknowledgment(package_client):
    response = package_client.post(
        "/model/add",
        json={"model_id": "project/1", "backend": "trt"},
        headers={"Origin": "https://client.example"},
    )
    assert response.status_code == 200
    assert response.headers["X-Roboflow-Model-Selection"] == "applied"
    assert "X-Roboflow-Model-Selection" in response.headers[
        "Access-Control-Expose-Headers"
    ].split(", ")


def test_http_cannot_reuse_another_credentials_restricted_package(package_client):
    payload = {
        "model_id": "project/1",
        "model_package_id": "engine-1",
        "image": {"type": "url", "value": "https://example.com/image.jpg"},
        "disable_model_monitoring": True,
    }
    allowed = package_client.post(
        "/infer/object_detection", json={**payload, "api_key": "owner"}
    )
    denied = package_client.post(
        "/infer/object_detection", json={**payload, "api_key": "other-workspace"}
    )
    assert allowed.status_code == 200
    assert denied.status_code in (401, 403)
    assert "engine-1" not in denied.json().get("served_package", "")


def test_lmm_rejects_model_package_selection(package_client):
    response = package_client.post(
        "/infer/lmm",
        json={
            "model_id": "project/1",
            "backend": "trt",
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
            "prompt": "Describe the image.",
        },
    )
    assert response.status_code == 422
    assert "not supported for LMM" in response.text
