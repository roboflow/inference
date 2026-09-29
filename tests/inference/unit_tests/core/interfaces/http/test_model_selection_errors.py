from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

import inference_models.errors as inference_models_errors
from inference.core import env
from inference.core.interfaces.http import http_api
from inference.core.managers import base
from inference_models.errors import (
    ModelLoadingError,
    ModelPackageNegotiationError,
    NoModelPackagesAvailableError,
    RuntimeIntrospectionError,
    UnknownBackendTypeError,
    UnknownQuantizationError,
)


@pytest.fixture(autouse=True)
def disable_usage_push(monkeypatch):
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )


@pytest.fixture
def failing_package_client(monkeypatch):
    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(base, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(base, "OFFLINE_MODE", True)
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "OFFLINE_MODE", True)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(
        base.ModelManager,
        "resolve_model_packages",
        lambda self, model_id, api_key, selectors, **kwargs: [
            SimpleNamespace(
                model_package_id=selectors.get("model_package_id", "package-1")
            )
        ],
    )
    constructor = MagicMock()
    constructor.supports_model_package_selection = True
    registry = MagicMock()
    registry.get_model.return_value = constructor
    manager = base.ModelManager(registry, content_addressed_artifact_cache=MagicMock())
    interface = http_api.HttpInterface(model_manager=manager)
    with TestClient(interface.app) as client:
        yield client, constructor


@pytest.mark.parametrize(
    "path", ["/infer/object_detection", "/model/add", "/project/1"]
)
@pytest.mark.parametrize("selector", [{"model_package_id": "missing"}, {}])
def test_missing_package_and_cache_handle_return_same_public_error(
    failing_package_client, path, selector
):
    client, constructor = failing_package_client
    constructor.side_effect = NoModelPackagesAvailableError("private package details")
    model_id = "project/1" if selector else "project/1:package:private"
    if path == "/project/1":
        response = client.post(
            f"/{model_id}",
            params={"image": "https://example.com/image.jpg", **selector},
        )
    else:
        response = client.post(
            path,
            json={
                "model_id": model_id,
                "image": {"type": "url", "value": "https://example.com/image.jpg"},
                **selector,
            },
        )
    assert response.status_code == 404
    assert response.json() == {"message": "Model or model package not found."}


@pytest.mark.parametrize("path", ["/infer/object_detection", "/model/add"])
@pytest.mark.parametrize(
    "selectors,error,status",
    [
        ({"backend": "unknown"}, UnknownBackendTypeError("details"), 400),
        ({"quantization": "unknown"}, UnknownQuantizationError("details"), 400),
        ({"backend": "trt"}, NoModelPackagesAvailableError("details"), 400),
        ({}, NoModelPackagesAvailableError("details"), 500),
        ({"backend": "trt"}, RuntimeIntrospectionError("details"), 500),
        ({"backend": "trt"}, ModelPackageNegotiationError("details"), 500),
        ({"model_package_id": "broken"}, ModelLoadingError("details"), 500),
        ({"backend": "trt"}, UnknownQuantizationError("server configuration"), 500),
    ],
)
def test_selector_errors_do_not_mask_server_failures(
    failing_package_client, path, selectors, error, status, monkeypatch
):
    client, constructor = failing_package_client
    monkeypatch.setattr(
        base.ModelManager, "resolve_model_packages", MagicMock(side_effect=error)
    )
    constructor.side_effect = error
    response = client.post(
        path,
        json={
            "model_id": "project/1",
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
            **selectors,
        },
    )
    assert response.status_code == status


@pytest.mark.parametrize(
    "selectors",
    [{"backend": "onnx"}, {"quantization": "fp32"}, {"model_package_id": "package-1"}],
)
def test_unsupported_model_selection_is_rejected_before_construction(
    failing_package_client, selectors
):
    client, constructor = failing_package_client
    constructor.supports_model_package_selection = False

    response = client.post(
        "/model/add", json={"model_id": "clip/ViT-B-16", **selectors}
    )

    assert response.status_code == 400
    assert "does not support explicit package selection" in response.json()["message"]
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "path", ["/model/add", "/infer/object_detection", "/project/1"]
)
def test_pinned_package_policy_rejection_returns_bad_request(
    failing_package_client, path
):
    error_type = getattr(inference_models_errors, "ModelPackagePolicyError", None)
    if error_type is None:
        pytest.skip(
            "Installed inference-models does not expose pinned policy validation"
        )
    client, constructor = failing_package_client
    constructor.side_effect = error_type("Unavailable on this runtime")
    if path == "/project/1":
        response = client.post(
            path,
            params={
                "model_package_id": "package-1",
                "image": "https://example.com/image.jpg",
            },
        )
    else:
        response = client.post(
            path,
            json={
                "model_id": "project/1",
                "model_package_id": "package-1",
                "image": {"type": "url", "value": "https://example.com/image.jpg"},
            },
        )
    assert response.status_code == 400
    assert "backend policy or runtime" in response.json()["message"]
