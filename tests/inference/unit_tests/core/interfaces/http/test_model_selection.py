from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel
from starlette.testclient import TestClient

from inference.core.interfaces.http import http_api


@pytest.fixture(autouse=True)
def disable_usage_push(monkeypatch):
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "OFFLINE_MODE", True)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "DEPTH_ESTIMATION_ENABLED", True)
    monkeypatch.setattr(http_api, "ACTION_RECOGNITION_ENABLED", True)
    interface = http_api.HttpInterface(model_manager=MagicMock())
    with TestClient(interface.app) as client:
        yield client


@pytest.mark.parametrize("selector", [{"backend": "trt"}, {"quantization": "fp16"}])
@pytest.mark.parametrize(
    "path",
    [
        "/infer/object_detection",
        "/infer/action_recognition",
        "/model/add",
        "/model/remove",
    ],
)
def test_http_rejects_package_id_combined_with_other_selectors(client, selector, path):
    response = client.post(
        path,
        json={
            "model_id": "project/1",
            "model_package_id": "package-1",
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
            "video": {"type": "url", "value": "https://example.com/clip.mp4"},
            **selector,
        },
    )
    assert response.status_code == 422
    assert "model_package_id" in response.text


@pytest.mark.parametrize(
    "path",
    [
        "/infer/object_detection",
        "/infer/action_recognition",
        "/model/add",
        "/model/remove",
    ],
)
def test_http_rejects_selection_when_flag_is_disabled(client, monkeypatch, path):
    from inference.core import env

    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", False)
    response = client.post(
        path,
        json={
            "model_id": "project/1",
            "backend": "trt",
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
            "video": {"type": "url", "value": "https://example.com/clip.mp4"},
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
def package_client(monkeypatch, request):
    from types import SimpleNamespace

    from inference.core import env
    from inference.core.managers import base
    from inference.core.managers.base import ModelManager
    from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache

    class PackageResponse(BaseModel):
        served_package: str

    from inference_models import AutoModel

    def resolve_package(
        model_id,
        api_key=None,
        model_package_id=None,
        backend=None,
        quantization=None,
        **kwargs
    ):
        if api_key == "other-workspace" and model_package_id == "engine-1":
            from inference_models.errors import UnauthorizedModelAccessError

            raise UnauthorizedModelAccessError(
                "Package is not accessible to this workspace."
            )
        selected_backend = (
            "trt"
            if model_package_id == "engine-1"
            or backend == "trt"
            or quantization == "fp16"
            else "onnx"
        )
        return [
            SimpleNamespace(
                model_id=model_id,
                model_package_id="engine-1" if selected_backend == "trt" else "onnx-1",
                backend=selected_backend,
                quantization="fp16" if selected_backend == "trt" else "fp32",
            )
        ]

    monkeypatch.setattr(
        AutoModel, "resolve_model_packages", resolve_package, raising=False
    )

    class PackageModel:
        supports_model_package_selection = True
        task_type = getattr(request, "param", "object-detection")
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
            if self.task_type == "action-recognition":
                from inference.core.entities.responses.action_recognition import (
                    ActionRecognitionInferenceResponse,
                )
                from inference.core.entities.responses.inference import ResolvedModel

                return ActionRecognitionInferenceResponse(
                    timeline=[],
                    source_fps=30,
                    frame_count=30,
                    windows_classified=1,
                    resolved_model=ResolvedModel(**vars(self.resolved_model)),
                )
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
    monkeypatch.setattr(http_api, "ACTION_RECOGNITION_ENABLED", True)
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
@pytest.mark.parametrize("remove_by_handle", [False, True])
def test_http_selected_package_coexists_with_automatic_package(
    package_client, selectors, remove_by_handle
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
    assert selected.headers["X-Model-Id"] == "project/1"
    registered = package_client.get("/model/registry").json()["models"]
    assert len(registered) == 2
    removal = {"model_id": "project/1", **selectors}
    if remove_by_handle:
        loaded = package_client.post("/model/add", json=removal)
        assert loaded.status_code == 200
        removal = {"model_id": loaded.json()["selected_model_id"]}
    removed = package_client.post("/model/remove", json=removal)
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


@pytest.mark.parametrize("package_client", ["action-recognition"], indirect=True)
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "selectors",
    [{"backend": "trt", "quantization": "fp16"}, {"model_package_id": "engine-1"}],
)
def test_action_recognition_selected_package_coexists_with_default(
    package_client, legacy, selectors
):
    def infer(selection):
        if legacy:
            return package_client.post(
                "/project/1",
                params={"image": "https://example.com/clip.mp4", **selection},
            )
        return package_client.post(
            "/infer/action_recognition",
            json={
                "model_id": "project/1",
                "video": {"type": "url", "value": "https://example.com/clip.mp4"},
                "disable_model_monitoring": True,
                **selection,
            },
        )

    automatic = infer({})
    selected = infer(selectors)
    again = infer({})
    assert automatic.status_code == selected.status_code == again.status_code == 200
    assert automatic.json()["resolved_model"]["model_package_id"] == "onnx-1"
    assert again.json()["resolved_model"]["model_package_id"] == "onnx-1"
    assert selected.json()["resolved_model"]["model_package_id"] == "engine-1"
    assert selected.headers["X-Roboflow-Model-Selection"] == "applied"
    assert "X-Roboflow-Model-Selection" not in automatic.headers
    mismatch = infer({"backend": "trt", "quantization": "fp32"})
    assert mismatch.status_code == 400
    assert "X-Roboflow-Model-Selection" not in mismatch.headers


@pytest.mark.parametrize("legacy", [False, True])
def test_selected_package_telemetry_uses_public_model_id(
    package_client, monkeypatch, legacy
):
    from inference.core.managers import base

    spans = MagicMock()
    loaded = MagicMock()
    inferred = MagicMock()
    monkeypatch.setattr(base, "start_span", spans)
    monkeypatch.setattr(base, "record_model_loaded", loaded)
    monkeypatch.setattr(base, "record_inference", inferred)
    selectors = {"backend": "trt", "quantization": "fp16"}
    if legacy:
        response = package_client.post(
            "/project/1",
            params={"image": "https://example.com/image.jpg", **selectors},
        )
    else:
        response = package_client.post(
            "/infer/object_detection",
            json={
                "model_id": "project/1",
                "image": {"type": "url", "value": "https://example.com/image.jpg"},
                "disable_model_monitoring": True,
                **selectors,
            },
        )
    assert response.status_code == 200
    assert response.headers["X-Model-Id"] == "project/1"
    assert loaded.call_args.args[0] == "project/1"
    assert inferred.call_args.args[0] == "project/1"
    model_spans = [
        call.args[1]["model.id"]
        for call in spans.call_args_list
        if len(call.args) > 1 and "model.id" in call.args[1]
    ]
    assert model_spans == ["project/1", "project/1"]
    selected = package_client.post(
        "/model/add", json={"model_id": "project/1", **selectors}
    )
    assert ":package:" in selected.json()["selected_model_id"]
    assert selected.headers["X-Model-Id"] == "project/1"


def test_equivalent_selectors_and_credentials_share_one_package(package_client):
    handles = []
    for selectors, key in [
        ({"backend": "trt"}, "owner"),
        ({"quantization": "fp16"}, "second-authorized-key"),
        ({"model_package_id": "engine-1"}, "owner"),
    ]:
        response = package_client.post(
            "/model/add", json={"model_id": "project/1", "api_key": key, **selectors}
        )
        assert response.status_code == 200, response.text
        handles.append(response.json()["selected_model_id"])
    assert len(set(handles)) == 1
    assert len(package_client.get("/model/registry").json()["models"]) == 1


@pytest.mark.parametrize("selected_first", [False, True])
def test_default_and_explicit_requests_share_the_same_package(
    package_client, selected_first
):
    payload = {"model_id": "project/1", "api_key": "owner"}
    requests = [{}, {"backend": "onnx"}]
    if selected_first:
        requests.reverse()
    for selection in requests:
        response = package_client.post("/model/add", json={**payload, **selection})
        assert response.status_code == 200, response.text
    models = package_client.get("/model/registry").json()["models"]
    assert len(models) == 1
    assert models[0]["model_id"] == "project/1"
    response = package_client.post("/model/remove", json=payload)
    assert response.status_code == 200
    assert response.json()["models"] == []


def test_concurrent_equivalent_requests_load_one_package(package_client):
    from concurrent.futures import ThreadPoolExecutor

    selections = [
        {"backend": "trt"},
        {"quantization": "fp16"},
        {"model_package_id": "engine-1"},
    ]
    with ThreadPoolExecutor(max_workers=3) as executor:
        responses = list(
            executor.map(
                lambda selection: package_client.post(
                    "/model/add", json={"model_id": "project/1", **selection}
                ),
                selections,
            )
        )
    assert all(response.status_code == 200 for response in responses)
    assert len({response.json()["selected_model_id"] for response in responses}) == 1
    assert len(package_client.get("/model/registry").json()["models"]) == 1


@pytest.mark.parametrize("reuse_cached_fallback", [False, True])
def test_selection_tries_next_eligible_package_after_load_failure(
    package_client, monkeypatch, reuse_cached_fallback
):
    from types import SimpleNamespace

    from inference.core.managers.base import ModelManager
    from inference_models import AutoModel
    from inference_models.errors import ModelPackageAlternativesExhaustedError

    monkeypatch.setattr(
        AutoModel,
        "resolve_model_packages",
        lambda **kwargs: [
            SimpleNamespace(model_package_id="broken-onnx"),
            SimpleNamespace(model_package_id="onnx-1"),
        ],
    )
    if reuse_cached_fallback:
        monkeypatch.setattr(
            AutoModel,
            "resolve_model_packages",
            lambda **kwargs: [SimpleNamespace(model_package_id="onnx-1")],
        )
        assert (
            package_client.post(
                "/model/add", json={"model_id": "project/1", "backend": "onnx"}
            ).status_code
            == 200
        )
        monkeypatch.setattr(
            AutoModel,
            "resolve_model_packages",
            lambda **kwargs: [
                SimpleNamespace(model_package_id="broken-onnx"),
                SimpleNamespace(model_package_id="onnx-1"),
            ],
        )
    original_add = ModelManager.add_model
    attempted_packages = []

    def add_model(self, *args, **kwargs):
        package_id = kwargs.get("model_package_id")
        attempted_packages.append(package_id)
        if package_id == "broken-onnx":
            raise ModelPackageAlternativesExhaustedError("Invalid weights")
        return original_add(self, *args, **kwargs)

    monkeypatch.setattr(ModelManager, "add_model", add_model)
    selection = {} if reuse_cached_fallback else {"backend": "onnx"}
    response = package_client.post(
        "/model/add", json={"model_id": "project/1", **selection}
    )
    assert response.status_code == 200, response.text
    assert attempted_packages == (
        ["broken-onnx"] if reuse_cached_fallback else ["broken-onnx", "onnx-1"]
    )
    assert len(response.json()["models"]) == 1
    inferred = package_client.post(
        "/infer/object_detection",
        json={
            "model_id": "project/1",
            **selection,
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
        },
    )
    assert inferred.status_code == 200, inferred.text
    assert inferred.json()["served_package"] == "onnx-1"


def test_eviction_removes_default_alias_and_allows_reload(package_client):
    payload = {"model_id": "project/1"}
    selected = package_client.post("/model/add", json={**payload, "backend": "onnx"})
    assert selected.status_code == 200
    assert package_client.post("/model/add", json=payload).status_code == 200
    for model_id in ["other/1", "third/1", "fourth/1"]:
        assert (
            package_client.post("/model/add", json={"model_id": model_id}).status_code
            == 200
        )
    assert all(
        model["model_id"] != "project/1"
        for model in package_client.get("/model/registry").json()["models"]
    )
    reloaded = package_client.post("/model/add", json=payload)
    assert reloaded.status_code == 200, reloaded.text
    matching = [
        model for model in reloaded.json()["models"] if model["model_id"] == "project/1"
    ]
    assert len(matching) == 1
    inferred = package_client.post(
        "/infer/object_detection",
        json={
            **payload,
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
        },
    )
    assert inferred.status_code == 200, inferred.text
    assert inferred.json()["served_package"] == "onnx-1"


def test_older_library_keeps_default_loading_but_rejects_selection(
    package_client, monkeypatch
):
    from inference_models import AutoModel

    monkeypatch.delattr(AutoModel, "resolve_model_packages", raising=False)
    automatic = package_client.post("/model/add", json={"model_id": "project/1"})
    selected = package_client.post(
        "/model/add", json={"model_id": "project/1", "backend": "onnx"}
    )
    assert automatic.status_code == 200
    assert selected.status_code == 400
    assert "cannot resolve package selection" in selected.json()["message"]
    assert len(package_client.get("/model/registry").json()["models"]) == 1
