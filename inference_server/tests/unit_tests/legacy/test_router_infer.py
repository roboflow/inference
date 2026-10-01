import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from inference_models.errors import (
    CorruptedModelPackageError,
    EnvironmentConfigurationError,
    FileHashSumMissmatch,
    ForbiddenModelAccessError,
    InvalidParameterError,
    MissingDependencyError,
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    NoModelPackagesAvailableError,
    PaymentRequiredModelAccessError,
    RetryError,
    UnauthorizedModelAccessError,
    UsagePausedModelAccessError,
)
from PIL import Image

from inference_server.gateway import ModelManagerGateway
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.legacy.test_errors import (
    FORBIDDEN,
    HELP_SUFFIX,
    HELP_URL,
    INTERNAL_ERROR,
    MISCONFIGURATION,
    NOT_FOUND,
    PAYMENT_REQUIRED,
    RESTRICTED,
    UNAUTHORIZED,
    USAGE_PAUSED,
)


def _jpeg_b64(w=8, h=6):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _det():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


def test_infer_object_detection_single(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": {"type": "base64", "value": _jpeg_b64()},
            "confidence": 0.3,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": 8, "height": 6}
    assert body["predictions"][0]["class"] == "cat"
    assert "time" in body and "inference_id" in body
    infer_call = next(c for c in gw.calls if c[0] == "infer")
    assert infer_call[3]["confidence"] == 0.3


def test_infer_object_detection_batch_shapes(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={"ds/1": {"class_names": ["cat"]}},
    )
    c = legacy_client(gw)
    two = c.post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "image": [{"type": "base64", "value": _jpeg_b64()}] * 2,
        },
    )
    assert two.status_code == 200 and isinstance(two.json(), list)
    assert len(two.json()) == 2
    one = c.post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "image": [{"type": "base64", "value": _jpeg_b64()}]},
    )
    assert isinstance(one.json(), list) and len(one.json()) == 1
    none = c.post("/infer/object_detection", json={"model_id": "ds/1", "image": []})
    assert none.status_code == 200 and none.json() == []


def test_infer_response_carries_resolved_model_and_registry_tracks_alias(
    legacy_client, fake_stat
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("coco/3", "infer"): _det()},
        model_info={"coco/3": {"class_names": ["cat"]}},
    )
    c = legacy_client(gw)
    r = c.post(
        "/infer/object_detection",
        json={
            "model_id": "yolov8n-640",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )
    assert r.json()["resolved_model"] == {"model_id": "coco/3"}
    entry = c.get("/model/registry").json()["models"][0]
    assert entry["model_id"] == "yolov8n-640"
    assert entry["request_aliases"] == ["coco/3"]
    assert entry["request_paths"] == ["/infer/object_detection"]


def test_infer_response_carries_full_resolved_model_when_reported(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={
            "ds/1": {
                "class_names": ["cat"],
                "resolved_model": {
                    "model_id": "ds/1",
                    "model_package_id": "pkg",
                    "backend": "onnx",
                    "quantization": "fp32",
                },
            }
        },
    )
    r = legacy_client(gw).post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert r.json()["resolved_model"] == {
        "model_id": "ds/1",
        "model_package_id": "pkg",
        "backend": "onnx",
        "quantization": "fp32",
    }


def test_oversized_content_length_is_413(legacy_client, fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.configuration.MAX_BODY_BYTES", 10)
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert r.status_code == 413 and r.json() == {
        "message": "Request payload too large."
    }


def test_wrong_task_type_is_400(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("classification", "infer")
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert r.status_code == 400


def test_bearer_header_used_when_body_key_missing(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("ds/1", "infer"): _det()},
        model_info={"ds/1": {"class_names": ["cat"]}},
    )
    legacy_client(gw).post(
        "/infer/object_detection",
        headers={"Authorization": "Bearer H"},
        json={"model_id": "ds/1", "image": {"type": "base64", "value": _jpeg_b64()}},
    )
    assert ("ensure_loaded", "ds/1", "H") in gw.calls


def test_unsupported_legacy_param_is_501(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "image": {"type": "base64", "value": _jpeg_b64()},
            "disable_preproc_auto_orient": True,
        },
    )
    assert r.status_code == 501


class FailingLoadManager:
    def __init__(self, error):
        self.error = error
        self.load_calls = 0
        self.executor = None

    def __contains__(self, key):
        return False

    def load(self, key, api_key, **kwargs):
        self.load_calls += 1
        raise self.error

    def unload(self, key):
        raise KeyError(key)

    def stats(self):
        return {"models": []}

    def shutdown(self):
        pass


LOAD_FAILURE_MATRIX = [
    pytest.param(
        UnauthorizedModelAccessError("denied"),
        401,
        {"message": UNAUTHORIZED},
        id="unauthorized",
    ),
    pytest.param(
        PaymentRequiredModelAccessError("no credits"),
        402,
        {"message": PAYMENT_REQUIRED},
        id="payment-required",
    ),
    pytest.param(
        ForbiddenModelAccessError("denied"),
        403,
        {"message": FORBIDDEN},
        id="forbidden",
    ),
    pytest.param(
        UsagePausedModelAccessError("paused"),
        423,
        {"message": USAGE_PAUSED},
        id="usage-paused",
    ),
    pytest.param(
        ModelNotFoundError("missing"), 404, {"message": NOT_FOUND}, id="not-found"
    ),
    pytest.param(
        ModelRetrievalError("empty package list", help_url=HELP_URL),
        500,
        {
            "message": f"Could not retrieve model empty package list{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="retrieval",
    ),
    pytest.param(
        RetryError("Connectivity error"), 500, INTERNAL_ERROR, id="registry-retry"
    ),
    pytest.param(
        ModelPackageAlternativesExhaustedError(
            "none loaded",
            help_url=HELP_URL,
            alternatives_errors=[RetryError("Connectivity error for URL")],
        ),
        500,
        {
            "message": f"Model loading failed: none loaded{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="alternatives-exhausted",
    ),
    pytest.param(
        ModelPackageAlternativesExhaustedError(
            "none loaded",
            help_url=HELP_URL,
            alternatives_errors=[
                RuntimeError("no cuda"),
                ModelPackageRestrictedError("too big"),
            ],
        ),
        507,
        {"message": RESTRICTED, "help_url": HELP_URL},
        id="alternatives-exhausted-restricted",
    ),
    pytest.param(
        ModelPackageRestrictedError("too big", help_url=HELP_URL),
        507,
        {"message": RESTRICTED},
        id="restricted",
    ),
    pytest.param(
        NoModelPackagesAvailableError("no package", help_url=HELP_URL),
        500,
        {
            "message": f"Could not negotiate model package - no package{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="negotiation",
    ),
    pytest.param(
        MissingDependencyError("no pycuda"),
        500,
        MISCONFIGURATION,
        id="missing-dependency",
    ),
    pytest.param(
        EnvironmentConfigurationError("no provider"),
        500,
        MISCONFIGURATION,
        id="environment",
    ),
    pytest.param(
        InvalidParameterError("bad device"),
        500,
        MISCONFIGURATION,
        id="invalid-parameter",
    ),
    pytest.param(
        CorruptedModelPackageError("bad file", help_url=HELP_URL),
        500,
        {
            "message": f"Model loading failed: bad file{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="corrupted-package",
    ),
    pytest.param(
        FileHashSumMissmatch("md5 differs", help_url=HELP_URL),
        500,
        {
            "message": f"Issue with model package file: md5 differs{HELP_SUFFIX}",
            "help_url": HELP_URL,
        },
        id="file-hash",
    ),
    pytest.param(ValueError("unsafe id"), 500, INTERNAL_ERROR, id="value-error"),
    pytest.param(TimeoutError("lock"), 500, INTERNAL_ERROR, id="timeout-in-load"),
]


def _infer_request(client):
    response = client.post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )
    return response


@pytest.mark.parametrize("error,status,body", LOAD_FAILURE_MATRIX)
def test_load_failure_on_an_inference_route_answers_like_legacy(
    legacy_client, fake_stat, error, status, body
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    manager = FailingLoadManager(error)

    response = _infer_request(legacy_client(ModelManagerGateway(manager)))

    assert response.status_code == status
    assert response.json() == body
    assert response.headers["content-type"] == "application/json"
    assert "retry-after" not in response.headers
    assert manager.load_calls == 1


@pytest.mark.parametrize("failure", [("error", 5), ("error", 3), ("error",)])
def test_load_failure_without_a_description_is_a_broken_package(
    legacy_client, fake_stat, failure
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    gw.ensure_results = [failure]

    response = _infer_request(legacy_client(gw))

    assert response.status_code == 500
    assert response.json() == {"message": "Model package is broken."}
    assert "retry-after" not in response.headers


@pytest.mark.parametrize(
    "detail",
    [
        {"error_type": "SomethingElseError", "message": "boom"},
        {"error_type": "Optional", "message": "boom"},
        {"error_type": "PermissionError", "message": "denied"},
        {"message": "boom"},
        {},
    ],
)
def test_load_failure_of_an_unknown_kind_is_an_internal_error(
    legacy_client, fake_stat, detail
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    gw.ensure_results = [("error", 5, detail)]

    response = _infer_request(legacy_client(gw))

    assert response.status_code == 500
    assert response.json() == INTERNAL_ERROR
    assert "retry-after" not in response.headers


def test_model_reported_as_not_loaded_answers_not_ready_with_retry_after(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    gw.ensure_results = [("error", 6)]

    response = _infer_request(legacy_client(gw))

    assert response.status_code == 503
    assert response.json() == {
        "message": "Model is temporarily not ready - retry request."
    }
    assert response.headers["retry-after"] == "1"


def test_load_deadline_answers_not_ready_with_retry_after(
    legacy_client, fake_stat, monkeypatch
):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_LOAD_POLL_INTERVAL_S", 0)
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_LOAD_TIMEOUT_S", 0)
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()
    gw.ensure_results = [("load_timeout", 10)]

    response = _infer_request(legacy_client(gw))

    assert response.status_code == 503
    assert response.json() == {
        "message": "Model is temporarily not ready - retry request."
    }
    assert response.headers["retry-after"] == "1"


@pytest.mark.parametrize(
    "error,message",
    [
        (
            ModelPackageAlternativesExhaustedError(
                "none loaded: Connectivity error for URL: "
                "https://host/x?api_key=SECRET&y=1 and "
                "https://storage.example/a/b.onnx?X-Goog-Signature=SECRET",
                alternatives_errors=[RetryError("https://host/x?api_key=SECRET")],
            ),
            "Model loading failed: none loaded: Connectivity error for URL: "
            "https://host/*** and https://storage.example/***",
        ),
        (
            ModelRetrievalError("failed for https://host/x?api_key=SECRET&y=1"),
            "Could not retrieve model failed for https://host/***",
        ),
        (
            FileHashSumMissmatch(
                "bad md5 for url: https://storage.example/a/b.onnx"
                "?X-Goog-Signature=SECRET&X-Goog-Expires=60"
            ),
            "Issue with model package file: bad md5 for url: "
            "https://storage.example/***",
        ),
        (
            ModelRetrievalError("request with api_key=SECRET failed"),
            "Could not retrieve model request with api_key=*** failed",
        ),
        (
            ModelRetrievalError("denied\nAuthorization: Bearer SECRET"),
            "Could not retrieve model denied\nAuthorization: ***",
        ),
    ],
)
def test_load_failure_answer_hides_urls_and_secret_values(
    legacy_client, fake_stat, error, message
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(ModelManagerGateway(FailingLoadManager(error)))

    response = _infer_request(client)

    assert response.status_code == 500
    assert response.json() == {"message": message, "help_url": None}
    assert "SECRET" not in response.text


def test_offline_load_failure_with_a_description_answers_by_its_cause(
    legacy_client, fake_stat, monkeypatch
):
    monkeypatch.setattr("inference_server.legacy.bridge.OFFLINE_MODE", True)
    error = ModelRetrievalError(
        "Cannot fetch Roboflow model metadata - OFFLINE_MODE is enabled. All "
        "models must be pre-cached locally.",
        help_url=HELP_URL,
    )
    manager = FailingLoadManager(error)

    response = _infer_request(legacy_client(ModelManagerGateway(manager)))

    assert response.status_code == 500
    assert response.json() == {
        "message": "Could not retrieve model Cannot fetch Roboflow model metadata "
        f"- OFFLINE_MODE is enabled. All models must be pre-cached locally.{HELP_SUFFIX}",
        "help_url": HELP_URL,
    }
    assert fake_stat == {}


def test_offline_load_failure_without_a_description_is_404(
    legacy_client, fake_stat, monkeypatch
):
    monkeypatch.setattr("inference_server.legacy.bridge.OFFLINE_MODE", True)
    gw = FakeGateway()
    gw.ensure_results = [("error", 5)]

    response = _infer_request(legacy_client(gw))

    assert response.status_code == 404
    assert response.json() == {"message": "Model ds/1 not available offline"}
