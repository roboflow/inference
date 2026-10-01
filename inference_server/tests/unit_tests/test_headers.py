import asyncio
import base64
import contextvars
import importlib
import io
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from inference_models.errors import (
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
    NoModelPackagesAvailableError,
    PaymentRequiredModelAccessError,
    UnauthorizedModelAccessError,
)
from inference_sdk.config import execution_id
from PIL import Image

import inference_server.app as app_mod
from inference_server import configuration
from inference_server.cors import PathAwareCORSMiddleware
from inference_server.framework.dispatch import handle_model_inference_request
from inference_server.framework.entities import (
    ModelHandlerDescription,
    ModelInterfaceDescription,
)
from inference_server.framework.registry import _HANDLERS
from inference_server.gateway import ModelManagerGateway
from inference_server.hosted import serverless_auth
from inference_server.hosted.serverless_auth import ServerlessAuthMiddleware
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    LoopBridge,
    Route,
    SyncLegacyBridge,
)
from inference_server.middlewares.correlation_id import (
    CorrelationIdMiddleware,
    correlation_id,
)
from inference_server.middlewares.model_load import (
    MODEL_LOAD_EVENTS,
    REMOTE_MODEL_LOADS,
    ModelLoadHeadersMiddleware,
    build_model_response_headers,
)
from tests.unit_tests.legacy.conftest import FakeGateway

MODEL_HEADERS_COLD = {
    "x-model-cold-start",
    "x-model-cold-start-count",
    "x-model-load-time",
    "x-model-load-details",
    "x-model-id",
}
MODEL_HEADERS_WARM = {"x-model-cold-start", "x-model-cold-start-count", "x-model-id"}
PASSTHROUGH_WF = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowParameter", "name": "x"}],
    "steps": [],
    "outputs": [{"type": "JsonField", "name": "y", "selector": "$inputs.x"}],
}
OD_WF = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "det",
            "image": "$inputs.image",
            "model_id": "ds/1",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.det.predictions",
        }
    ],
}


def _jpeg_b64():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode()


def _det():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


def _infer_body():
    return {
        "model_id": "ds/1",
        "api_key": "k",
        "image": {"type": "base64", "value": _jpeg_b64()},
    }


def _model_headers(response):
    return {name for name in response.headers.keys() if name.startswith("x-model-")}


class _Manager:
    def __init__(self):
        self.loaded = set()
        self.evict_next = False
        self.executor = None

    def __contains__(self, key):
        return key in self.loaded

    def load(self, key, api_key, **kwargs):
        self.loaded.add(key)

    def unload(self, key):
        self.loaded.discard(key)

    def stats(self):
        return {
            "models": [
                {"model_id": key, "class_names": ["cat"], "actions": {"infer": {}}}
                for key in self.loaded
            ]
        }

    def shutdown(self):
        pass

    async def process_async(self, key, **kwargs):
        if self.evict_next:
            self.evict_next = False
            self.loaded.discard(key)
            raise KeyError(key)
        return _det()


def test_cold_start_headers_follow_first_load_warm_hit_and_internal_reload(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    manager = _Manager()
    client = legacy_client(ModelManagerGateway(manager))

    first = client.post("/infer/object_detection", json=_infer_body())
    second = client.post("/infer/object_detection", json=_infer_body())
    manager.evict_next = True
    third = client.post("/infer/object_detection", json=_infer_body())

    assert first.status_code == second.status_code == third.status_code == 200
    assert _model_headers(first) == MODEL_HEADERS_COLD
    assert first.headers["X-Model-Cold-Start"] == "true"
    assert first.headers["X-Model-Cold-Start-Count"] == "1"
    assert first.headers["X-Model-Id"] == "ds/1"
    load_time = float(first.headers["X-Model-Load-Time"])
    assert first.headers["X-Model-Load-Details"] == json.dumps(
        [{"m": "ds/1", "t": load_time}]
    )
    assert _model_headers(second) == MODEL_HEADERS_WARM
    assert second.headers["X-Model-Cold-Start"] == "false"
    assert second.headers["X-Model-Cold-Start-Count"] == "0"
    assert second.headers["X-Model-Id"] == "ds/1"
    assert _model_headers(third) == MODEL_HEADERS_COLD
    assert third.headers["X-Model-Cold-Start"] == "true"
    assert third.headers["X-Model-Cold-Start-Count"] == "1"


def test_every_response_carries_engine_and_default_model_headers(legacy_client):
    client = legacy_client(FakeGateway())

    health = client.get("/v2/server/health")
    rejected = client.post("/v2/models/infer")

    for response in (health, rejected):
        assert response.headers["x-inference-engine"] == "inference-models"
        assert _model_headers(response) == {
            "x-model-cold-start",
            "x-model-cold-start-count",
        }
        assert response.headers["X-Model-Cold-Start"] == "false"
        assert response.headers["X-Model-Cold-Start-Count"] == "0"
        assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])
    assert rejected.status_code == 401


def test_incoming_correlation_id_is_echoed_by_the_app(legacy_client):
    client = legacy_client(FakeGateway())
    request_id = uuid.uuid4().hex

    response = client.get("/v2/server/health", headers={"X-Request-ID": request_id})

    assert response.headers["X-Request-ID"] == request_id


def test_non_uuid_correlation_id_is_replaced_by_the_app_when_logging_is_disabled(
    legacy_client, monkeypatch
):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    client = legacy_client(FakeGateway())

    response = client.get("/v2/server/health", headers={"X-Request-ID": "trace-42"})

    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])
    assert response.headers.get_list("x-request-id") == [
        response.headers["X-Request-ID"]
    ]


def test_legacy_project_version_route_carries_model_headers(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(ModelManagerGateway(_Manager()))
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")

    response = client.post(
        "/ds/1?api_key=k",
        content=base64.b64encode(buffer.getvalue()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Model-Id"] == "ds/1"
    assert response.headers["X-Model-Cold-Start"] == "true"
    assert response.headers["X-Model-Cold-Start-Count"] == "1"
    assert float(response.headers["X-Model-Load-Time"]) >= 0.0


def test_explicit_load_reports_one_cold_start_and_the_model_id(
    legacy_client, monkeypatch
):
    monkeypatch.setattr(app_mod._cfg, "ENABLE_CONTROL_PLANE_ROUTES", True)
    monkeypatch.setattr(
        app_mod, "validate_api_key", AsyncMock(return_value=(True, "ws-1"))
    )
    client = legacy_client(ModelManagerGateway(_Manager()))
    headers = {"Authorization": "Bearer k"}

    cold = client.post("/v2/models/load?model_id=ds/1", headers=headers)
    warm = client.post("/v2/models/load?model_id=ds/1", headers=headers)

    assert cold.status_code == warm.status_code == 200, cold.text
    assert _model_headers(cold) == MODEL_HEADERS_COLD
    assert cold.headers["X-Model-Id"] == "ds/1"
    assert cold.headers["X-Model-Cold-Start"] == "true"
    assert cold.headers["X-Model-Cold-Start-Count"] == "1"
    load_time = float(cold.headers["X-Model-Load-Time"])
    assert cold.headers["X-Model-Load-Details"] == json.dumps(
        [{"m": "ds/1", "t": load_time}]
    )
    assert _model_headers(warm) == MODEL_HEADERS_WARM
    assert warm.headers["X-Model-Id"] == "ds/1"
    assert warm.headers["X-Model-Cold-Start"] == "false"


def test_alias_and_canonical_requests_report_their_own_model_id(
    legacy_client, fake_stat
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    manager = _Manager()
    client = legacy_client(ModelManagerGateway(manager))
    image = {"type": "base64", "value": _jpeg_b64()}

    aliased = client.post(
        "/infer/object_detection", json={"model_id": "yolov8n-640", "image": image}
    )
    canonical = client.post(
        "/infer/object_detection", json={"model_id": "coco/3", "image": image}
    )

    assert aliased.status_code == canonical.status_code == 200
    assert manager.loaded == {"coco/3"}
    assert aliased.headers["X-Model-Id"] == "yolov8n-640"
    assert aliased.headers["X-Model-Cold-Start"] == "true"
    load_time = float(aliased.headers["X-Model-Load-Time"])
    assert aliased.headers["X-Model-Load-Details"] == json.dumps(
        [{"m": "yolov8n-640", "t": load_time}]
    )
    assert canonical.headers["X-Model-Id"] == "coco/3"
    assert canonical.headers["X-Model-Cold-Start"] == "false"


def test_predefined_workflow_sets_workflow_id_header(legacy_client, monkeypatch):
    monkeypatch.setattr(
        "inference_server.workflows.host.get_workflow_specification",
        lambda **kwargs: PASSTHROUGH_WF,
    )
    client = legacy_client(FakeGateway())

    response = client.post("/ws/workflows/wf", json={"inputs": {"x": 1}})
    plain = client.post(
        "/workflows/run", json={"specification": PASSTHROUGH_WF, "inputs": {"x": 1}}
    )

    assert response.status_code == 200, response.text
    assert response.headers["X-Workflow-Id"] == "wf"
    assert "X-Workflow-Id" not in plain.headers


def test_workflow_model_load_is_reported(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(ModelManagerGateway(_Manager()))
    body = {
        "specification": OD_WF,
        "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
        "api_key": "k",
    }

    first = client.post("/workflows/run", json=body)
    second = client.post("/workflows/run", json=body)

    assert first.status_code == 200, first.text
    assert first.headers["X-Model-Id"] == "ds/1"
    assert first.headers["X-Model-Cold-Start"] == "true"
    assert second.headers["X-Model-Id"] == "ds/1"
    assert second.headers["X-Model-Cold-Start"] == "false"


class _SlowManager(_Manager):
    def load(self, key, api_key, **kwargs):
        time.sleep(0.2)
        super().load(key, api_key, **kwargs)


async def _ensure_loaded_in_own_request(gateway, model_id):
    events = []
    token = MODEL_LOAD_EVENTS.set(events)
    try:
        status = await gateway.ensure_loaded(model_id)
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    return status, events


async def _wait_for_events(events, count):
    for _ in range(200):
        if len(events) >= count:
            return
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_only_the_request_that_started_the_load_records_it():
    gateway = ModelManagerGateway(_SlowManager())

    (first, first_events), (second, second_events) = await asyncio.gather(
        _ensure_loaded_in_own_request(gateway, "m"),
        _ensure_loaded_in_own_request(gateway, "m"),
    )

    assert first == second == ("model_ready",)
    assert [(model_id, cold) for model_id, cold, _ in first_events] == [("m", True)]
    assert first_events[0][2] >= 0.2
    assert second_events == []


@pytest.mark.asyncio
async def test_request_that_already_has_the_model_records_nothing():
    gateway = ModelManagerGateway(_SlowManager())

    _, first_events = await _ensure_loaded_in_own_request(gateway, "m")
    _, second_events = await _ensure_loaded_in_own_request(gateway, "m")

    assert len(first_events) == 1
    assert second_events == []


@pytest.mark.asyncio
async def test_load_outside_a_request_records_nothing():
    manager = _SlowManager()
    gateway = ModelManagerGateway(manager)

    assert MODEL_LOAD_EVENTS.get() is None
    assert await gateway.ensure_loaded("m") == ("model_ready",)
    assert "m" in manager.loaded


@pytest.mark.asyncio
async def test_initiator_that_timed_out_still_gets_the_load_recorded():
    gateway = ModelManagerGateway(_SlowManager(), load_wait_s=0.01)
    events = []
    token = MODEL_LOAD_EVENTS.set(events)
    try:
        timed_out = await gateway.ensure_loaded("m")
        await _wait_for_events(events, 1)
        ready = await gateway.ensure_loaded("m")
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert timed_out == ("load_timeout", 0)
    assert ready == ("model_ready",)
    assert [(model_id, cold) for model_id, cold, _ in events] == [("m", True)]
    assert events[0][2] >= 0.2


@pytest.mark.asyncio
async def test_failed_load_records_no_cold_start():
    class _FailingManager(_Manager):
        def load(self, key, api_key, **kwargs):
            raise RuntimeError("weights download failed")

    gateway = ModelManagerGateway(_FailingManager())
    events = []
    token = MODEL_LOAD_EVENTS.set(events)
    try:
        status = await gateway.ensure_loaded("m")
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert status == (
        "error",
        5,
        {
            "error_type": "RuntimeError",
            "message": "weights download failed",
            "help_url": None,
            "status_code": None,
            "restricted": False,
        },
    )
    assert events == []


@pytest.mark.asyncio
async def test_concurrent_requests_attribute_the_cold_start_to_the_loader():
    import httpx

    gateway = ModelManagerGateway(_SlowManager())
    bridge = LegacyModelBridge(gateway)
    inner = FastAPI()

    @inner.get("/probe")
    async def _probe():
        await bridge.ensure_loaded(SimpleNamespace(registry_id="m"), None)
        return {"ok": True}

    inner.add_middleware(ModelLoadHeadersMiddleware)
    transport = httpx.ASGITransport(app=inner)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        responses = await asyncio.gather(client.get("/probe"), client.get("/probe"))

    cold_starts = sorted(
        response.headers["X-Model-Cold-Start-Count"] for response in responses
    )
    assert cold_starts == ["0", "1"]
    assert sorted(response.headers["X-Model-Cold-Start"] for response in responses) == [
        "false",
        "true",
    ]


@pytest.mark.asyncio
async def test_load_time_excludes_executor_queue_wait():
    manager = _Manager()
    manager.executor = ThreadPoolExecutor(max_workers=1)
    gateway = ModelManagerGateway(manager)

    def _load(key, api_key, **kwargs):
        time.sleep(0.1)
        manager.loaded.add(key)

    manager.load = _load
    events = []
    token = MODEL_LOAD_EVENTS.set(events)
    try:
        manager.executor.submit(time.sleep, 0.4)
        status = await gateway.ensure_loaded("m")
    finally:
        MODEL_LOAD_EVENTS.reset(token)
        manager.executor.shutdown(wait=True)

    assert status == ("model_ready",)
    assert [(model_id, cold) for model_id, cold, _ in events] == [("m", True)]
    assert 0.1 <= events[0][2] < 0.3


class _UnloadingManager(_Manager):
    def __init__(self):
        super().__init__()
        self.healthy = False

    def is_healthy(self, key):
        return self.healthy

    def unload(self, key):
        time.sleep(0.2)
        super().unload(key)

    def load(self, key, api_key, **kwargs):
        self.healthy = True
        super().load(key, api_key, **kwargs)


@pytest.mark.asyncio
async def test_load_time_excludes_dead_backend_unload():
    manager = _UnloadingManager()
    manager.loaded.add("m")
    gateway = ModelManagerGateway(manager)
    events = []
    token = MODEL_LOAD_EVENTS.set(events)
    try:
        status = await gateway.ensure_loaded("m")
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert status == ("model_ready",)
    assert [(model_id, cold) for model_id, cold, _ in events] == [("m", True)]
    assert events[0][2] < 0.1


def test_concurrent_alias_and_canonical_operations_share_one_load(fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gateway = ModelManagerGateway(_SlowManager())
    bridge = LegacyModelBridge(gateway)
    inner = FastAPI()

    @inner.get("/probe")
    async def _probe():
        sync_bridge = SyncLegacyBridge(bridge, LoopBridge(asyncio.get_running_loop()))

        def _operation(model_id):
            route = sync_bridge.resolve(model_id, "k")
            sync_bridge.infer(route, "k", "infer", [None], {})

        await asyncio.gather(
            asyncio.to_thread(_operation, "yolov8n-640"),
            asyncio.to_thread(_operation, "coco/3"),
        )
        return {"ok": True}

    inner.add_middleware(ModelLoadHeadersMiddleware)

    response = TestClient(inner).get("/probe")

    assert response.status_code == 200
    assert response.headers["X-Model-Id"] == "coco/3,yolov8n-640"
    assert response.headers["X-Model-Cold-Start"] == "true"
    assert response.headers["X-Model-Cold-Start-Count"] == "1"
    details = json.loads(response.headers["X-Model-Load-Details"])
    assert [entry["m"] for entry in details] in (["coco/3"], ["yolov8n-640"])
    assert details[0]["t"] >= 0.2


def test_failed_load_still_reports_the_requested_model_id(legacy_client, fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gateway = FakeGateway()
    gateway.ensure_results = [("error", 5)]

    response = legacy_client(gateway).post(
        "/infer/object_detection",
        json={
            "model_id": "yolov8n-640",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
    )

    assert response.status_code == 500
    assert response.headers["X-Model-Id"] == "yolov8n-640"
    assert response.headers["X-Model-Cold-Start"] == "false"
    assert response.headers["X-Model-Cold-Start-Count"] == "0"


class _FailingManager(_Manager):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def load(self, key, api_key, **kwargs):
        raise self.error


_LOAD_ERRORS = [
    UnauthorizedModelAccessError("denied"),
    PaymentRequiredModelAccessError("no credits"),
    ModelNotFoundError("missing"),
    ModelPackageRestrictedError("too big", help_url="https://help.example"),
    ModelPackageAlternativesExhaustedError(
        "none loaded",
        help_url="https://help.example",
        alternatives_errors=[ModelPackageRestrictedError("too big")],
    ),
    NoModelPackagesAvailableError("no package", help_url="https://help.example"),
    ValueError("unsafe id"),
]


@pytest.mark.parametrize("error", _LOAD_ERRORS, ids=lambda e: type(e).__name__)
def test_v2_explicit_load_answers_every_load_failure_the_same(
    legacy_client, monkeypatch, error
):
    monkeypatch.setattr(app_mod._cfg, "ENABLE_CONTROL_PLANE_ROUTES", True)
    monkeypatch.setattr(
        app_mod, "validate_api_key", AsyncMock(return_value=(True, "ws-1"))
    )
    client = legacy_client(ModelManagerGateway(_FailingManager(error)))

    response = client.post(
        "/v2/models/load?model_id=ds/1", headers={"Authorization": "Bearer k"}
    )

    assert response.status_code == 500
    assert response.json() == {
        "error_code": "LOAD_FAILED",
        "description": "model load failed",
    }
    assert "retry-after" not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("error", _LOAD_ERRORS, ids=lambda e: type(e).__name__)
async def test_v2_inference_answers_every_load_failure_the_same(error):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/models/infer",
        "query_string": b"model_id=ds/1",
        "headers": [(b"authorization", b"Bearer k1")],
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    interface = ModelInterfaceDescription(task="t", params={}, output_schema={})
    _HANDLERS[("fake-task", "infer")] = ModelHandlerDescription(
        input_parser=AsyncMock(return_value={"images": [b"x"], "params": {}}),
        handler=AsyncMock(),
        output_serializer=MagicMock(),
        interface_provider=lambda: interface,
    )
    try:
        with patch(
            "inference_server.framework.dispatch.stat_model_while_checking_auth",
            new=AsyncMock(return_value=("fake-task", "infer")),
        ):
            response = await handle_model_inference_request(
                Request(scope, _receive),
                ModelManagerGateway(_FailingManager(error)),
            )
    finally:
        del _HANDLERS[("fake-task", "infer")]

    assert response.status_code == 500
    assert json.loads(response.body) == {
        "error_code": "LOAD_FAILED",
        "description": "model load failed",
    }
    assert "retry-after" not in response.headers


def _workflow_run(client):
    response = client.post(
        "/workflows/run",
        json={
            "specification": OD_WF,
            "inputs": {"image": {"type": "base64", "value": _jpeg_b64()}},
            "api_key": "k",
        },
    )
    return response


@pytest.mark.parametrize(
    "error,status,error_type,inner_error_type,message",
    [
        (
            UnauthorizedModelAccessError("denied"),
            401,
            "ClientCausedStepExecutionError",
            "UnauthorizedModelAccessError",
            "Unauthorized error occurred while execution of step det - details of "
            "error: denied. This error usually mean the problem with Roboflow API "
            "key.",
        ),
        (
            PaymentRequiredModelAccessError("no credits"),
            402,
            "ClientCausedStepExecutionError",
            "PaymentRequiredModelAccessError",
            "Not enough credits to execute step det. Verify your workspace billing "
            "page. Details: no credits",
        ),
        (
            ModelNotFoundError("missing"),
            404,
            "ClientCausedStepExecutionError",
            "ModelNotFoundError",
            "Could not find requested Roboflow resource while execution of step det "
            "- details of error: missing. This error usually mean the problem with "
            "not existing model.",
        ),
        (
            ModelPackageRestrictedError("too big"),
            507,
            "RuntimeLimitsCausedStepExecutionError",
            "ModelPackageRestrictedError",
            "Model loading failed due to restrictions of server configuration - "
            "usually due to excessive runtime memory requirement of the model (for "
            "instance caused by large input size).",
        ),
        (
            NoModelPackagesAvailableError("no package"),
            500,
            "StepExecutionError",
            "NoModelPackagesAvailableError",
            "no package",
        ),
        (
            ValueError("unsafe id"),
            500,
            "StepExecutionError",
            "ModelLoadFailedError",
            "unsafe id",
        ),
    ],
)
def test_workflow_step_load_failure_is_answered_by_its_cause(
    legacy_client, fake_stat, error, status, error_type, inner_error_type, message
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(ModelManagerGateway(_FailingManager(error)))

    response = _workflow_run(client)

    assert response.status_code == status
    body = response.json()
    assert body["message"] == message
    assert body["error_type"] == error_type
    assert body["inner_error_type"] == inner_error_type
    assert "retry-after" not in response.headers


def test_workflow_step_load_failure_without_a_description_is_a_broken_package(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = FakeGateway()
    gateway.ensure_results = [("error", 5)]

    response = _workflow_run(legacy_client(gateway))

    assert response.status_code == 500
    body = response.json()
    assert body["message"] == "Model package is broken."
    assert body["error_type"] == "ClientCausedStepExecutionError"
    assert body["inner_error_type"] == "LegacyHTTPError"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,status_code",
    [
        (("error", 5), 500),
        (("error", 5, {"error_type": "ModelNotFoundError", "message": "x"}), 500),
        (("load_timeout", 1), 503),
    ],
)
async def test_v2_dispatch_records_the_model_id_before_a_failed_load(
    status, status_code
):
    proxy = MagicMock()
    proxy.ensure_loaded = AsyncMock(return_value=status)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/models/infer",
        "query_string": b"model_id=ds/1",
        "headers": [(b"authorization", b"Bearer k1")],
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    interface = ModelInterfaceDescription(task="t", params={}, output_schema={})
    _HANDLERS[("fake-task", "infer")] = ModelHandlerDescription(
        input_parser=AsyncMock(return_value={"images": [b"x"], "params": {}}),
        handler=AsyncMock(),
        output_serializer=MagicMock(),
        interface_provider=lambda: interface,
    )
    events = []
    token = MODEL_LOAD_EVENTS.set(events)
    try:
        with patch(
            "inference_server.framework.dispatch.stat_model_while_checking_auth",
            new=AsyncMock(return_value=("fake-task", "infer")),
        ):
            response = await handle_model_inference_request(
                Request(scope, _receive), proxy
            )
    finally:
        MODEL_LOAD_EVENTS.reset(token)
        del _HANDLERS[("fake-task", "infer")]

    assert response.status_code == status_code
    assert events == [("ds/1", False, 0.0)]


@pytest.mark.asyncio
async def test_bridge_records_the_attempted_model_before_each_load():
    gateway = FakeGateway()
    bridge = LegacyModelBridge(gateway)
    route = SimpleNamespace(registry_id="ds/1")
    events = []
    token = MODEL_LOAD_EVENTS.set(events)
    try:
        await bridge.ensure_loaded(route, None)
        await bridge.ensure_loaded(route, None)
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert events == [("ds/1", False, 0.0), ("ds/1", False, 0.0)]


def test_workflow_thread_load_is_recorded_into_the_request_events():
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        bridge = LegacyModelBridge(ModelManagerGateway(_SlowManager()))
        sync_bridge = SyncLegacyBridge(bridge, LoopBridge(loop))
        route = Route(
            model_id="ds/1",
            registry_id="ds/1",
            task_type="object-detection",
            action="infer",
        )
        events = []
        token = MODEL_LOAD_EVENTS.set(events)
        try:
            context = contextvars.copy_context()
            worker = threading.Thread(
                target=lambda: context.run(sync_bridge.ensure_loaded, route, "k")
            )
            worker.start()
            worker.join(timeout=10)
        finally:
            MODEL_LOAD_EVENTS.reset(token)

        assert [(model_id, cold) for model_id, cold, _ in events] == [
            ("ds/1", False),
            ("ds/1", True),
        ]
        assert events[1][2] >= 0.2
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=5)
        loop.close()


@pytest.mark.asyncio
async def test_gateway_records_internal_reload():
    manager = _Manager()
    gateway = ModelManagerGateway(manager)
    await gateway.ensure_loaded("m")
    manager.evict_next = True
    events = []
    token = MODEL_LOAD_EVENTS.set(events)
    try:
        await gateway.infer(model_id="m", image=b"x")
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert len(events) == 1
    assert events[0][:2] == ("m", True)


def test_build_headers_merges_remote_entries():
    headers = build_model_response_headers(
        local_model_ids={"local/1"},
        local_cold_start_entries=[("local/1", 0.5)],
        remote_model_ids={"remote/2"},
        remote_cold_start_entries=[("remote/2", 1.5)],
        remote_cold_start_count=1,
        remote_cold_start_total_load_time=1.5,
    )

    assert headers == {
        "X-Model-Cold-Start": "true",
        "X-Model-Cold-Start-Count": "2",
        "X-Model-Id": "local/1,remote/2",
        "X-Model-Load-Time": "2.0",
        "X-Model-Load-Details": '[{"m": "local/1", "t": 0.5}, {"m": "remote/2", "t": 1.5}]',
    }


def test_build_headers_omits_details_when_remote_count_has_no_entries():
    headers = build_model_response_headers(
        local_model_ids=set(),
        local_cold_start_entries=[],
        remote_model_ids={"remote/2"},
        remote_cold_start_entries=[],
        remote_cold_start_count=1,
        remote_cold_start_total_load_time=1.5,
    )

    assert headers == {
        "X-Model-Cold-Start": "true",
        "X-Model-Cold-Start-Count": "1",
        "X-Model-Id": "remote/2",
        "X-Model-Load-Time": "1.5",
    }


def test_build_headers_drops_details_over_size_limit():
    entries = [(f"model-{index}/1", 0.1) for index in range(200)]

    headers = build_model_response_headers(
        local_model_ids={model_id for model_id, _ in entries},
        local_cold_start_entries=entries,
        remote_model_ids=set(),
        remote_cold_start_entries=[],
        remote_cold_start_count=0,
        remote_cold_start_total_load_time=0.0,
    )

    assert headers["X-Model-Cold-Start-Count"] == "200"
    assert "X-Model-Load-Details" not in headers


def test_empty_remote_extension_point_changes_nothing():
    inner = FastAPI()
    seen = []

    @inner.get("/probe")
    async def _probe():
        seen.append(REMOTE_MODEL_LOADS.get())
        return {"ok": True}

    inner.add_middleware(ModelLoadHeadersMiddleware)

    response = TestClient(inner).get("/probe")

    assert seen == [[]]
    assert _model_headers(response) == {
        "x-model-cold-start",
        "x-model-cold-start-count",
    }
    assert response.headers["X-Model-Cold-Start"] == "false"


def _correlation_app():
    inner = FastAPI()

    @inner.get("/probe")
    async def _probe(request: Request):
        return {
            "correlation_id": correlation_id.get(),
            "request_header": request.headers.get(configuration.CORRELATION_ID_HEADER),
        }

    inner.add_middleware(CorrelationIdMiddleware)
    return TestClient(inner)


def test_correlation_id_is_generated_published_and_echoed():
    response = _correlation_app().get("/probe")

    generated = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", generated)
    assert response.json() == {"correlation_id": generated, "request_header": generated}
    assert correlation_id.get() is None


def test_correlation_id_keeps_a_valid_uuid_when_api_logging_is_disabled(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    incoming = str(uuid.uuid4())

    response = _correlation_app().get("/probe", headers={"X-Request-ID": incoming})

    assert response.headers["X-Request-ID"] == incoming
    assert response.json() == {"correlation_id": incoming, "request_header": incoming}


def test_correlation_id_replaces_an_invalid_value_when_api_logging_is_disabled(
    monkeypatch, caplog
):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)

    with caplog.at_level("WARNING", logger="inference_server.middlewares"):
        response = _correlation_app().get("/probe", headers={"X-Request-ID": "req-1"})

    replaced = response.headers["X-Request-ID"]
    assert re.fullmatch(r"[0-9a-f]{32}", replaced)
    assert response.json() == {"correlation_id": replaced, "request_header": replaced}
    assert [record.getMessage() for record in caplog.records] == [
        f"Generated new request ID ({replaced}), since request header value "
        "'req-1' was invalid"
    ]


def test_correlation_id_accepts_any_value_when_api_logging_is_enabled(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", True)

    response = _correlation_app().get("/probe", headers={"X-Request-ID": "req-1"})

    assert response.headers["X-Request-ID"] == "req-1"
    assert response.json() == {"correlation_id": "req-1", "request_header": "req-1"}


def test_correlation_id_uses_configured_header(monkeypatch):
    monkeypatch.setattr(configuration, "CORRELATION_ID_HEADER", "X-Correlation")
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", True)

    response = _correlation_app().get("/probe", headers={"X-Correlation": "abc"})

    assert response.headers["X-Correlation"] == "abc"
    assert response.json() == {"correlation_id": "abc", "request_header": "abc"}
    assert "X-Request-ID" not in response.headers


def test_correlation_id_header_is_x_request_id_when_api_logging_is_disabled(
    monkeypatch,
):
    monkeypatch.setattr(configuration, "CORRELATION_ID_HEADER", "X-Correlation")
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    inner = FastAPI()

    @inner.get("/probe")
    async def _probe(request: Request):
        return {
            "correlation_id": correlation_id.get(),
            "x_request_id": request.headers.get("X-Request-ID"),
            "x_correlation": request.headers.get("X-Correlation"),
        }

    inner.add_middleware(CorrelationIdMiddleware)
    incoming = uuid.uuid4().hex

    response = TestClient(inner).get(
        "/probe", headers={"X-Request-ID": incoming, "X-Correlation": "abc"}
    )

    assert response.headers["X-Request-ID"] == incoming
    assert "X-Correlation" not in response.headers
    assert response.json() == {
        "correlation_id": incoming,
        "x_request_id": incoming,
        "x_correlation": "abc",
    }


def test_correlation_id_is_appended_to_a_response_that_already_carries_one():
    inner = FastAPI()

    @inner.get("/probe")
    async def _probe():
        return Response(content=b"{}", headers={"X-Request-ID": "route-set"})

    inner.add_middleware(CorrelationIdMiddleware)

    response = TestClient(inner).get("/probe")

    echoed = response.headers.get_list("x-request-id")
    assert len(echoed) == 2
    assert echoed[0] == "route-set"
    assert re.fullmatch(r"[0-9a-f]{32}", echoed[1])


def test_correlation_id_already_echoed_by_the_app_is_appended_again():
    inner = FastAPI()

    @inner.get("/probe")
    async def _probe():
        return Response(content=b"{}", headers={"X-Request-ID": correlation_id.get()})

    inner.add_middleware(CorrelationIdMiddleware)

    response = TestClient(inner).get("/probe")

    echoed = response.headers.get_list("x-request-id")
    assert len(echoed) == 2
    assert echoed[0] == echoed[1]
    assert re.fullmatch(r"[0-9a-f]{32}", echoed[0])


@pytest.fixture
def serverless_app(monkeypatch):
    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", True)
    module = importlib.reload(app_mod)
    yield module.app
    monkeypatch.undo()
    importlib.reload(app_mod)


def test_serverless_denial_echoes_supplied_execution_id(serverless_app):
    request_id = uuid.uuid4().hex

    response = TestClient(serverless_app).post(
        "/infer/object_detection",
        json={},
        headers={"X-Request-ID": request_id, "execution_id": "exec-7"},
    )

    assert response.status_code == 401
    assert response.headers["execution_id"] == "exec-7"
    assert response.headers.get_list("x-request-id") == [request_id, request_id]
    assert float(response.headers["X-Processing-Time"]) >= 0.0
    assert response.headers["X-Model-Cold-Start"] == "false"
    assert response.headers["x-inference-engine"] == "inference-models"
    assert execution_id.get() is None


@pytest.mark.parametrize(
    "api_logging_enabled,custom_count,request_id_count",
    [(False, 1, 1), (True, 2, 0)],
)
def test_serverless_denial_carries_the_legacy_correlation_header_pair(
    serverless_app, monkeypatch, api_logging_enabled, custom_count, request_id_count
):
    monkeypatch.setattr(configuration, "CORRELATION_ID_HEADER", "X-Custom")
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", api_logging_enabled)

    response = TestClient(serverless_app).post("/infer/object_detection", json={})

    assert response.status_code == 401
    custom = response.headers.get_list("x-custom")
    request_id = response.headers.get_list("x-request-id")
    assert len(custom) == custom_count
    assert len(request_id) == request_id_count
    assert len(set(custom + request_id)) == 1
    assert re.fullmatch(r"[0-9a-f]{32}", custom[0])


def test_serverless_denial_carries_generated_execution_id(serverless_app):
    response = TestClient(serverless_app).post("/infer/object_detection", json={})

    assert response.status_code == 401
    assert re.fullmatch(r"\d+_[0-9a-f]{4}", response.headers["execution_id"])
    echoed = response.headers.get_list("x-request-id")
    assert len(echoed) == 2
    assert echoed[0] == echoed[1]
    assert re.fullmatch(r"[0-9a-f]{32}", echoed[0])


def test_serverless_credit_denial_carries_workspace_header(monkeypatch):
    from inference_server import platform_http

    def _request(method, url, **kwargs):
        return SimpleNamespace(
            status_code=402, json=lambda: {"workspace": "ws-1", "underCap": False}
        )

    monkeypatch.setattr(platform_http, "_platform_request", _request)
    monkeypatch.setattr(serverless_auth, "_cache", {})
    inner = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @inner.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe(request: Request):
        return JSONResponse({"ok": True})

    inner.add_middleware(ServerlessAuthMiddleware)
    inner.add_middleware(ModelLoadHeadersMiddleware)
    inner.add_middleware(CorrelationIdMiddleware)

    response = TestClient(inner).post("/infer/object_detection?api_key=k", json={})

    assert response.status_code == 402
    assert response.headers["X-Workspace-Id"] == "ws-1"
    assert float(response.headers["X-Processing-Time"]) >= 0.0
    echoed = response.headers.get_list("x-request-id")
    assert len(echoed) == 2
    assert echoed[0] == echoed[1]
    assert re.fullmatch(r"[0-9a-f]{32}", echoed[0])


def test_serverless_denial_carries_the_trace_id_of_the_active_span(monkeypatch):
    from inference_server import telemetry

    monkeypatch.setattr(serverless_auth, "_cache", {})
    monkeypatch.setattr(telemetry, "get_trace_id", lambda: "ab" * 16)
    inner = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @inner.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe(request: Request):
        return JSONResponse({"ok": True})

    inner.add_middleware(ServerlessAuthMiddleware)
    inner.add_middleware(ModelLoadHeadersMiddleware)
    inner.add_middleware(CorrelationIdMiddleware)
    inner.add_middleware(telemetry.TraceIdResponseMiddleware)

    response = TestClient(inner).post("/infer/object_detection", json={})

    assert response.status_code == 401
    assert response.headers["X-Trace-Id"] == "ab" * 16
    assert float(response.headers["X-Processing-Time"]) >= 0.0
    assert len(response.headers.get_list("x-request-id")) == 2


def test_observability_middlewares_wrap_every_other_middleware():
    classes = [entry.cls for entry in app_mod.app.user_middleware]

    assert classes[:2] == [CorrelationIdMiddleware, ModelLoadHeadersMiddleware]


def test_cors_exposes_legacy_response_headers():
    cors = next(
        entry
        for entry in app_mod.app.user_middleware
        if entry.cls is PathAwareCORSMiddleware
        and entry.kwargs.get("match_paths") == r"^(?!/build).*"
    )

    assert cors.kwargs["expose_headers"] == [
        "X-Processing-Time",
        "X-Remote-Processing-Time",
        "X-Remote-Processing-Times",
        "X-Model-Cold-Start",
        "X-Model-Cold-Start-Count",
        "X-Model-Load-Time",
        "X-Model-Load-Details",
        "X-Model-Id",
        "X-Workflow-Id",
        "X-Workspace-Id",
        "X-Trace-Id",
        "execution_id",
        "traceparent",
        "tracestate",
    ]


@pytest.mark.parametrize(
    "flag,value,message",
    [
        (
            "LAMBDA",
            True,
            "OFFLINE_MODE is not supported together with LAMBDA / "
            "GCP_SERVERLESS deployments because authentication and usage "
            "accounting require API connectivity.",
        ),
        (
            "GCP_SERVERLESS",
            True,
            "OFFLINE_MODE is not supported together with LAMBDA / "
            "GCP_SERVERLESS deployments because authentication and usage "
            "accounting require API connectivity.",
        ),
        (
            "DEDICATED_DEPLOYMENT_WORKSPACE_URL",
            "ws-url",
            "OFFLINE_MODE is not supported together with dedicated or "
            "workspace-whitelist authentication because API keys cannot be "
            "mapped to workspaces without API connectivity.",
        ),
        (
            "WORKSPACES_WHITELISTED_FOR_LOCAL_DEPLOYMENT",
            ["ws-a"],
            "OFFLINE_MODE is not supported together with dedicated or "
            "workspace-whitelist authentication because API keys cannot be "
            "mapped to workspaces without API connectivity.",
        ),
    ],
)
def test_offline_mode_refuses_hosted_flags(monkeypatch, flag, value, message):
    monkeypatch.setattr(app_mod._cfg, "OFFLINE_MODE", True)
    monkeypatch.setattr(app_mod._cfg, flag, value)

    with pytest.raises(RuntimeError) as error:
        app_mod._ensure_offline_mode_is_supported()

    assert str(error.value) == message


def test_offline_mode_alone_and_hosted_flags_alone_start(monkeypatch):
    monkeypatch.setattr(app_mod._cfg, "OFFLINE_MODE", True)
    app_mod._ensure_offline_mode_is_supported()

    monkeypatch.setattr(app_mod._cfg, "OFFLINE_MODE", False)
    monkeypatch.setattr(app_mod._cfg, "GCP_SERVERLESS", True)
    monkeypatch.setattr(app_mod._cfg, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", "ws-url")
    app_mod._ensure_offline_mode_is_supported()
