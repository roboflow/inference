import base64
import asyncio
import copy
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi import Request, Response
from PIL import Image
from roboflow_workflows.utils.in_memory_cache import InMemoryWorkflowsCache

from inference_server import configuration
from inference_server.active_learning import middlewares as middlewares_module
from inference_server.active_learning.cache_operations import (
    get_current_strategy_limit_usage,
)
from inference_server.active_learning.entities import StrategyLimitType
from inference_server.framework.dispatch import handle_model_inference_request
from inference_server.framework.entities import (
    ModelHandlerDescription,
    ModelInterfaceDescription,
)
from inference_server.framework.registry import _HANDLERS
from inference_server.legacy import active_learning_registration as registration
from inference_server.legacy import entities as entities_module
from inference_server.legacy import router as router_module
from inference_server.legacy.bridge import Route
from inference_server.workflows.models_provider import GatewayModelsProvider
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.workflows.test_models_provider import FakeSyncBridge

FORM_HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}
IMAGE_URL = "https://example.com/a.png"
SECRET = "secret-key-123"


def _png(width=8, height=6):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _png_b64():
    return base64.b64encode(_png()).decode()


def _image():
    return {"type": "base64", "value": _png_b64()}


def _expected_bgr():
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[..., 2] = 255
    return image


def _det():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


def _gateway(model_id="ds/1", prediction=None, **model_info):
    info = {"class_names": ["cat"], "actions": {"infer": {}}}
    info.update(model_info)
    return FakeGateway(
        predictions={(model_id, "infer"): prediction or _det()},
        model_info={model_id: info},
    )


@pytest.fixture
def active_learning(monkeypatch):
    state = SimpleNamespace(
        created=[],
        calls=[],
        submitted=[],
        started=threading.Event(),
        hold=None,
        error=None,
        mutate=False,
        inactive_keys=set(),
    )

    class FakeMiddleware:
        @property
        def active(self):
            return self.api_key not in state.inactive_keys

        @classmethod
        def init(cls, api_key, target_dataset, model_id, cache, platform_client):
            middleware = cls()
            middleware.api_key = api_key
            middleware.target_dataset = target_dataset
            middleware.model_id = model_id
            middleware.cache = cache
            middleware.platform_client = platform_client
            state.created.append(middleware)
            return middleware

        def register_batch(self, images, predictions, prediction_type, inference_id):
            state.started.set()
            if state.hold is not None:
                state.hold.wait(10)
            if state.error is not None:
                raise state.error
            state.calls.append(
                SimpleNamespace(
                    middleware=self,
                    images=images,
                    predictions=copy.deepcopy(predictions),
                    prediction_type=prediction_type,
                    inference_id=inference_id,
                )
            )
            if state.mutate:
                for prediction in predictions:
                    prediction["predictions"].clear()
                    prediction["image"]["width"] = -1

    real_submit = registration.ActiveLearningRegistrar.submit

    def recording_submit(self, job):
        state.submitted.append(job)
        return real_submit(self, job)

    monkeypatch.setattr(middlewares_module, "ActiveLearningMiddleware", FakeMiddleware)
    monkeypatch.setattr(
        registration.ActiveLearningRegistrar, "submit", recording_submit
    )
    monkeypatch.setattr(configuration, "ACTIVE_LEARNING_ENABLED", True)
    monkeypatch.setattr(configuration, "LAMBDA", False)
    monkeypatch.setattr(configuration, "GCP_SERVERLESS", False)
    return state


def _settle():
    registrar = registration._REGISTRAR
    deadline = time.monotonic() + 5
    while registrar._pending and time.monotonic() < deadline:
        time.sleep(0.01)
    assert registrar._pending == 0


def _post_detection(client, **overrides):
    body = {"model_id": "ds/1", "api_key": "k", "image": _image()}
    body.update(overrides)
    return client.post("/infer/object_detection", json=body)


def _post_catch_all(client, query="api_key=k"):
    return client.post(f"/ds/1?{query}", content=_png_b64(), headers=FORM_HEADERS)


def _detection_client(legacy_client, fake_stat, task_type="object-detection"):
    fake_stat["ds/1"] = (task_type, "infer")
    return legacy_client(_gateway())


def test_object_detection_route_registers_the_request(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    response = _post_detection(client)
    _settle()

    assert response.status_code == 200, response.text
    assert len(active_learning.calls) == 1
    call = active_learning.calls[0]
    assert len(call.images) == 1
    assert np.array_equal(call.images[0], _expected_bgr())
    assert call.prediction_type == "object-detection"
    assert call.inference_id == response.json()["inference_id"]
    assert len(call.predictions) == 1
    assert call.predictions[0]["image"] == {"width": 8, "height": 6}
    assert call.predictions[0]["predictions"][0]["class"] == "cat"
    assert "visualization" not in call.predictions[0]


def test_registration_dump_has_no_visualization_when_one_is_rendered(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    response = _post_detection(client, visualize_predictions=True)
    _settle()

    assert response.status_code == 200, response.text
    assert "visualization" in response.json()
    assert "visualization" not in active_learning.calls[0].predictions[0]


def test_instance_segmentation_route_registers_the_request(
    legacy_client, fake_stat, active_learning
):
    mask = np.zeros((1, 6, 8), dtype=bool)
    mask[0, 1:4, 1:4] = True
    prediction = SimpleNamespace(
        xyxy=np.array([[1, 1, 4, 4]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        mask=mask,
    )
    fake_stat["ds/1"] = ("instance-segmentation", "infer")
    client = legacy_client(_gateway(prediction=prediction))

    response = client.post(
        "/infer/instance_segmentation",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )
    _settle()

    assert response.status_code == 200, response.text
    assert len(active_learning.calls) == 1
    assert active_learning.calls[0].prediction_type == "instance-segmentation"


def test_classification_route_registers_the_request(
    legacy_client, fake_stat, active_learning
):
    prediction = [
        SimpleNamespace(confidence=np.array([0.2, 0.8]), class_id=np.array([1]))
    ]
    fake_stat["ds/1"] = ("classification", "infer")
    client = legacy_client(_gateway(prediction=prediction, class_names=["a", "b"]))

    response = client.post(
        "/infer/classification",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )
    _settle()

    assert response.status_code == 200, response.text
    assert len(active_learning.calls) == 1
    assert active_learning.calls[0].prediction_type == "classification"
    assert active_learning.calls[0].predictions[0]["top"] == "b"


def test_catch_all_route_registers_the_request(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    response = _post_catch_all(client)
    _settle()

    assert response.status_code == 200, response.text
    assert len(active_learning.calls) == 1
    call = active_learning.calls[0]
    assert np.array_equal(call.images[0], _expected_bgr())
    assert call.inference_id == response.json()["inference_id"]
    assert call.middleware.model_id == "ds/1"
    assert call.middleware.target_dataset == "ds"


def _keypoints_gateway():
    keypoints = SimpleNamespace(
        xy=np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        class_id=np.array([0]),
        confidence=np.array([[0.9, 0.8]]),
    )
    return _gateway(
        prediction=([keypoints], [_det()]),
        class_names=["person"],
        key_points_classes=[["nose", "eye"]],
    )


def test_catch_all_route_registers_a_keypoints_model(
    legacy_client, fake_stat, active_learning
):
    fake_stat["ds/1"] = ("keypoint-detection", "infer")
    client = legacy_client(_keypoints_gateway())

    response = _post_catch_all(client)
    _settle()

    assert response.status_code == 200, response.text
    assert len(active_learning.calls) == 1
    assert active_learning.calls[0].prediction_type == "keypoint-detection"


def test_keypoints_route_does_not_register(legacy_client, fake_stat, active_learning):
    fake_stat["ds/1"] = ("keypoint-detection", "infer")
    client = legacy_client(_keypoints_gateway())

    response = client.post(
        "/infer/keypoints_detection",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert response.status_code == 200, response.text
    assert active_learning.submitted == []
    assert active_learning.created == []


def test_core_model_route_does_not_register(legacy_client, fake_stat, active_learning):
    gateway = FakeGateway(
        predictions={("clip/ViT-B-16", "embed_images"): np.array([[0.5, 0.5]])},
        model_info={"clip/ViT-B-16": {"actions": {"embed_images": {}}}},
    )

    response = legacy_client(gateway).post(
        "/clip/embed_image", json={"image": _image(), "api_key": "k"}
    )

    assert response.status_code == 200, response.text
    assert active_learning.submitted == []


def test_workflow_step_does_not_register(legacy_client, fake_stat, active_learning):
    legacy_client(_gateway())
    bridge = FakeSyncBridge()
    bridge.routes["ds/1"] = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="object-detection",
        action="infer",
        actions={"infer"},
        class_names=["cat"],
    )
    bridge.predictions[("ds/1", "infer")] = _det()
    provider = GatewayModelsProvider(bridge, api_key="k")
    provider.add_model("ds/1", "k")

    output = provider.run_object_detection(
        "ds/1",
        [{"type": "numpy_object", "value": np.zeros((4, 6, 3), dtype=np.uint8)}],
        api_key="k",
        confidence=0.5,
    )

    assert output[0]["predictions"][0]["class"] == "cat"
    assert registration._REGISTRAR is not None
    assert active_learning.submitted == []


@pytest.mark.asyncio
async def test_v2_inference_does_not_register(active_learning):
    interface = ModelInterfaceDescription(task="t", params={}, output_schema={})
    description = ModelHandlerDescription(
        input_parser=AsyncMock(return_value={"images": [b"x"], "params": {}}),
        handler=AsyncMock(return_value=MagicMock()),
        output_serializer=MagicMock(
            return_value=Response(status_code=200, content=b"ok")
        ),
        interface_provider=lambda: interface,
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/models/infer",
        "query_string": b"model_id=m",
        "headers": [(b"authorization", b"Bearer k1")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    proxy = MagicMock()
    proxy.ensure_loaded = AsyncMock(return_value=("model_ready",))
    proxy.infer = AsyncMock(return_value=MagicMock())
    registration.start()
    _HANDLERS[("active-learning-pin", "infer")] = description
    try:
        with patch(
            "inference_server.framework.dispatch.stat_model_while_checking_auth",
            new=AsyncMock(return_value=("active-learning-pin", "infer")),
        ):
            response = await handle_model_inference_request(
                Request(scope, receive), proxy
            )
        registered = registration._REGISTRAR is not None
    finally:
        del _HANDLERS[("active-learning-pin", "infer")]
        registration.stop()

    assert response.status_code == 200
    assert registered
    assert active_learning.submitted == []


def test_semantic_segmentation_route_is_a_silent_no_op(
    legacy_client, fake_stat, active_learning, caplog
):
    prediction = SimpleNamespace(
        segmentation_map=np.array([[0, 1], [1, 0]]),
        confidence=np.array([[1.0, 0.5], [0.5, 1.0]]),
    )
    fake_stat["ds/1"] = ("semantic-segmentation", "infer")
    client = legacy_client(_gateway(prediction=prediction, class_names=["bg", "fg"]))

    with caplog.at_level(logging.DEBUG, logger=registration.__name__):
        response = client.post(
            "/infer/semantic_segmentation",
            json={"model_id": "ds/1", "api_key": "k", "image": _image()},
        )

    assert response.status_code == 200, response.text
    assert active_learning.submitted == []
    assert [r for r in caplog.records if r.name == registration.__name__] == []


def test_setting_turns_active_learning_off(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    monkeypatch.setattr(configuration, "ACTIVE_LEARNING_ENABLED", False)
    client = _detection_client(legacy_client, fake_stat)

    response = _post_detection(client)

    assert response.status_code == 200, response.text
    assert registration._REGISTRAR is None
    assert active_learning.submitted == []


def test_body_field_disables_registration(legacy_client, fake_stat, active_learning):
    client = _detection_client(legacy_client, fake_stat)

    response = _post_detection(client, disable_active_learning=True)

    assert response.status_code == 200, response.text
    assert active_learning.submitted == []


def test_catch_all_query_parameter_disables_registration(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    response = _post_catch_all(client, query="api_key=k&disable_active_learning=true")

    assert response.status_code == 200, response.text
    assert active_learning.submitted == []


def test_request_without_api_key_does_not_register(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    response = client.post(
        "/infer/object_detection", json={"model_id": "ds/1", "image": _image()}
    )

    assert response.status_code == 200, response.text
    assert active_learning.submitted == []


def test_server_works_without_the_workflows_package(
    legacy_client, fake_stat, active_learning, monkeypatch, caplog
):
    monkeypatch.setattr(registration, "_workflows_installed", lambda: False)

    with caplog.at_level(logging.INFO, logger=registration.__name__):
        client = _detection_client(legacy_client, fake_stat)
        response = _post_detection(client)

    assert response.status_code == 200, response.text
    assert registration._REGISTRAR is None
    assert active_learning.submitted == []
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert messages == ["Active learning disabled: roboflow-workflows is not installed"]


def _pin_response_fields(monkeypatch):
    monkeypatch.setattr(
        router_module, "time", SimpleNamespace(perf_counter=lambda: 1.0)
    )
    monkeypatch.setattr(entities_module, "uuid4", lambda: "fixed-detection-id")


def test_response_body_is_the_same_with_and_without_active_learning(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    _pin_response_fields(monkeypatch)
    monkeypatch.setattr(configuration, "LAMBDA", True)
    active_learning.mutate = True
    client = _detection_client(legacy_client, fake_stat)

    registered = _post_detection(client, id="fixed-id")
    calls_after_first = len(active_learning.calls)
    plain = _post_detection(client, id="fixed-id", disable_active_learning=True)

    assert registered.status_code == plain.status_code == 200
    assert calls_after_first == len(active_learning.calls) == 1
    assert registered.content == plain.content
    assert registered.json()["predictions"][0]["class"] == "cat"


def test_registration_dump_shares_nothing_with_the_response(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    served = []
    real_response = router_module.orjson_response

    def recording_response(obj, **kwargs):
        served.append(obj)
        return real_response(obj, **kwargs)

    monkeypatch.setattr(router_module, "orjson_response", recording_response)
    active_learning.mutate = True
    client = _detection_client(legacy_client, fake_stat)

    response = _post_detection(client)
    _settle()

    assert response.status_code == 200, response.text
    assert len(active_learning.calls) == 1
    assert len(served[0].predictions) == 1
    assert served[0].image.width == 8


def test_target_dataset_comes_from_the_request_field(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    _post_detection(client, active_learning_target_dataset="other")
    _settle()

    assert len(active_learning.created) == 1
    assert active_learning.created[0].target_dataset == "other"
    assert active_learning.created[0].model_id == "ds/1"


def test_target_dataset_defaults_to_the_dataset_of_the_model(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    _post_detection(client)
    _settle()

    assert len(active_learning.created) == 1
    assert active_learning.created[0].target_dataset == "ds"
    assert active_learning.created[0].model_id == "ds/1"


def test_alias_model_id_is_resolved_for_model_and_target(
    legacy_client, fake_stat, active_learning
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    client = legacy_client(_gateway(model_id="coco/3"))

    response = _post_detection(client, model_id="yolov8n-640")
    _settle()

    assert response.status_code == 200, response.text
    assert len(active_learning.created) == 1
    assert active_learning.created[0].model_id == "coco/3"
    assert active_learning.created[0].target_dataset == "coco"


def test_middleware_gets_the_cache_and_platform_client_of_the_server(
    legacy_client, fake_stat, active_learning
):
    from inference_server.workflows import host

    client = _detection_client(legacy_client, fake_stat)

    _post_detection(client)
    _settle()

    cache = active_learning.created[0].cache
    assert isinstance(cache, registration._ConfigurationFreeCache)
    assert cache._cache is host.WORKFLOWS_CACHE
    assert active_learning.created[0].platform_client is host.PLATFORM_CLIENT


def test_each_api_key_gets_its_own_middleware(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    for api_key in ("key-a", "key-b", "key-a"):
        _post_detection(client, api_key=api_key)
        _settle()

    assert [m.api_key for m in active_learning.created] == ["key-a", "key-b"]
    assert [c.middleware.api_key for c in active_learning.calls] == [
        "key-a",
        "key-b",
        "key-a",
    ]
    assert active_learning.calls[0].middleware is active_learning.calls[2].middleware


def _job(api_key="k", model_id="ds/1", target_dataset="ds"):
    return registration._Job(
        api_key=api_key,
        model_id=model_id,
        target_dataset=target_dataset,
        prediction_type="object-detection",
        inference_id="id",
        images=[],
        responses=[],
    )


class _CountingFactory:
    def __init__(self):
        self.created = []
        self.delay = 0.0

    def __call__(self, api_key, target_dataset, model_id):
        time.sleep(self.delay)
        middleware = MagicMock()
        self.created.append((api_key, target_dataset, model_id))
        return middleware


@pytest.fixture
def registrar_parts():
    factory = _CountingFactory()
    clock = SimpleNamespace(now=1000.0)
    registrar = registration.ActiveLearningRegistrar(
        create_middleware=factory, max_age_s=900, clock=lambda: clock.now
    )
    yield registrar, factory, clock
    registrar.shutdown()


def test_state_keeps_at_most_the_fixed_number_of_middlewares(registrar_parts):
    registrar, factory, _ = registrar_parts

    for index in range(registration.MAX_MIDDLEWARES + 44):
        registrar.run(_job(api_key=f"key-{index}"))
    registrar.run(_job(api_key="key-0"))

    assert registration.MAX_MIDDLEWARES == 256
    assert len(registrar._slots) == 256
    assert len(factory.created) == 256 + 44 + 1


def test_middleware_is_rebuilt_after_the_configuration_lifetime(registrar_parts):
    registrar, factory, clock = registrar_parts

    registrar.run(_job())
    clock.now += 899
    registrar.run(_job())
    created_before_expiry = len(factory.created)
    clock.now += 1
    registrar.run(_job())
    registrar.run(_job())

    assert created_before_expiry == 1
    assert len(factory.created) == 2


def test_configuration_lifetime_is_the_one_of_the_package(active_learning):
    from inference_server.active_learning.configuration import (
        ACTIVE_LEARNING_CONFIG_CACHE_EXPIRE,
    )

    registrar = registration.start()
    try:
        assert ACTIVE_LEARNING_CONFIG_CACHE_EXPIRE == 900
        assert registrar._max_age_s == 900
    finally:
        registration.stop()


def test_concurrent_first_requests_create_one_middleware(registrar_parts):
    registrar, factory, _ = registrar_parts
    factory.delay = 0.2
    threads = [threading.Thread(target=registrar.run, args=(_job(),)) for _ in range(8)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert len(factory.created) == 1


def test_response_is_returned_before_a_slow_registration_finishes(
    legacy_client, fake_stat, active_learning
):
    active_learning.hold = threading.Event()
    client = _detection_client(legacy_client, fake_stat)

    try:
        response = _post_detection(client)
        started = active_learning.started.wait(5)
        calls_while_held = len(active_learning.calls)
    finally:
        active_learning.hold.set()
    _settle()

    assert response.status_code == 200, response.text
    assert started
    assert calls_while_held == 0
    assert len(active_learning.calls) == 1


def test_job_over_the_pending_bound_is_dropped_and_counted(
    legacy_client, fake_stat, active_learning, monkeypatch, caplog
):
    monkeypatch.setattr(registration, "MAX_PENDING_JOBS", 2)
    active_learning.hold = threading.Event()
    client = _detection_client(legacy_client, fake_stat)

    try:
        with caplog.at_level(logging.DEBUG, logger=registration.__name__):
            responses = [_post_detection(client) for _ in range(3)]
        dropped = registration._REGISTRAR.dropped
    finally:
        active_learning.hold.set()
    _settle()

    assert [r.status_code for r in responses] == [200, 200, 200]
    assert registration.MAX_PENDING_JOBS == 2
    assert dropped == 1
    assert len(active_learning.calls) == 2
    assert [r for r in caplog.records if r.name == registration.__name__] == []


def test_pending_bound_and_pool_size_constants():
    assert registration.MAX_PENDING_JOBS == 64
    assert registration.WORKERS == 4


def test_failing_registration_leaves_the_response_and_logs_the_class_name(
    legacy_client, fake_stat, active_learning, monkeypatch, caplog
):
    _pin_response_fields(monkeypatch)
    client = _detection_client(legacy_client, fake_stat)
    plain = _post_detection(client, id="fixed-id", disable_active_learning=True)
    active_learning.error = RuntimeError(f"platform said no to {SECRET}")

    with caplog.at_level(logging.DEBUG):
        failed = _post_detection(client, id="fixed-id")
        _settle()

    assert failed.status_code == plain.status_code == 200
    assert failed.content == plain.content
    warnings = [
        r.getMessage() for r in caplog.records if r.name == registration.__name__
    ]
    assert warnings == ["Active learning registration failed: RuntimeError"]
    assert SECRET not in caplog.text


@pytest.mark.parametrize("setting", ["LAMBDA", "GCP_SERVERLESS"])
def test_serverless_registration_finishes_before_the_response(
    legacy_client, fake_stat, active_learning, monkeypatch, setting
):
    monkeypatch.setattr(configuration, setting, True)
    client = _detection_client(legacy_client, fake_stat)

    response = _post_catch_all(client)
    calls_at_response = len(active_learning.calls)

    assert response.status_code == 200, response.text
    assert calls_at_response == 1


def test_serverless_registration_does_not_block_the_event_loop(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    monkeypatch.setattr(configuration, "LAMBDA", True)
    active_learning.hold = threading.Event()
    client = _detection_client(legacy_client, fake_stat)
    results = {}

    def infer():
        results["infer"] = _post_catch_all(client)

    def other():
        results["other"] = _post_detection(client, disable_active_learning=True)

    infer_thread = threading.Thread(target=infer)
    other_thread = threading.Thread(target=other)
    try:
        infer_thread.start()
        started = active_learning.started.wait(5)
        other_thread.start()
        other_thread.join(5)
        served_while_registering = "other" in results
        inference_answered_while_registering = "infer" in results
    finally:
        active_learning.hold.set()
    infer_thread.join(5)
    other_thread.join(5)

    assert started
    assert served_while_registering
    assert not inference_answered_while_registering
    assert results["other"].status_code == 200
    assert results["infer"].status_code == 200
    assert len(active_learning.calls) == 1


def test_serverless_job_over_the_pending_bound_is_skipped_and_counted(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    monkeypatch.setattr(configuration, "LAMBDA", True)
    monkeypatch.setattr(registration, "MAX_PENDING_JOBS", 0)
    client = _detection_client(legacy_client, fake_stat)

    response = _post_catch_all(client)

    assert response.status_code == 200, response.text
    assert registration._REGISTRAR.dropped == 1
    assert active_learning.calls == []


def test_url_image_is_fetched_once(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    fetched = []

    async def _fetch(urls, destination_policy=None):
        fetched.append(urls)
        return [_png() for _ in urls], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)
    client = _detection_client(legacy_client, fake_stat)

    response = client.post(f"/ds/1?api_key=k&image={IMAGE_URL}")
    _settle()

    assert response.status_code == 200, response.text
    assert fetched == [[IMAGE_URL]]
    assert len(active_learning.calls) == 1
    assert np.array_equal(active_learning.calls[0].images[0], _expected_bgr())


def _inactive_job_parts():
    decoded = []
    dumped = []
    middleware = MagicMock()
    middleware.active = False
    response = MagicMock()
    response.model_dump.side_effect = lambda **kwargs: dumped.append(kwargs)
    job = registration._Job(
        api_key="k",
        model_id="ds/1",
        target_dataset="ds",
        prediction_type="object-detection",
        inference_id="id",
        images=[b"image"],
        responses=[response],
    )
    return middleware, job, decoded, dumped


def test_inactive_project_job_neither_decodes_nor_dumps(monkeypatch):
    middleware, job, decoded, dumped = _inactive_job_parts()
    monkeypatch.setattr(
        registration, "_decode_image", lambda data: decoded.append(data)
    )
    registrar = registration.ActiveLearningRegistrar(
        create_middleware=lambda **kwargs: middleware, max_age_s=900
    )

    try:
        registrar.run(job)
    finally:
        registrar.shutdown()

    assert decoded == []
    assert dumped == []
    middleware.register_batch.assert_not_called()


def _inactive_project_client(legacy_client, fake_stat, active_learning):
    active_learning.inactive_keys.add("off")
    return _detection_client(legacy_client, fake_stat)


def test_known_inactive_project_submits_no_further_jobs(
    legacy_client, fake_stat, active_learning
):
    client = _inactive_project_client(legacy_client, fake_stat, active_learning)

    _post_detection(client, api_key="off")
    _settle()
    responses = [_post_detection(client, api_key="off") for _ in range(200)]

    assert {r.status_code for r in responses} == {200}
    assert len(active_learning.submitted) == 1
    assert registration._REGISTRAR.dropped == 0
    assert registration._REGISTRAR._pending == 0
    assert active_learning.calls == []


def test_inactive_project_is_checked_again_after_the_configuration_lifetime(
    legacy_client, fake_stat, active_learning
):
    client = _inactive_project_client(legacy_client, fake_stat, active_learning)
    clock = SimpleNamespace(now=1000.0)
    registration._REGISTRAR._clock = lambda: clock.now

    _post_detection(client, api_key="off")
    _settle()
    clock.now += 899
    _post_detection(client, api_key="off")
    submitted_before_expiry = len(active_learning.submitted)
    clock.now += 1
    _post_detection(client, api_key="off")
    _settle()
    _post_detection(client, api_key="off")

    assert submitted_before_expiry == 1
    assert len(active_learning.submitted) == 2
    assert len(active_learning.created) == 2


def test_active_project_submits_one_job_per_request(
    legacy_client, fake_stat, active_learning
):
    client = _detection_client(legacy_client, fake_stat)

    for _ in range(3):
        _post_detection(client)
        _settle()

    assert len(active_learning.submitted) == 3
    assert len(active_learning.calls) == 3


def test_known_inactive_project_does_not_touch_a_full_pending_bound(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    monkeypatch.setattr(registration, "MAX_PENDING_JOBS", 2)
    client = _inactive_project_client(legacy_client, fake_stat, active_learning)
    _post_detection(client, api_key="off")
    _settle()
    active_learning.hold = threading.Event()

    try:
        for _ in range(2):
            _post_detection(client, api_key="on")
        pending_when_full = registration._REGISTRAR._pending
        for _ in range(5):
            _post_detection(client, api_key="off")
        pending_after = registration._REGISTRAR._pending
        dropped = registration._REGISTRAR.dropped
        submitted = len(active_learning.submitted)
    finally:
        active_learning.hold.set()
    _settle()

    assert pending_when_full == 2
    assert pending_after == 2
    assert dropped == 0
    assert submitted == 3


def test_serverless_known_inactive_project_waits_for_nothing(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    monkeypatch.setattr(configuration, "LAMBDA", True)
    client = _inactive_project_client(legacy_client, fake_stat, active_learning)
    _post_detection(client, api_key="off")
    monkeypatch.setattr(registration, "MAX_PENDING_JOBS", 0)

    response = _post_detection(client, api_key="off")

    assert response.status_code == 200, response.text
    assert len(active_learning.submitted) == 1
    assert registration._REGISTRAR.dropped == 0


def test_lifespan_creates_and_shuts_down_the_pool(active_learning, monkeypatch):
    from fastapi.testclient import TestClient

    import inference_server.app as app_mod

    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: _gateway()
    )
    with TestClient(app_mod.app, raise_server_exceptions=False):
        registrar = registration._REGISTRAR
        running = registrar is not None and not registrar._executor._shutdown

    assert running
    assert registration._REGISTRAR is None
    assert registrar._executor._shutdown
    assert registrar.submit(_job()) is None
    assert registrar.dropped == 1


def test_lifespan_creates_no_pool_when_active_learning_is_off(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    monkeypatch.setattr(configuration, "ACTIVE_LEARNING_ENABLED", False)

    legacy_client(_gateway())

    assert registration._REGISTRAR is None


def test_gray_numpy_payload_is_registered_as_bgr():
    gray = np.full((4, 5), 7, dtype=np.uint8)

    image = registration._decode_image(gray)

    assert image.shape == (4, 5, 3)
    assert np.array_equal(image[..., 0], gray)


SERVERLESS_TIMEOUT_S = 0.3


def _post_in_thread(client, results, name):
    def post():
        results[name] = _post_catch_all(client)

    thread = threading.Thread(target=post)
    thread.start()
    return thread


@pytest.mark.parametrize("setting", ["LAMBDA", "GCP_SERVERLESS"])
def test_serverless_wait_is_released_at_the_deadline(
    legacy_client, fake_stat, active_learning, monkeypatch, setting
):
    monkeypatch.setattr(configuration, setting, True)
    monkeypatch.setattr(registration, "SERVERLESS_WAIT_TIMEOUT_S", SERVERLESS_TIMEOUT_S)
    active_learning.hold = threading.Event()
    client = _detection_client(legacy_client, fake_stat)
    registrar = registration._REGISTRAR

    try:
        started_at = time.monotonic()
        response = _post_catch_all(client)
        waited = time.monotonic() - started_at
        timed_out = registrar.timed_out
        calls_while_held = len(active_learning.calls)
    finally:
        active_learning.hold.set()
    _settle()

    assert response.status_code == 200, response.text
    assert SERVERLESS_TIMEOUT_S <= waited < 5
    assert timed_out == 1
    assert calls_while_held == 0
    assert registrar._pending == 0
    assert len(active_learning.calls) == 1


def test_serverless_wait_default_deadline_is_thirty_seconds():
    assert registration.SERVERLESS_WAIT_TIMEOUT_S == 30


def test_serverless_wait_covers_the_time_in_the_queue(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    monkeypatch.setattr(configuration, "LAMBDA", True)
    monkeypatch.setattr(registration, "SERVERLESS_WAIT_TIMEOUT_S", SERVERLESS_TIMEOUT_S)
    monkeypatch.setattr(registration, "WORKERS", 1)
    active_learning.hold = threading.Event()
    client = _detection_client(legacy_client, fake_stat)
    registrar = registration._REGISTRAR
    results = {}

    try:
        first = _post_in_thread(client, results, "first")
        active_learning.started.wait(5)
        second = _post_in_thread(client, results, "second")
        first.join(5)
        second.join(5)
        answered = set(results)
        timed_out = registrar.timed_out
    finally:
        active_learning.hold.set()
    _settle()

    assert answered == {"first", "second"}
    assert {r.status_code for r in results.values()} == {200}
    assert timed_out == 2
    assert registrar._pending == 0


def test_shutdown_releases_a_request_waiting_in_the_queue(
    legacy_client, fake_stat, active_learning, monkeypatch
):
    monkeypatch.setattr(configuration, "LAMBDA", True)
    monkeypatch.setattr(registration, "WORKERS", 1)
    active_learning.hold = threading.Event()
    client = _detection_client(legacy_client, fake_stat)
    registrar = registration._REGISTRAR
    results = {}

    try:
        first = _post_in_thread(client, results, "first")
        active_learning.started.wait(5)
        second = _post_in_thread(client, results, "second")
        deadline = time.monotonic() + 5
        while registrar._pending < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        registration.stop()
        second.join(5)
        queued_answered = "second" in results
        running_answered = "first" in results
    finally:
        active_learning.hold.set()
    first.join(5)
    deadline = time.monotonic() + 5
    while registrar._pending and time.monotonic() < deadline:
        time.sleep(0.01)

    assert queued_answered
    assert not running_answered
    assert results["second"].status_code == 200
    assert results["first"].status_code == 200
    assert registrar._pending == 0
    assert registrar.timed_out == 0


def _held_registrar():
    release = threading.Event()
    started = threading.Event()

    def factory(api_key, target_dataset, model_id):
        started.set()
        release.wait(10)
        return MagicMock()

    registrar = registration.ActiveLearningRegistrar(
        create_middleware=factory, max_age_s=900
    )
    return registrar, started, release


def test_completion_callback_is_safe_when_the_loop_is_closed():
    registrar, started, release = _held_registrar()
    future = registrar.submit(_job())
    started.wait(5)
    loop = asyncio.new_event_loop()
    try:
        completed = loop.run_until_complete(registration._completion_of(future, 0.05))
    finally:
        loop.close()

    class _Records(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = _Records()
    futures_logger = logging.getLogger("concurrent.futures")
    futures_logger.addHandler(handler)
    try:
        release.set()
        future.result(5)
        deadline = time.monotonic() + 5
        while registrar._pending and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        futures_logger.removeHandler(handler)
        registrar.shutdown()

    assert completed is False
    assert handler.records == []
    assert registrar._pending == 0


def test_cancelled_waiter_leaks_no_pending_job_and_leaves_a_quiet_callback():
    registrar, started, release = _held_registrar()
    future = registrar.submit(_job())
    started.wait(5)

    async def wait_and_cancel():
        task = asyncio.ensure_future(registration._completion_of(future, 30))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(wait_and_cancel())
    pending_while_held = registrar._pending
    release.set()
    future.result(5)
    deadline = time.monotonic() + 5
    while registrar._pending and time.monotonic() < deadline:
        time.sleep(0.01)
    registrar.shutdown()

    assert pending_while_held == 1
    assert registrar._pending == 0
    assert registrar.timed_out == 0


def _platform_fake(state):
    client = MagicMock()
    client.get_roboflow_workspace.return_value = "my-workspace"
    client.get_roboflow_dataset_type.return_value = "object-detection"

    def configuration_of(api_key, workspace_id, dataset_id):
        state.loads.append(api_key)
        return {
            "enabled": True,
            "persist_predictions": True,
            "sampling_strategies": [
                {
                    "name": "everything",
                    "type": "random",
                    "traffic_percentage": 1.1,
                    "tags": [],
                    "limits": [{"type": "hourly", "value": 5}],
                }
            ],
            "batching_strategy": {
                "batches_name_prefix": "al_batch",
                "recreation_interval": "never",
            },
            "tags": [],
        }

    client.get_roboflow_active_learning_configuration.side_effect = configuration_of
    client.register_image_at_roboflow.return_value = {"success": True, "id": "rf-id"}
    client.annotate_image_at_roboflow.return_value = {"success": True}
    client.get_labeling_batches.return_value = {"batches": []}
    return client


@pytest.fixture
def real_wiring(monkeypatch):
    from inference_server.workflows import host

    state = SimpleNamespace(loads=[], shared=InMemoryWorkflowsCache())
    monkeypatch.setattr(configuration, "ACTIVE_LEARNING_ENABLED", True)
    monkeypatch.setattr(host, "WORKFLOWS_CACHE", state.shared)
    monkeypatch.setattr(host, "PLATFORM_CLIENT", _platform_fake(state))
    state.registrar = registration.start()
    yield state
    registration.stop()


def test_configuration_entries_never_reach_the_shared_cache(real_wiring):
    registrar = real_wiring.registrar

    for index in range(300):
        registrar.run(
            _job(
                api_key=f"key-{index}",
                model_id=f"ds-{index % 7}/1",
                target_dataset=f"ds-{index % 7}",
            )
        )

    assert len(real_wiring.loads) == 300
    assert len(registrar._slots) == 256
    assert [
        key
        for key in real_wiring.shared._values
        if key.startswith("active_learning:configurations:")
    ] == []


def _png_job(api_key="k"):
    job = _job(api_key=api_key, model_id="ds/1", target_dataset="ds")
    job.images = [np.zeros((6, 8, 3), dtype=np.uint8)]
    response = MagicMock()
    response.model_dump.side_effect = lambda **kwargs: {
        "image": {"width": 8, "height": 6},
        "predictions": [],
    }
    job.responses = [response]
    return job


def test_usage_counter_goes_to_the_shared_cache_under_the_legacy_key(real_wiring):
    real_wiring.registrar.run(_png_job())

    usage = get_current_strategy_limit_usage(
        cache=real_wiring.shared,
        workspace="my-workspace",
        project="ds",
        strategy_name="everything",
        limit_type=StrategyLimitType.HOURLY,
    )

    assert usage == 1
    assert [
        key for key in real_wiring.shared._values if key.startswith("active_learning:")
    ] == [
        key
        for key in real_wiring.shared._values
        if key.startswith("active_learning:usage:my-workspace:ds:everything:")
    ]


def test_evicted_object_loads_its_configuration_from_the_platform_again(real_wiring):
    registrar = real_wiring.registrar

    registrar.run(_job(api_key="first"))
    registrar.run(_job(api_key="first"))
    loads_while_cached = real_wiring.loads.count("first")
    for index in range(registration.MAX_MIDDLEWARES):
        registrar.run(_job(api_key=f"other-{index}"))
    registrar.run(_job(api_key="first"))

    assert loads_while_cached == 1
    assert real_wiring.loads.count("first") == 2


def test_configuration_keys_start_with_the_prefix_constant():
    from inference_server.active_learning.configuration import (
        ACTIVE_LEARNING_CONFIG_CACHE_KEY_PREFIX,
        construct_cache_key_for_active_learning_config,
    )

    key = construct_cache_key_for_active_learning_config("k", "ds", "ds/1")

    assert key.startswith(ACTIVE_LEARNING_CONFIG_CACHE_KEY_PREFIX)
    assert ACTIVE_LEARNING_CONFIG_CACHE_KEY_PREFIX == "active_learning:configurations:"


def test_cache_adapter_delegates_everything_else_to_the_shared_cache():
    shared = InMemoryWorkflowsCache()
    shared.lock = MagicMock()
    adapter = registration._ConfigurationFreeCache(shared, "active_learning:cfg:")

    adapter.set("active_learning:cfg:a", 1, expire=5)
    adapter.set(key="active_learning:usage:a", value=2, expire=5)

    assert adapter.get("active_learning:cfg:a") is None
    assert adapter.get("active_learning:usage:a") == 2
    assert shared.get("active_learning:cfg:a") is None
    assert shared.get("active_learning:usage:a") == 2
    assert adapter.lock is shared.lock


def _gated_factory(gated_key):
    state = SimpleNamespace(
        created=[], started=threading.Event(), release=threading.Event()
    )

    def factory(api_key, target_dataset, model_id):
        state.created.append(api_key)
        if api_key == gated_key:
            state.started.set()
            state.release.wait(10)
        return MagicMock()

    state.factory = factory
    return state


def test_entry_being_created_is_not_evicted_and_not_created_twice():
    gate = _gated_factory("A")
    registrar = registration.ActiveLearningRegistrar(
        create_middleware=gate.factory, max_age_s=900
    )
    first = threading.Thread(target=registrar.run, args=(_job(api_key="A"),))
    second = threading.Thread(target=registrar.run, args=(_job(api_key="A"),))

    try:
        first.start()
        gate.started.wait(5)
        for index in range(registration.MAX_MIDDLEWARES):
            registrar.run(_job(api_key=f"other-{index}"))
        second.start()
        time.sleep(0.1)
    finally:
        gate.release.set()
    first.join(5)
    second.join(5)
    registrar.shutdown()

    assert gate.created.count("A") == 1
    assert len(registrar._slots) == registration.MAX_MIDDLEWARES


def test_eviction_takes_the_least_recently_used_idle_entry():
    gate = _gated_factory("A")
    registrar = registration.ActiveLearningRegistrar(
        create_middleware=gate.factory, max_age_s=900
    )
    first = threading.Thread(target=registrar.run, args=(_job(api_key="A"),))

    try:
        first.start()
        gate.started.wait(5)
        for index in range(registration.MAX_MIDDLEWARES):
            registrar.run(_job(api_key=f"other-{index}"))
        keys = {key[0] for key in registrar._slots}
    finally:
        gate.release.set()
    first.join(5)
    registrar.shutdown()

    assert len(keys) == registration.MAX_MIDDLEWARES
    assert registrar._key_of(_job(api_key="A"))[0] in keys
    assert registrar._key_of(_job(api_key="other-0"))[0] not in keys
    assert registrar._key_of(_job(api_key="other-1"))[0] in keys


def test_new_entry_is_not_cached_when_every_entry_is_being_created(monkeypatch):
    monkeypatch.setattr(registration, "MAX_MIDDLEWARES", 1)
    gate = _gated_factory("A")
    registrar = registration.ActiveLearningRegistrar(
        create_middleware=gate.factory, max_age_s=900
    )
    first = threading.Thread(target=registrar.run, args=(_job(api_key="A"),))

    try:
        first.start()
        gate.started.wait(5)
        registrar.run(_job(api_key="B"))
        cached = list(registrar._slots)
    finally:
        gate.release.set()
    first.join(5)
    registrar.shutdown()

    assert gate.created == ["A", "B"]
    assert cached == [registrar._key_of(_job(api_key="A"))]


def test_inactive_hits_keep_an_entry_recent():
    inactive = MagicMock()
    inactive.active = False
    created = []

    def factory(api_key, target_dataset, model_id):
        created.append(api_key)
        return inactive if api_key == "A" else MagicMock()

    registrar = registration.ActiveLearningRegistrar(
        create_middleware=factory, max_age_s=900
    )
    registrar.run(_job(api_key="A"))
    for index in range(registration.MAX_MIDDLEWARES - 1):
        registrar.run(_job(api_key=f"other-{index}"))
    for _ in range(3):
        assert registrar.is_known_inactive(_job(api_key="A")) is True
    registrar.run(_job(api_key="new"))
    registrar.shutdown()

    keys = {key[0] for key in registrar._slots}
    assert registrar._key_of(_job(api_key="A"))[0] in keys
    assert registrar._key_of(_job(api_key="other-0"))[0] not in keys
    assert registrar._key_of(_job(api_key="new"))[0] in keys
    assert len(keys) == registration.MAX_MIDDLEWARES


def test_startup_without_the_workflows_package_serves_inference():
    code = """
import base64, io, json, sys
sys.modules["roboflow_workflows"] = None
sys.path.insert(0, ".")
from types import SimpleNamespace
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import inference_model_manager.watchdogs as watchdogs
import inference_server.gateway_resolver as resolver
from inference_server import configuration
from inference_server.framework import model_stat
from tests.unit_tests.legacy.conftest import FakeGateway

watchdogs.start_enabled_watchdogs = lambda: []
configuration.ACTIVE_LEARNING_ENABLED = True
model_stat.get_one_page_of_model_metadata = lambda model_id, api_key=None, **_: (
    SimpleNamespace(task_type="object-detection")
)
detection = SimpleNamespace(
    xyxy=np.array([[1, 1, 3, 5]], dtype=float),
    confidence=np.array([0.9]),
    class_id=np.array([0]),
)
gateway = FakeGateway(
    predictions={("ds/1", "infer"): detection},
    model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
)
resolver.resolve_gateway = lambda: gateway
import inference_server.app as app_mod
from inference_server.legacy import active_learning_registration as registration

buffer = io.BytesIO()
Image.new("RGB", (8, 6)).save(buffer, format="PNG")
body = {
    "model_id": "ds/1",
    "api_key": "k",
    "image": {"type": "base64", "value": base64.b64encode(buffer.getvalue()).decode()},
}
with TestClient(app_mod.app, raise_server_exceptions=False) as client:
    response = client.post("/infer/object_detection", json=body)
    registrar = registration._REGISTRAR
print(json.dumps({
    "status": response.status_code,
    "registrar_absent": registrar is None,
    "loaded": sorted(
        name for name, module in sys.modules.items()
        if name.startswith("roboflow_workflows") and module is not None
    ),
}))
"""
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {"status": 200, "registrar_absent": True, "loaded": []}
