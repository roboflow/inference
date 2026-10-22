import asyncio
import base64
import hashlib
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import requests
import requests_mock
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse
from PIL import Image
from roboflow_workflows.prototypes.platform_errors import (
    RoboflowAPIUnsuccessfulRequestError,
)
from roboflow_workflows.utils.in_memory_cache import InMemoryWorkflowsCache

import inference_server.app as app_mod
import inference_server.telemetry as telemetry_module
from inference_server import configuration, pingback, platform_http
from inference_server.framework.entities import (
    ModelHandlerDescription,
    ModelInterfaceDescription,
)
from inference_server.framework.registry import _HANDLERS
from inference_server.hosted import serverless_auth
from inference_server.hosted.serverless_auth import ServerlessAuthMiddleware
from inference_server.legacy import errors as legacy_errors
from inference_server.legacy.bridge import LegacyModelBridge, Route
from inference_server.legacy.errors import LegacyHTTPError, with_legacy_errors
from inference_server.legacy.telemetry_recording import request_telemetry_scope
from inference_server.middlewares.model_load import (
    MODEL_LOAD_EVENTS,
    record_model_load,
)
from inference_server.workflows import errors as workflow_errors
from inference_server.workflows import host
from inference_server.workflows.errors import with_workflow_errors
from inference_server.workflows.models_provider import GatewayModelsProvider
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.test_telemetry import (  # noqa: F401
    _probe_app,
    _span,
    fake_otel,
    no_otel,
)
from tests.unit_tests.workflows.test_models_provider import FakeSyncBridge

API_URL = "https://api.example.com"
LOAD_TIME_S = 1.5
LOADED = "inference.models.loaded"
LOADS = "inference.model.loads"
UNLOADS = "inference.model.unloads"
LOAD_DURATION = "inference.model.load.duration"
INFER_COUNT = "inference.model.infer.count"
INFER_DURATION = "inference.model.infer.duration"
API_DURATION = "inference.roboflow_api.duration"
ERRORS = "inference.errors"


class _Instrument:
    def __init__(self, name: str, points: list) -> None:
        self._name = name
        self._points = points

    def add(self, value, attributes=None) -> None:
        self._points.append((self._name, value, attributes))

    def record(self, value, attributes=None) -> None:
        self._points.append((self._name, value, attributes))


class _Recorded:
    def __init__(self, points: list, span) -> None:
        self.points = points
        self.span = span

    def named(self, name: str) -> list:
        return [point for point in self.points if point[0] == name]

    def attributes(self, name: str) -> list:
        return [point[2] for point in self.named(name)]

    def values(self, name: str) -> list:
        return [point[1] for point in self.named(name)]


@pytest.fixture
def otel(fake_otel):
    telemetry, mocks, patch_ = fake_otel
    points = []
    meter = mocks["meter_provider_cls"].return_value.get_meter.return_value
    for factory in ("create_counter", "create_histogram", "create_up_down_counter"):
        getattr(meter, factory).side_effect = lambda name, **kwargs: _Instrument(
            name, points
        )
    telemetry.setup_telemetry(_probe_app())
    span = _span()
    mocks["get_current_span"].return_value = span

    return _Recorded(points, span)


class _ColdStartGateway(FakeGateway):
    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        cold = model_id not in self.loaded
        result = await super().ensure_loaded(model_id, instance, api_key, device)
        if cold:
            record_model_load(model_id, cold_start=True, load_time_s=LOAD_TIME_S)
        return result


def _jpeg_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    return encoded


def _image() -> dict:
    return {"type": "base64", "value": _jpeg_b64()}


def _detections():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


def _detection_gateway(model_id: str = "ds/1", gateway_cls=_ColdStartGateway):
    return gateway_cls(
        predictions={(model_id, "infer"): _detections()},
        model_info={model_id: {"class_names": ["cat"], "actions": {"infer": {}}}},
    )


def _infer_body(model_id: str = "ds/1", image=None) -> dict:
    return {
        "model_id": model_id,
        "api_key": "k",
        "image": image if image is not None else _image(),
    }


def test_legacy_inference_records_duration_and_the_first_load(
    legacy_client, fake_stat, otel
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway())

    first = client.post("/infer/object_detection", json=_infer_body())

    assert first.status_code == 200, first.text
    assert otel.named(LOADED) == [(LOADED, 1, None)]
    assert otel.named(LOADS) == [(LOADS, 1, {"model.id": "ds/1"})]
    assert otel.named(LOAD_DURATION) == [
        (LOAD_DURATION, LOAD_TIME_S, {"model.id": "ds/1"})
    ]
    assert otel.named(INFER_COUNT) == [(INFER_COUNT, 1, {"model.id": "ds/1"})]
    assert otel.attributes(INFER_DURATION) == [{"model.id": "ds/1"}]
    assert 0.0 <= otel.values(INFER_DURATION)[0] < 60.0
    assert otel.named(ERRORS) == []
    assert otel.named(UNLOADS) == []
    otel.span.record_exception.assert_not_called()

    second = client.post("/infer/object_detection", json=_infer_body())

    assert second.status_code == 200, second.text
    assert len(otel.named(LOADS)) == 1
    assert len(otel.named(LOAD_DURATION)) == 1
    assert len(otel.named(LOADED)) == 1
    assert otel.attributes(INFER_COUNT) == [{"model.id": "ds/1"}] * 2
    assert len(otel.named(INFER_DURATION)) == 2


def test_legacy_batch_inference_records_one_inference(legacy_client, fake_stat, otel):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway())

    response = client.post(
        "/infer/object_detection",
        json=_infer_body(image=[_image(), _image(), _image()]),
    )

    assert response.status_code == 200, response.text
    assert len(response.json()) == 3
    assert otel.named(INFER_COUNT) == [(INFER_COUNT, 1, {"model.id": "ds/1"})]
    assert len(otel.named(INFER_DURATION)) == 1
    assert len(otel.named(LOADS)) == 1


def test_legacy_inference_records_the_model_id_as_requested(
    legacy_client, fake_stat, otel
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway("coco/3"))

    response = client.post("/infer/object_detection", json=_infer_body("yolov8n-640"))

    assert response.status_code == 200, response.text
    assert otel.attributes(INFER_COUNT) == [{"model.id": "yolov8n-640"}]
    assert otel.attributes(INFER_DURATION) == [{"model.id": "yolov8n-640"}]
    assert otel.attributes(LOADS) == [{"model.id": "yolov8n-640"}]
    assert otel.attributes(LOAD_DURATION) == [{"model.id": "yolov8n-640"}]


def test_legacy_embedding_route_records_one_inference_for_all_its_calls(
    legacy_client, fake_stat, otel
):
    gateway = _ColdStartGateway(
        predictions={
            ("clip/ViT-B-16", "embed_text"): lambda image, params: np.array(
                [[1.0, 0.0]] * len(params["texts"])
            )
        },
        model_info={"clip/ViT-B-16": {"actions": {"embed_text": {}, "compare": {}}}},
    )

    response = legacy_client(gateway).post(
        "/clip/compare",
        json={
            "subject": "a",
            "subject_type": "text",
            "prompt": {"x": "b", "y": "c"},
            "prompt_type": "text",
        },
    )

    assert response.status_code == 200, response.text
    assert len([call for call in gateway.calls if call[0] == "infer"]) > 1
    assert otel.named(INFER_COUNT) == [(INFER_COUNT, 1, {"model.id": "clip/ViT-B-16"})]
    assert otel.attributes(INFER_DURATION) == [{"model.id": "clip/ViT-B-16"}]
    assert otel.attributes(LOADS) == [{"model.id": "clip/ViT-B-16"}]


def test_workflow_embedding_records_one_inference_for_all_its_calls(otel):
    bridge = FakeSyncBridge()
    bridge.routes["clip/ViT-B-16"] = Route(
        model_id="clip/ViT-B-16",
        registry_id="clip/ViT-B-16",
        task_type="embedding",
        action="embed_images",
        actions={"embed_images", "embed_text", "compare"},
    )
    bridge.predictions[("clip/ViT-B-16", "embed_text")] = np.array([[1.0, 0.0]])

    GatewayModelsProvider(bridge, api_key=None).run_clip_comparison(
        subject="a",
        subject_type="text",
        prompt=["b"],
        prompt_type="text",
        version_id="ViT-B-16",
    )

    assert bridge.records == [False, False]
    assert otel.named(INFER_COUNT) == [(INFER_COUNT, 1, {"model.id": "clip/ViT-B-16"})]
    assert otel.attributes(INFER_DURATION) == [{"model.id": "clip/ViT-B-16"}]


def test_workflow_step_inference_records_duration_and_the_first_load(
    legacy_client, fake_stat, otel
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    specification = {
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

    response = legacy_client(_detection_gateway()).post(
        "/workflows/run",
        json={
            "specification": specification,
            "inputs": {"image": _image()},
            "api_key": "k",
        },
    )

    assert response.status_code == 200, response.text
    assert otel.named(INFER_COUNT) == [(INFER_COUNT, 1, {"model.id": "ds/1"})]
    assert otel.attributes(INFER_DURATION) == [{"model.id": "ds/1"}]
    assert otel.named(LOADS) == [(LOADS, 1, {"model.id": "ds/1"})]
    assert otel.values(LOAD_DURATION) == [LOAD_TIME_S]
    assert otel.named(ERRORS) == []


def test_legacy_route_failure_records_one_error(legacy_client, fake_stat, otel):
    client = legacy_client(_detection_gateway())

    response = client.post("/infer/object_detection", json=_infer_body("missing/1"))

    assert response.status_code == 404, response.text
    assert otel.named(ERRORS) == [(ERRORS, 1, {"error.type": "ModelNotFoundError"})]
    assert otel.span.record_exception.call_count == 1
    recorded = otel.span.record_exception.call_args.args[0]
    assert isinstance(recorded, LookupError)
    otel.span.set_status.assert_called_once_with("ERROR", str(recorded))
    assert otel.named(INFER_COUNT) == []
    assert otel.named(INFER_DURATION) == []
    assert otel.named(LOADS) == []


def test_legacy_inference_failure_records_the_error_and_no_inference(
    legacy_client, fake_stat, otel
):
    fake_stat["ds/1"] = ("object-detection", "infer")

    def _fail(image, params):
        raise RuntimeError("model failed")

    gateway = _ColdStartGateway(
        predictions={("ds/1", "infer"): _fail},
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
    )

    response = legacy_client(gateway).post(
        "/infer/object_detection", json=_infer_body()
    )

    assert response.status_code == 500, response.text
    assert otel.named(ERRORS) == [(ERRORS, 1, {"error.type": "RuntimeError"})]
    assert otel.span.record_exception.call_count == 1
    assert otel.named(INFER_COUNT) == []
    assert otel.named(INFER_DURATION) == []
    assert len(otel.named(LOADS)) == 1


def test_workflow_route_failure_records_one_error(legacy_client, otel):
    response = legacy_client(FakeGateway()).post(
        "/workflows/run",
        json={
            "specification": {
                "version": "1.0",
                "inputs": [],
                "steps": [{"type": "nope"}],
                "outputs": [],
            },
            "inputs": {},
        },
    )

    assert response.status_code == 400, response.text
    assert otel.named(ERRORS) == [(ERRORS, 1, {"error.type": "WorkflowSyntaxError"})]
    assert otel.span.record_exception.call_count == 1
    recorded = otel.span.record_exception.call_args.args[0]
    assert type(recorded).__name__ == "WorkflowSyntaxError"
    otel.span.set_status.assert_called_once_with("ERROR", str(recorded))


@pytest.mark.parametrize(
    "decorator,module",
    [(with_legacy_errors, legacy_errors), (with_workflow_errors, workflow_errors)],
)
def test_route_wrappers_record_the_error_before_mapping_it(
    otel, monkeypatch, decorator, module
):
    order = []
    monkeypatch.setattr(
        telemetry_module, "record_error", lambda error: order.append("span")
    )
    monkeypatch.setattr(
        telemetry_module,
        "record_error_metric",
        lambda error_type: order.append(f"metric:{error_type}"),
    )

    def _mapped(error):
        order.append("mapped")
        return JSONResponse(status_code=500, content={})

    monkeypatch.setattr(module, "legacy_error_response", _mapped)

    @decorator
    async def route():
        raise KeyError("boom")

    response = asyncio.run(route())

    assert response.status_code == 500
    assert order == ["span", "metric:KeyError", "mapped"]


@pytest.mark.parametrize("decorator", [with_legacy_errors, with_workflow_errors])
def test_route_wrappers_record_an_http_exception_and_reraise_it(otel, decorator):
    error = HTTPException(status_code=418, detail="teapot")

    @decorator
    async def route():
        raise error

    with pytest.raises(HTTPException):
        asyncio.run(route())

    assert otel.named(ERRORS) == [(ERRORS, 1, {"error.type": "HTTPException"})]
    otel.span.record_exception.assert_called_once_with(error)


@pytest.mark.parametrize("decorator", [with_legacy_errors, with_workflow_errors])
def test_route_wrappers_record_nothing_for_an_answered_request(otel, decorator):
    @decorator
    async def route():
        return JSONResponse(status_code=400, content={"message": "handled"})

    response = asyncio.run(route())

    assert response.status_code == 400
    assert otel.points == []
    otel.span.record_exception.assert_not_called()


def test_model_add_records_the_load_once(legacy_client, fake_stat, otel):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway())

    first = client.post("/model/add", json={"model_id": "ds/1", "api_key": "k"})
    second = client.post("/model/add", json={"model_id": "ds/1", "api_key": "k"})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert otel.named(LOADED) == [(LOADED, 1, None)]
    assert otel.named(LOADS) == [(LOADS, 1, {"model.id": "ds/1"})]
    assert otel.named(LOAD_DURATION) == [
        (LOAD_DURATION, LOAD_TIME_S, {"model.id": "ds/1"})
    ]
    assert otel.named(INFER_COUNT) == []


def test_model_remove_and_clear_record_unloads(legacy_client, fake_stat, otel):
    fake_stat["ds/1"] = ("object-detection", "infer")
    fake_stat["ds/2"] = ("object-detection", "infer")
    fake_stat["ds/3"] = ("object-detection", "infer")
    client = legacy_client(_ColdStartGateway())
    for model_id in ("ds/1", "ds/2", "ds/3"):
        added = client.post("/model/add", json={"model_id": model_id, "api_key": "k"})
        assert added.status_code == 200, added.text
    assert otel.values(LOADED) == [1, 1, 1]

    removed = client.post("/model/remove", json={"model_id": "ds/1"})

    assert removed.status_code == 200, removed.text
    assert otel.named(UNLOADS) == [(UNLOADS, 1, {"model.id": "ds/1"})]
    assert otel.values(LOADED) == [1, 1, 1, -1]

    absent = client.post("/model/remove", json={"model_id": "ds/1"})

    assert absent.status_code == 200, absent.text
    assert len(otel.named(UNLOADS)) == 1

    cleared = client.post("/model/clear")

    assert cleared.status_code == 200, cleared.text
    assert sorted(
        attributes["model.id"] for attributes in otel.attributes(UNLOADS)
    ) == ["ds/1", "ds/2", "ds/3"]
    assert otel.values(LOADED) == [1, 1, 1, -1, -1, -1]
    assert otel.named(ERRORS) == []


def test_clear_cache_route_records_unloads(legacy_client, fake_stat, otel):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_ColdStartGateway())
    client.post("/model/add", json={"model_id": "ds/1", "api_key": "k"})

    response = client.get("/clear_cache")

    assert response.status_code == 200, response.text
    assert otel.named(UNLOADS) == [(UNLOADS, 1, {"model.id": "ds/1"})]


@pytest.mark.asyncio
async def test_two_calls_of_one_request_record_one_load_for_one_cold_start(otel):
    class _SharedLoadGateway(FakeGateway):
        def __init__(self) -> None:
            super().__init__()
            self.started = False

        async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
            starts_the_load = not self.started
            self.started = True
            await asyncio.sleep(0)
            if starts_the_load:
                record_model_load(model_id, cold_start=True, load_time_s=LOAD_TIME_S)
            else:
                await asyncio.sleep(0)
            return ("model_ready",)

    bridge = LegacyModelBridge(_SharedLoadGateway())
    route = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="object-detection",
        action="infer",
    )
    token = MODEL_LOAD_EVENTS.set([])
    try:
        with request_telemetry_scope():
            await asyncio.gather(
                bridge.ensure_loaded(route, "k"), bridge.ensure_loaded(route, "k")
            )
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert otel.named(LOADS) == [(LOADS, 1, {"model.id": "ds/1"})]
    assert otel.named(LOADED) == [(LOADED, 1, None)]


@pytest.mark.asyncio
async def test_a_cold_start_of_another_model_is_not_recorded_by_this_call(otel):
    class _OtherModelGateway(FakeGateway):
        async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
            record_model_load("other/1", cold_start=True, load_time_s=LOAD_TIME_S)
            return ("model_ready",)

    bridge = LegacyModelBridge(_OtherModelGateway())
    route = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="object-detection",
        action="infer",
    )
    token = MODEL_LOAD_EVENTS.set([])
    try:
        with request_telemetry_scope():
            await bridge.ensure_loaded(route, "k")
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert otel.named(LOADS) == []


@pytest.fixture
def platform_api(monkeypatch):
    monkeypatch.setattr(host.configuration, "API_BASE_URL", API_URL + "/")
    monkeypatch.setattr(host.configuration, "LEGACY_OFFLINE_MODE", False)
    monkeypatch.setattr(host, "WORKFLOWS_CACHE", InMemoryWorkflowsCache())
    host.clear_workspace_cache()
    yield
    host.clear_workspace_cache()


PLATFORM_CALLS = [
    (
        "add_custom_metadata",
        lambda client: client.add_custom_metadata(
            api_key="k",
            workspace_id="ws",
            inference_ids=["i"],
            field_name="f",
            field_value="v",
        ),
    ),
    (
        "register_image_at_roboflow",
        lambda client: client.register_image_at_roboflow(
            api_key="k",
            dataset_id="ds",
            local_image_id="img",
            image_bytes=b"x",
            batch_name="b",
        ),
    ),
    (
        "annotate_image_at_roboflow",
        lambda client: client.annotate_image_at_roboflow(
            api_key="k",
            dataset_id="ds",
            local_image_id="img",
            roboflow_image_id="rf",
            annotation_content="{}",
            annotation_file_type="json",
        ),
    ),
    (
        "update_image_metadata_at_roboflow",
        lambda client: client.update_image_metadata_at_roboflow(
            api_key="k", workspace_id="ws", image_id="img", metadata={"a": 1}
        ),
    ),
    (
        "batch_update_image_metadata_at_roboflow",
        lambda client: client.batch_update_image_metadata_at_roboflow(
            api_key="k", workspace_id="ws", updates=[{"image_id": "img"}]
        ),
    ),
    (
        "_make_request",
        lambda client: client.search_project_images_at_roboflow(
            api_key="k", workspace="ws", project="p", image_base64="aW1n", limit=1
        ),
    ),
    (
        "send_inference_results_to_model_monitoring",
        lambda client: client.send_inference_results_to_model_monitoring(
            api_key="k", workspace_id="ws", inference_data={"a": 1}
        ),
    ),
    ("get_roboflow_workspace", lambda client: client.get_roboflow_workspace("k")),
    (
        "get_roboflow_dataset_type",
        lambda client: client.get_roboflow_dataset_type(
            api_key="k", workspace_id="ws", dataset_id="p"
        ),
    ),
    (
        "get_roboflow_active_learning_configuration",
        lambda client: client.get_roboflow_active_learning_configuration(
            api_key="k", workspace_id="ws", dataset_id="p"
        ),
    ),
    (
        "get_roboflow_labeling_batches",
        lambda client: client.get_roboflow_labeling_batches(
            api_key="k", workspace_id="ws", dataset_id="p"
        ),
    ),
    (
        "get_roboflow_labeling_jobs",
        lambda client: client.get_roboflow_labeling_jobs(
            api_key="k", workspace_id="ws", dataset_id="p"
        ),
    ),
    (
        "_make_request",
        lambda client: client.post(endpoint="ws/vision-events", api_key="k"),
    ),
]


@pytest.mark.parametrize("function_name,call", PLATFORM_CALLS)
def test_platform_call_records_its_duration_under_the_legacy_function_name(
    otel, platform_api, function_name, call
):
    with requests_mock.Mocker() as mocker:
        mocker.register_uri(
            requests_mock.ANY,
            requests_mock.ANY,
            json={"success": True, "workspace": "ws"},
        )

        call(host.PLATFORM_CLIENT)

        assert mocker.call_count == 1
    assert otel.attributes(API_DURATION) == [{"roboflow_api.function": function_name}]
    assert 0.0 <= otel.values(API_DURATION)[0] < 60.0
    assert [point for point in otel.points if point[0] != API_DURATION] == []


@pytest.mark.parametrize("function_name,call", PLATFORM_CALLS)
def test_failed_platform_call_records_its_duration_once(
    otel, platform_api, function_name, call
):
    with requests_mock.Mocker() as mocker:
        mocker.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=500)

        with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
            call(host.PLATFORM_CLIENT)

    assert otel.attributes(API_DURATION) == [{"roboflow_api.function": function_name}]


def test_cached_workspace_lookup_records_no_second_api_call(otel, platform_api):
    with requests_mock.Mocker() as mocker:
        mocker.get(requests_mock.ANY, json={"workspace": "ws"})

        first = host.PLATFORM_CLIENT.get_roboflow_workspace("k")
        second = host.PLATFORM_CLIENT.get_roboflow_workspace("k")

        assert mocker.call_count == 1
    assert (first, second) == ("ws", "ws")
    assert otel.attributes(API_DURATION) == [
        {"roboflow_api.function": "get_roboflow_workspace"}
    ]


def _platform_response(status_code: int, content: bytes) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response._content = content

    return response


def test_workflow_definition_fetch_records_the_api_call_on_every_lookup(
    legacy_client, monkeypatch, otel, platform_api
):
    definition = (
        b'{"workflow": {"id": "wf", "config": "{\\"specification\\": '
        b'{\\"version\\": \\"1.0\\", \\"inputs\\": [], \\"steps\\": [], '
        b'\\"outputs\\": []}}"}}'
    )
    fetches = []

    def _fetch(*args, **kwargs):
        fetches.append(args)
        return _platform_response(200, definition)

    monkeypatch.setattr(host, "_platform_request", _fetch)
    client = legacy_client(FakeGateway())

    first = client.post("/ws/workflows/wf", json={"api_key": "k", "inputs": {}})
    second = client.post("/ws/workflows/wf", json={"api_key": "k", "inputs": {}})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert len(fetches) == 1
    assert (
        otel.attributes(API_DURATION)
        == [{"roboflow_api.function": "get_workflow_specification"}] * 2
    )
    assert otel.named(ERRORS) == []


def test_failed_workflow_definition_fetch_records_the_api_call_and_the_error(
    legacy_client, monkeypatch, otel, platform_api
):
    monkeypatch.setattr(
        host,
        "_platform_request",
        lambda *args, **kwargs: _platform_response(404, b"{}"),
    )

    response = legacy_client(FakeGateway()).post(
        "/ws/workflows/wf", json={"api_key": "k", "use_cache": False, "inputs": {}}
    )

    assert response.status_code == 404, response.text
    assert otel.attributes(API_DURATION) == [
        {"roboflow_api.function": "get_workflow_specification"}
    ]
    assert otel.named(ERRORS) == [(ERRORS, 1, {"error.type": "LegacyHTTPError"})]
    assert otel.span.record_exception.call_count == 1


def test_pingback_post_records_no_api_call_like_legacy(otel, monkeypatch):
    posts = []

    def _post(url, **kwargs):
        posts.append(url)
        return MagicMock(status_code=200)

    monkeypatch.setattr(pingback.requests, "post", _post)
    monkeypatch.setattr(configuration, "LEGACY_OFFLINE_MODE", False)

    pingback.PingbackSender().post()

    assert len(posts) == 1
    assert otel.points == []


def _v2_handler(model):
    async def handler(action, input_data, proxy, hooks):
        return model()

    description = ModelHandlerDescription(
        input_parser=AsyncMock(return_value={"images": [b"x"], "params": {}}),
        handler=handler,
        output_serializer=lambda prediction, common: Response(status_code=200),
        interface_provider=lambda: ModelInterfaceDescription(
            task="t", params={}, output_schema={}
        ),
    )

    return description


def _v2_infer(client, model):
    keys_before = set(_HANDLERS)
    _HANDLERS[("telemetry-task", "infer")] = _v2_handler(model)
    try:
        with patch(
            "inference_server.framework.dispatch.stat_model_while_checking_auth",
            new=AsyncMock(return_value=("telemetry-task", "infer")),
        ):
            response = client.post(
                "/v2/models/infer?model_id=ds/1",
                headers={"Authorization": "Bearer k"},
                content=b"x",
            )
    finally:
        for key in set(_HANDLERS) - keys_before:
            del _HANDLERS[key]

    return response


@pytest.fixture
def v2_client(legacy_client, monkeypatch):
    monkeypatch.setattr(
        app_mod, "validate_api_key", AsyncMock(return_value=(True, None))
    )
    monkeypatch.setattr(app_mod._cfg, "ENABLE_CONTROL_PLANE_ROUTES", True)

    return legacy_client


def test_v2_inference_records_nothing(v2_client, otel):
    gateway = _ColdStartGateway()
    client = v2_client(gateway)

    response = _v2_infer(client, lambda: "prediction")

    assert response.status_code == 200, response.text
    assert ("ensure_loaded", "ds/1", "k") in gateway.calls
    assert otel.points == []
    otel.span.record_exception.assert_not_called()


def test_v2_inference_failure_records_nothing(v2_client, otel):
    def _fail():
        raise RuntimeError("model failed")

    client = v2_client(_ColdStartGateway())

    response = _v2_infer(client, _fail)

    assert response.status_code == 500, response.text
    assert otel.points == []
    otel.span.record_exception.assert_not_called()


def test_v2_load_and_unload_record_nothing(v2_client, otel):
    client = v2_client(_ColdStartGateway())
    headers = {"Authorization": "Bearer k"}

    loaded = client.post("/v2/models/load?model_id=ds/1", headers=headers)
    unloaded = client.post("/v2/models/unload?model_id=ds/1", headers=headers)

    assert loaded.status_code == 200, loaded.text
    assert unloaded.status_code == 200, unloaded.text
    assert otel.points == []


def test_legacy_routes_work_with_telemetry_off(legacy_client, fake_stat, no_otel):
    telemetry, _ = no_otel
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway())

    served = client.post("/infer/object_detection", json=_infer_body())
    failed = client.post("/infer/object_detection", json=_infer_body("missing/1"))
    added = client.post("/model/add", json={"model_id": "ds/1", "api_key": "k"})
    removed = client.post("/model/remove", json={"model_id": "ds/1"})
    cleared = client.post("/model/clear")
    workflow_failed = client.post(
        "/workflows/run",
        json={
            "specification": {
                "version": "1.0",
                "inputs": [],
                "steps": [{"type": "nope"}],
                "outputs": [],
            },
            "inputs": {},
        },
    )

    assert telemetry._metrics is None
    assert served.status_code == 200, served.text
    assert failed.status_code == 404, failed.text
    assert added.status_code == 200, added.text
    assert removed.status_code == 200, removed.text
    assert cleared.status_code == 200, cleared.text
    assert workflow_failed.status_code == 400, workflow_failed.text


def test_metrics_off_with_tracing_on_records_no_point(
    legacy_client, fake_stat, fake_otel
):
    telemetry, mocks, _ = fake_otel
    span = _span()
    mocks["get_current_span"].return_value = span
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway())

    served = client.post("/infer/object_detection", json=_infer_body())
    failed = client.post("/infer/object_detection", json=_infer_body("missing/1"))

    assert telemetry._metrics is None
    assert served.status_code == 200, served.text
    assert failed.status_code == 404, failed.text
    assert span.record_exception.call_count == 1


BROKEN_FUNCTIONS = (
    "record_error",
    "record_error_metric",
    "record_inference",
    "record_model_loaded",
    "record_model_unloaded",
    "record_api_call",
)


@pytest.fixture
def broken_telemetry(monkeypatch):
    calls = []

    def _broken(name):
        def _raise(*args, **kwargs):
            calls.append(name)
            raise RuntimeError(f"{name} failed")

        return _raise

    for name in BROKEN_FUNCTIONS:
        monkeypatch.setattr(telemetry_module, name, _broken(name))

    return calls


@pytest.mark.parametrize("decorator", [with_legacy_errors, with_workflow_errors])
def test_a_failing_recorder_keeps_the_original_http_exception(
    broken_telemetry, decorator
):
    @decorator
    async def route():
        raise HTTPException(status_code=418, detail="teapot")

    with pytest.raises(HTTPException) as raised:
        asyncio.run(route())

    assert raised.value.status_code == 418
    assert "record_error" in broken_telemetry


@pytest.mark.parametrize("decorator", [with_legacy_errors, with_workflow_errors])
def test_a_failing_recorder_keeps_the_mapped_error_response(
    broken_telemetry, decorator
):
    @decorator
    async def route():
        raise KeyError("boom")

    response = asyncio.run(route())

    assert response.status_code == 500
    assert broken_telemetry == ["record_error", "record_error_metric"]


def test_legacy_route_failure_keeps_its_status_when_telemetry_fails(
    legacy_client, fake_stat, broken_telemetry
):
    response = legacy_client(_detection_gateway()).post(
        "/infer/object_detection", json=_infer_body("missing/1")
    )

    assert response.status_code == 404, response.text
    assert "record_error" in broken_telemetry


def test_workflow_route_failure_keeps_its_status_when_telemetry_fails(
    legacy_client, broken_telemetry
):
    response = legacy_client(FakeGateway()).post(
        "/workflows/run",
        json={
            "specification": {
                "version": "1.0",
                "inputs": [],
                "steps": [{"type": "nope"}],
                "outputs": [],
            },
            "inputs": {},
        },
    )

    assert response.status_code == 400, response.text
    assert "record_error" in broken_telemetry


def test_successful_inference_answers_200_when_telemetry_fails(
    legacy_client, fake_stat, broken_telemetry
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(_detection_gateway())

    response = client.post("/infer/object_detection", json=_infer_body())

    assert response.status_code == 200, response.text
    assert "record_inference" in broken_telemetry
    assert "record_model_loaded" in broken_telemetry


def test_embedding_route_answers_200_when_telemetry_fails(
    legacy_client, fake_stat, broken_telemetry
):
    gateway = _ColdStartGateway(
        predictions={
            ("clip/ViT-B-16", "embed_text"): lambda image, params: np.array(
                [[1.0, 0.0]] * len(params["texts"])
            )
        },
        model_info={"clip/ViT-B-16": {"actions": {"embed_text": {}, "compare": {}}}},
    )

    response = legacy_client(gateway).post(
        "/clip/compare",
        json={
            "subject": "a",
            "subject_type": "text",
            "prompt": {"x": "b", "y": "c"},
            "prompt_type": "text",
        },
    )

    assert response.status_code == 200, response.text
    assert "record_inference" in broken_telemetry


def test_workflow_embedding_returns_when_telemetry_fails(broken_telemetry):
    bridge = FakeSyncBridge()
    bridge.routes["clip/ViT-B-16"] = Route(
        model_id="clip/ViT-B-16",
        registry_id="clip/ViT-B-16",
        task_type="embedding",
        action="embed_images",
        actions={"embed_images", "embed_text", "compare"},
    )
    bridge.predictions[("clip/ViT-B-16", "embed_text")] = np.array([[1.0, 0.0]])

    result = GatewayModelsProvider(bridge, api_key=None).run_clip_comparison(
        subject="a",
        subject_type="text",
        prompt=["b"],
        prompt_type="text",
        version_id="ViT-B-16",
    )

    assert result is not None
    assert "record_inference" in broken_telemetry


def test_model_remove_and_clear_answer_as_before_when_telemetry_fails(
    legacy_client, fake_stat, broken_telemetry
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    fake_stat["ds/2"] = ("object-detection", "infer")
    client = legacy_client(_ColdStartGateway())
    for model_id in ("ds/1", "ds/2"):
        added = client.post("/model/add", json={"model_id": model_id, "api_key": "k"})
        assert added.status_code == 200, added.text

    removed = client.post("/model/remove", json={"model_id": "ds/1"})
    cleared = client.post("/model/clear")

    assert removed.status_code == 200, removed.text
    assert cleared.status_code == 200, cleared.text
    assert broken_telemetry.count("record_model_unloaded") == 2


def test_successful_platform_call_returns_its_value_when_telemetry_fails(
    platform_api, broken_telemetry
):
    with requests_mock.Mocker() as mocker:
        mocker.get(requests_mock.ANY, json={"workspace": "ws"})

        workspace = host.PLATFORM_CLIENT.get_roboflow_workspace("k")

    assert workspace == "ws"
    assert "record_api_call" in broken_telemetry


def test_failed_platform_call_raises_its_own_error_when_telemetry_fails(
    platform_api, broken_telemetry
):
    with requests_mock.Mocker() as mocker:
        mocker.register_uri(requests_mock.ANY, requests_mock.ANY, status_code=500)

        with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
            host.PLATFORM_CLIENT.get_roboflow_workspace("k")

    assert "record_api_call" in broken_telemetry


def _cold_route(model_id: str = "ds/1") -> Route:
    return Route(
        model_id=model_id,
        registry_id=model_id,
        task_type="object-detection",
        action="infer",
    )


class _ScriptedColdGateway(FakeGateway):
    def __init__(self, cold_events_per_call: dict) -> None:
        super().__init__()
        self.cold_events_per_call = cold_events_per_call
        self.entered = 0

    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        call = self.entered
        self.entered += 1
        await asyncio.sleep(0)
        if call in self.cold_events_per_call:
            record_model_load(
                model_id,
                cold_start=True,
                load_time_s=self.cold_events_per_call[call],
            )
        await asyncio.sleep(0)
        return ("model_ready",)


@pytest.mark.asyncio
async def test_two_cold_events_of_one_model_scanned_by_overlapping_calls_record_twice(
    otel,
):
    bridge = LegacyModelBridge(_ScriptedColdGateway({0: 1.0, 1: 2.0}))
    route = _cold_route()
    token = MODEL_LOAD_EVENTS.set([])
    try:
        with request_telemetry_scope():
            await asyncio.gather(
                bridge.ensure_loaded(route, "k"), bridge.ensure_loaded(route, "k")
            )
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert sorted(otel.values(LOAD_DURATION)) == [1.0, 2.0]
    assert len(otel.named(LOADS)) == 2
    assert otel.values(LOADED) == [1, 1]


@pytest.mark.asyncio
async def test_concurrent_calls_record_each_event_once(otel):
    count = 6
    bridge = LegacyModelBridge(
        _ScriptedColdGateway({call: float(call + 1) for call in range(count)})
    )
    route = _cold_route()
    token = MODEL_LOAD_EVENTS.set([])
    try:
        with request_telemetry_scope():
            await asyncio.gather(
                *[bridge.ensure_loaded(route, "k") for _ in range(count)]
            )
            await bridge.ensure_loaded(route, "k")
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert sorted(otel.values(LOAD_DURATION)) == [float(i + 1) for i in range(count)]
    assert len(otel.named(LOADS)) == count


@pytest.mark.asyncio
async def test_an_event_of_another_request_is_never_recorded(otel):
    bridge = LegacyModelBridge(_ScriptedColdGateway({0: 1.0}))
    route = _cold_route()

    async def _request(events_gateway_calls):
        bridge.gateway = _ScriptedColdGateway(events_gateway_calls)
        token = MODEL_LOAD_EVENTS.set([])
        try:
            with request_telemetry_scope():
                await bridge.ensure_loaded(route, "k")
        finally:
            MODEL_LOAD_EVENTS.reset(token)

    await _request({0: 1.0})
    await _request({})

    assert otel.values(LOAD_DURATION) == [1.0]


@pytest.mark.asyncio
async def test_a_call_outside_a_request_scope_records_nothing(otel):
    bridge = LegacyModelBridge(_ScriptedColdGateway({0: 1.0}))
    token = MODEL_LOAD_EVENTS.set([])
    try:
        await bridge.ensure_loaded(_cold_route(), "k")
    finally:
        MODEL_LOAD_EVENTS.reset(token)

    assert otel.named(LOADS) == []
    assert otel.named(LOADED) == []


@pytest.mark.asyncio
async def test_the_bridge_keeps_no_per_model_telemetry_state_across_requests(otel):
    gateway = FakeGateway()
    bridge = LegacyModelBridge(gateway)
    assert not hasattr(bridge, "_recorded_load_events")

    def _sizes():
        return {
            name: len(value)
            for name, value in vars(bridge).items()
            if hasattr(value, "__len__") and not isinstance(value, str)
        }

    before = _sizes()
    for index in range(1000):
        model_id = f"ds/{index}"

        async def _ensure(model_id, instance="", api_key="", device=""):
            record_model_load(model_id, cold_start=True, load_time_s=0.1)
            return ("model_ready",)

        gateway.ensure_loaded = _ensure
        token = MODEL_LOAD_EVENTS.set([])
        try:
            with request_telemetry_scope():
                await bridge.ensure_loaded(_cold_route(model_id), "k")
        finally:
            MODEL_LOAD_EVENTS.reset(token)

    assert len(otel.named(LOADS)) == 1000
    assert _sizes() == before


def test_a_workflow_step_platform_call_records_one_api_call(
    legacy_client, fake_stat, otel, platform_api
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    api_key_hash = hashlib.md5(b"k").hexdigest()
    host.WORKFLOWS_CACHE.set(f"workflows:api_key_to_workspace:{api_key_hash}", "ws")
    specification = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "det",
                "image": "$inputs.image",
                "model_id": "ds/1",
            },
            {
                "type": "roboflow_core/roboflow_custom_metadata@v1",
                "name": "metadata",
                "predictions": "$steps.det.predictions",
                "field_name": "location",
                "field_value": "toronto",
                "fire_and_forget": False,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "error_status",
                "selector": "$steps.metadata.error_status",
            }
        ],
    }

    with requests_mock.Mocker() as mocker:
        mocker.register_uri(requests_mock.ANY, requests_mock.ANY, json={})
        response = legacy_client(_detection_gateway()).post(
            "/workflows/run",
            json={
                "specification": specification,
                "inputs": {"image": _image()},
                "api_key": "k",
            },
        )
        platform_calls = [
            request
            for request in mocker.request_history
            if "inference-stats" in request.url
        ]

    assert response.status_code == 200, response.text
    assert response.json()["outputs"][0]["error_status"] is False
    assert len(platform_calls) == 1
    assert otel.attributes(API_DURATION) == [
        {"roboflow_api.function": "add_custom_metadata"}
    ]


def _serverless_client(monkeypatch, usage_check):
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient

    inner = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @inner.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe(request: Request):
        return JSONResponse({"path": request.scope["path"]})

    inner.add_middleware(ServerlessAuthMiddleware)
    monkeypatch.setattr(platform_http, "_platform_request", usage_check)

    return TestClient(inner, raise_server_exceptions=False)


def _usage_ok(method, url, **kwargs):
    payload = {"workspace": "ws-1", "workspaceId": "db-1", "underCap": True}

    return SimpleNamespace(status_code=200, json=lambda: payload)


@pytest.fixture
def clean_auth_cache():
    serverless_auth._cache.clear()
    yield
    serverless_auth._cache.clear()


USAGE_CHECK = "get_serverless_usage_check_async"


def test_serverless_usage_check_of_a_legacy_request_records_one_api_call(
    monkeypatch, otel, clean_auth_cache
):
    client = _serverless_client(monkeypatch, _usage_ok)

    first = client.post("/infer/object_detection", json={"api_key": "k"})
    second = client.post("/infer/object_detection", json={"api_key": "k"})

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert otel.attributes(API_DURATION) == [{"roboflow_api.function": USAGE_CHECK}]


def test_serverless_usage_check_of_a_v2_request_records_nothing(
    monkeypatch, otel, clean_auth_cache
):
    client = _serverless_client(monkeypatch, _usage_ok)

    response = client.post("/v2/models/infer?api_key=k", json={})

    assert response.status_code == 200, response.text
    assert otel.named(API_DURATION) == []


def test_failed_serverless_usage_check_still_records_and_keeps_its_outcome(
    monkeypatch, otel, clean_auth_cache
):
    def _fail(method, url, **kwargs):
        raise LegacyHTTPError(500, "platform down")

    client = _serverless_client(monkeypatch, _fail)

    response = client.post("/infer/object_detection", json={"api_key": "k"})

    assert response.status_code == 500
    assert otel.attributes(API_DURATION) == [{"roboflow_api.function": USAGE_CHECK}]


def test_serverless_usage_check_outcome_survives_a_failing_recorder(
    monkeypatch, broken_telemetry, clean_auth_cache
):
    client = _serverless_client(monkeypatch, _usage_ok)

    response = client.post("/infer/object_detection", json={"api_key": "k"})

    assert response.status_code == 200, response.text
    assert "record_api_call" in broken_telemetry
