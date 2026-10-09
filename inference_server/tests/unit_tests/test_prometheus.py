import asyncio
import base64
import importlib
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from PIL import Image

import inference_server.prometheus as prometheus_mod
from inference_server.framework.dispatch import handle_model_inference_request
from inference_server.framework.entities import (
    ModelHandlerDescription,
    ModelInterfaceDescription,
)
from inference_server.framework.registry import _HANDLERS
from inference_server.prometheus import (
    MODEL_METRICS,
    UNPARSEABLE_SOURCE,
    ModelMetricsCollector,
    install_prometheus_metrics,
    measure_inference,
    record_inference,
    sanitize_source_reference,
)
from tests.unit_tests.legacy.conftest import FakeGateway


@pytest.fixture(autouse=True)
def _clear_model_metrics():
    MODEL_METRICS.clear()
    yield
    MODEL_METRICS.clear()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(
        prometheus_mod, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    return now


def _reloaded_app(monkeypatch, gateway, **overrides):
    import inference_server.app as app_mod
    from inference_server import configuration

    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: gateway
    )
    for name, value in overrides.items():
        monkeypatch.setattr(configuration, name, value)
    return importlib.reload(app_mod)


def _jpeg_b64() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode()


def _gauge_value(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split(" ", 1)[1])
    raise AssertionError(f"{name} not in metrics output")


class _StatsGateway(FakeGateway):
    def __init__(self, models: dict):
        super().__init__()
        self.models = models

    async def stats(self):
        return {"models": {k: dict(v, model_id=k) for k, v in self.models.items()}}


def test_metrics_route_is_served_by_default(monkeypatch):
    module = _reloaded_app(
        monkeypatch, _StatsGateway({"coco/3": {}}), ENABLE_PROMETHEUS=True
    )
    try:
        with TestClient(module.app) as client:
            client.get("/v2/server/health")
            record_inference("coco/3", [0.02], False)
            response = client.get("/metrics")
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=" in response.headers["content-type"]
    body = response.text
    # Same families the legacy server exposes on /metrics.
    for name in (
        "python_gc_objects_collected_total",
        "python_info",
        "http_requests_total",
        "http_request_size_bytes",
        "http_response_size_bytes",
        "http_request_duration_seconds",
        "http_request_duration_highr_seconds",
        "num_inferences_coco_3",
        "avg_inference_time_coco_3",
        "num_errors_coco_3",
        "num_inferences_total",
        "avg_inference_time_total",
        "num_errors_total",
    ):
        assert f"# TYPE {name}" in body or f"# HELP {name}" in body, name
    assert (
        'http_requests_total{handler="/v2/server/health",method="GET",status="2xx"} 1.0'
        in body
    )


def test_metrics_route_needs_no_auth_and_is_not_control_plane_gated(monkeypatch):
    module = _reloaded_app(
        monkeypatch,
        _StatsGateway({}),
        ENABLE_PROMETHEUS=True,
        ENABLE_CONTROL_PLANE_ROUTES=False,
    )
    try:
        with TestClient(module.app) as client:
            metrics = client.get("/metrics")
            v2_metrics = client.get("/v2/server/metrics")
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert metrics.status_code == 200
    # /v2/server/metrics keeps its own gate.
    assert v2_metrics.status_code == 403


def test_metrics_route_absent_when_disabled(monkeypatch):
    module = _reloaded_app(monkeypatch, _StatsGateway({}), ENABLE_PROMETHEUS=False)
    try:
        with TestClient(module.app) as client:
            response = client.get("/metrics")
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert "http_requests_total" not in response.text
    assert response.status_code != 200 or not response.headers.get(
        "content-type", ""
    ).startswith("text/plain; version=")


def test_metrics_route_renders_when_stats_fail():
    class _Broken:
        async def stats(self):
            raise RuntimeError("down")

    app = FastAPI()
    install_prometheus_metrics(app)
    app.state.model_manager = _Broken()
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "python_gc_objects_collected_total" in response.text
    assert "num_inferences_total 0.0" in response.text


def test_two_apps_do_not_collide_in_registry():
    for _ in range(2):
        app = FastAPI()
        install_prometheus_metrics(app)
        with TestClient(app) as client:
            assert client.get("/metrics").status_code == 200


def _values(collector: ModelMetricsCollector, now: float) -> dict:
    values = {}
    for model_id, metrics in collector.metrics(now=now).items():
        for name, value in metrics.items():
            values[f"{name}_{model_id}"] = value
    return values


def _collector(*loaded: str, **kwargs) -> ModelMetricsCollector:
    collector = ModelMetricsCollector(time_window=10, **kwargs)
    collector.observe_loaded(loaded or ["m/1"])
    return collector


def test_model_gauges_count_successes_errors_and_mean_duration():
    collector = _collector()
    collector.record("m/1", [0.01], error=False, now=1000.0)
    collector.record("m/1", [0.1], error=False, now=1001.0)
    collector.record("m/1", [], error=True, now=1002.0)

    values = _values(collector, now=1002.0)

    assert values["num_inferences_m/1"] == 2
    assert values["num_errors_m/1"] == 1
    assert values["avg_inference_time_m/1"] == pytest.approx(0.055)


def test_model_gauges_average_over_every_response_like_legacy():
    collector = _collector()
    collector.record("m/1", [0.01], error=False, now=1000.0)
    collector.record("m/1", [0.1, 0.1], error=False, now=1001.0)

    values = _values(collector, now=1001.0)

    assert values["num_inferences_m/1"] == 2
    assert values["avg_inference_time_m/1"] == pytest.approx(0.07)


def test_model_gauges_report_inference_on_first_scrape():
    collector = _collector()
    collector.record("m/1", [0.02], error=False, now=1000.0)

    assert _values(collector, now=1000.0)["num_inferences_m/1"] == 1


def test_model_gauges_drop_samples_older_than_window():
    collector = _collector()
    collector.record("m/1", [0.01], error=False, now=1000.0)
    collector.record("m/1", [], error=True, now=1005.0)
    collector.record("m/1", [0.05], error=False, now=1009.0)

    values = _values(collector, now=1010.5)

    assert values["num_inferences_m/1"] == 1
    assert values["num_errors_m/1"] == 1
    assert values["avg_inference_time_m/1"] == pytest.approx(0.05)
    assert _values(collector, now=1020.0)["num_inferences_m/1"] == 0


def test_model_gauges_do_not_depend_on_scrape_interval():
    frequent = _collector()
    sparse = _collector()
    for collector in (frequent, sparse):
        collector.record("m/1", [0.01], error=False, now=1000.0)
        collector.record("m/1", [0.01], error=False, now=1004.0)
    for ts in (1001.0, 1002.0, 1003.0, 1004.0):
        frequent.metrics(now=ts)

    assert frequent.metrics(now=1005.0) == sparse.metrics(now=1005.0)
    assert sparse.metrics(now=1005.0)["m/1"]["num_inferences"] == 2


def test_model_gauges_mean_is_zero_with_only_errors():
    collector = _collector()
    collector.record("m/1", [], error=True, now=1000.0)

    values = _values(collector, now=1000.0)

    assert values["num_inferences_m/1"] == 0
    assert values["num_errors_m/1"] == 1
    assert values["avg_inference_time_m/1"] == 0.0


def test_model_gauge_totals_sum_across_models(clock):
    collector = _collector("a/1", "b/1")
    collector.record("a/1", [0.01], error=False, now=1000.0)
    collector.record("b/1", [0.03], error=False, now=1000.0)
    collector.record("b/1", [], error=True, now=1000.0)

    values = {m.name: m.samples[0].value for m in collector.collect()}

    assert values["num_inferences_a_1"] == 1.0
    assert values["num_inferences_total"] == 2.0
    assert values["num_errors_total"] == 1.0
    assert values["avg_inference_time_total"] == pytest.approx(0.04)


def test_model_gauges_emit_first_25_loaded_models_in_load_order():
    loaded = [f"m{i}" for i in range(26)]
    collector = _collector(*loaded)
    for model_id in loaded:
        collector.record(model_id, [0.01], error=False, now=1000.0)

    emitted = list(collector.metrics(now=1000.0))

    assert emitted == loaded[:25]

    collector.observe_loaded(["m25"] + loaded[:25])

    assert collector.metrics(now=1000.0)["m25"]["num_inferences"] == 1


def test_model_gauges_report_only_loaded_models():
    collector = _collector("loaded/1")
    collector.record("sampled/1", [0.01], error=False, now=1000.0)

    assert list(collector.metrics(now=1000.0)) == ["loaded/1"]


def test_model_gauges_use_the_resolved_display_id():
    collector = ModelMetricsCollector(time_window=10)
    collector.observe_loaded(["coco/3"], {"coco/3": "yolov8n-640"}.get)
    collector.record("coco/3", [0.02], error=False, now=1000.0)

    assert collector.metrics(now=1000.0) == {
        "yolov8n-640": {
            "num_inferences": 1,
            "num_errors": 0,
            "avg_inference_time": 0.02,
        }
    }


def test_loaded_model_without_inferences_reads_zero():
    collector = _collector("idle/1")

    assert collector.metrics(now=1000.0) == {
        "idle/1": {"num_inferences": 0, "num_errors": 0, "avg_inference_time": 0.0}
    }


def test_unloaded_model_disappears_from_scrape_and_totals_immediately(clock):
    collector = _collector("m/1", "kept/1")
    collector.record("m/1", [0.01], error=False, now=1000.0)
    collector.observe_loaded(["kept/1"])

    values = {m.name: m.samples[0].value for m in collector.collect()}

    assert "num_inferences_m_1" not in values
    assert values["num_inferences_total"] == 0.0

    collector.observe_loaded(["m/1", "kept/1"])

    assert collector.metrics(now=1005.0)["m/1"]["num_inferences"] == 1


def test_loaded_model_keeps_zero_gauges_after_samples_expire():
    collector = _collector()
    collector.record("m/1", [0.01], error=False, now=1000.0)

    assert collector.metrics(now=1011.0)["m/1"]["num_inferences"] == 0


@pytest.mark.parametrize(
    "overrides",
    [{"OFFLINE_MODE": True}, {"DISABLE_INFERENCE_CACHE": True}, {}],
)
def test_measure_inference_honours_legacy_monitoring_opt_outs(monkeypatch, overrides):
    from inference_server import configuration

    for name, value in overrides.items():
        monkeypatch.setattr(configuration, name, value)
    monitoring = bool(overrides)
    MODEL_METRICS.observe_loaded(["m/1"])

    with measure_inference("m/1", responses=1, monitoring=monitoring):
        pass
    with pytest.raises(RuntimeError):
        with measure_inference("m/1", responses=1, monitoring=monitoring):
            raise RuntimeError("model failed")

    assert MODEL_METRICS.metrics()["m/1"] == {
        "num_inferences": 0,
        "num_errors": 0,
        "avg_inference_time": 0.0,
    }


def test_metrics_route_reports_loaded_idle_models(monkeypatch):
    module = _reloaded_app(monkeypatch, _StatsGateway({"coco/3": {}}))
    try:
        with TestClient(module.app) as client:
            body = client.get("/metrics").text
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert _gauge_value(body, "num_inferences_coco_3") == 0.0
    assert _gauge_value(body, "avg_inference_time_coco_3") == 0.0
    assert _gauge_value(body, "num_errors_coco_3") == 0.0


def test_model_gauge_help_strings_match_legacy():
    collector = _collector()
    collector.record("m/1", [0.01], error=False)

    helps = {m.name: m.documentation for m in collector.collect()}

    assert helps["num_inferences_m_1"] == "Number of inferences made in 10s"
    assert helps["avg_inference_time_m_1"] == (
        "Average inference time (over inferences completed in 10s) to infer this model"
    )
    assert helps["num_errors_m_1"] == "Number of errors in 10s"
    assert helps["num_inferences_total"] == "Total number of inferences made in 10s"
    assert helps["avg_inference_time_total"] == (
        "Average inference time (over inferences completed in 10s) to infer all models."
    )
    assert helps["num_errors_total"] == "Total number of errors in 10s"


@pytest.mark.parametrize("overrides", [{}, {"GCP_SERVERLESS": True}])
def test_metrics_endpoint_unauthenticated_text_format(monkeypatch, overrides):
    from inference_server import platform_http

    monkeypatch.setattr(
        platform_http, "_platform_request", lambda *a, **k: pytest.fail("no auth")
    )
    module = _reloaded_app(monkeypatch, _StatsGateway({}), **overrides)
    try:
        names = [m.cls.__name__ for m in module.app.user_middleware]
        with TestClient(module.app) as client:
            response = client.get("/metrics")
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert ("ServerlessAuthMiddleware" in names) is bool(overrides)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=")
    body = response.text
    assert "# HELP num_inferences_total Total number of inferences made in 10s" in body
    assert (
        "# HELP avg_inference_time_total Average inference time (over inferences "
        "completed in 10s) to infer all models." in body
    )
    assert "# HELP num_errors_total Total number of errors in 10s" in body
    assert "# TYPE num_inferences_total gauge" in body


def _scripted_model(clock, script=None):
    script = [0.01, 0.1, None] if script is None else script

    def _infer(image, params):
        duration = script.pop(0)
        if duration is None:
            raise RuntimeError("model failed")
        clock[0] += duration
        return SimpleNamespace(
            xyxy=np.array([[1, 1, 3, 5]], dtype=float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )

    return _infer


def _detection_gateway(model_id: str, clock, script=None) -> FakeGateway:
    return FakeGateway(
        predictions={(model_id, "infer"): _scripted_model(clock, script)},
        model_info={model_id: {"class_names": ["cat"], "actions": {"infer": {}}}},
    )


def _infer_body(model_id: str, images: int = 1, **extra) -> dict:
    image = {"type": "base64", "value": _jpeg_b64()}
    return {
        "model_id": model_id,
        "api_key": "k",
        "image": image if images == 1 else [image] * images,
        **extra,
    }


def test_model_metrics_after_infer(monkeypatch, fake_stat, clock):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway("ds/1", clock)
    module = _reloaded_app(monkeypatch, gateway)
    try:
        with TestClient(module.app, raise_server_exceptions=False) as client:
            statuses = [
                client.post(
                    "/infer/object_detection", json=_infer_body("ds/1")
                ).status_code
                for _ in range(3)
            ]
            first = client.get("/metrics").text
            clock[0] += 11.0
            expired = client.get("/metrics").text
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert statuses[:2] == [200, 200]
    assert statuses[2] >= 500
    assert _gauge_value(first, "num_inferences_ds_1") == 2.0
    assert _gauge_value(first, "num_errors_ds_1") == 1.0
    assert _gauge_value(first, "avg_inference_time_ds_1") == pytest.approx(0.055)
    assert _gauge_value(first, "num_inferences_total") == 2.0
    assert _gauge_value(first, "num_errors_total") == 1.0
    assert _gauge_value(expired, "num_inferences_ds_1") == 0.0
    assert _gauge_value(expired, "num_errors_ds_1") == 0.0
    assert _gauge_value(expired, "num_inferences_total") == 0.0


def test_batch_request_weights_average_per_response(monkeypatch, fake_stat, clock):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway("ds/1", clock, [0.01, 0.05, 0.05])
    module = _reloaded_app(monkeypatch, gateway)
    try:
        with TestClient(module.app) as client:
            single = client.post("/infer/object_detection", json=_infer_body("ds/1"))
            batch = client.post(
                "/infer/object_detection", json=_infer_body("ds/1", images=2)
            )
            scraped = client.get("/metrics").text
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert single.status_code == 200
    assert len(batch.json()) == 2
    assert _gauge_value(scraped, "num_inferences_ds_1") == 2.0
    assert _gauge_value(scraped, "avg_inference_time_ds_1") == pytest.approx(0.07)


def test_disable_model_monitoring_records_nothing(monkeypatch, fake_stat, clock):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = _detection_gateway("ds/1", clock, [0.01, None])
    body = _infer_body("ds/1", disable_model_monitoring=True)
    module = _reloaded_app(monkeypatch, gateway)
    try:
        with TestClient(module.app, raise_server_exceptions=False) as client:
            statuses = [
                client.post("/infer/object_detection", json=body).status_code
                for _ in range(2)
            ]
            scraped = client.get("/metrics").text
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert statuses[0] == 200
    assert statuses[1] >= 500
    assert _gauge_value(scraped, "num_inferences_ds_1") == 0.0
    assert _gauge_value(scraped, "num_errors_ds_1") == 0.0


def test_alias_request_and_idle_model_each_have_one_series(
    monkeypatch, fake_stat, clock
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gateway = _detection_gateway("coco/3", clock)
    gateway.loaded["idle/1"] = {"state": "loaded"}
    module = _reloaded_app(monkeypatch, gateway)
    try:
        with TestClient(module.app) as client:
            response = client.post(
                "/infer/object_detection", json=_infer_body("yolov8n-640")
            )
            scraped = client.get("/metrics").text
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert response.status_code == 200
    assert _series(scraped) == [
        "num_inferences_idle_1",
        "num_inferences_yolov8n_640",
        "num_inferences_total",
    ]
    assert _gauge_value(scraped, "num_inferences_yolov8n_640") == 1.0
    assert _gauge_value(scraped, "avg_inference_time_yolov8n_640") == pytest.approx(
        0.01
    )
    assert _gauge_value(scraped, "num_inferences_idle_1") == 0.0
    assert "coco_3" not in scraped


def _series(scraped: str) -> list:
    return [
        line.split(" ", 1)[0]
        for line in scraped.splitlines()
        if line.startswith("num_inferences_")
    ]


def _v2_request(model_id: str, instance: str = "") -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v2/models/infer",
        "query_string": f"model_id={model_id}&instance={instance}".encode(),
        "headers": [(b"authorization", b"Bearer k1")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


async def _dispatch_v2(model_id: str, model, count: int, instance: str = "") -> list:
    async def handler(action, input_data, proxy, hooks):
        return model(None, None)

    proxy = SimpleNamespace(ensure_loaded=AsyncMock(return_value=("model_ready",)))
    keys_before = set(_HANDLERS)
    _HANDLERS[("metrics-task", "infer")] = ModelHandlerDescription(
        input_parser=AsyncMock(return_value={"images": [b"x"], "params": {}}),
        handler=handler,
        output_serializer=lambda prediction, common: Response(status_code=200),
        interface_provider=lambda: ModelInterfaceDescription(
            task="t", params={}, output_schema={}
        ),
    )
    try:
        with patch(
            "inference_server.framework.dispatch.stat_model_while_checking_auth",
            new=AsyncMock(return_value=("metrics-task", "infer")),
        ):
            statuses = [
                (
                    await handle_model_inference_request(
                        _v2_request(model_id, instance), proxy
                    )
                ).status_code
                for _ in range(count)
            ]
    finally:
        for key in set(_HANDLERS) - keys_before:
            del _HANDLERS[key]
    return statuses


@pytest.mark.asyncio
async def test_model_metrics_after_v2_dispatch(clock):
    statuses = await _dispatch_v2("ws/7", _scripted_model(clock), 3)
    MODEL_METRICS.observe_loaded(["ws/7"])

    metrics = MODEL_METRICS.metrics()

    assert statuses == [200, 200, 500]
    assert metrics["ws/7"]["num_inferences"] == 2
    assert metrics["ws/7"]["num_errors"] == 1
    assert metrics["ws/7"]["avg_inference_time"] == pytest.approx(0.055)


def test_legacy_alias_and_v2_registry_id_share_one_series(
    monkeypatch, fake_stat, clock
):
    fake_stat["coco/3"] = ("object-detection", "infer")
    gateway = _detection_gateway("coco/3", clock, [0.01])
    module = _reloaded_app(monkeypatch, gateway)
    try:
        with TestClient(module.app) as client:
            response = client.post(
                "/infer/object_detection", json=_infer_body("yolov8n-640")
            )
            statuses = asyncio.run(
                _dispatch_v2("coco/3", _scripted_model(clock, [0.03]), 1)
            )
            scraped = client.get("/metrics").text
    finally:
        monkeypatch.undo()
        importlib.reload(module)

    assert response.status_code == 200
    assert statuses == [200]
    assert _series(scraped) == ["num_inferences_yolov8n_640", "num_inferences_total"]
    assert _gauge_value(scraped, "num_inferences_yolov8n_640") == 2.0
    assert _gauge_value(scraped, "avg_inference_time_yolov8n_640") == pytest.approx(
        0.02
    )


@pytest.mark.asyncio
async def test_v2_named_instance_records_under_its_routing_key(clock):
    statuses = await _dispatch_v2(
        "ds/1", _scripted_model(clock, [0.01]), 1, instance="blue"
    )
    MODEL_METRICS.observe_loaded(["ds/1:blue"])

    metrics = MODEL_METRICS.metrics()

    assert statuses == [200]
    assert metrics["ds/1:blue"]["num_inferences"] == 1
    assert metrics["ds/1:blue"]["avg_inference_time"] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_v2_named_instance_and_default_instance_have_separate_series(clock):
    default_statuses = await _dispatch_v2("ds/1", _scripted_model(clock, [0.01]), 1)
    named_statuses = await _dispatch_v2(
        "ds/1", _scripted_model(clock, [0.03, None]), 2, instance="blue"
    )
    MODEL_METRICS.observe_loaded(["ds/1", "ds/1:blue"])

    metrics = MODEL_METRICS.metrics()

    assert default_statuses == [200]
    assert named_statuses == [200, 500]
    assert metrics["ds/1"]["num_inferences"] == 1
    assert metrics["ds/1"]["num_errors"] == 0
    assert metrics["ds/1"]["avg_inference_time"] == pytest.approx(0.01)
    assert metrics["ds/1:blue"]["num_inferences"] == 1
    assert metrics["ds/1:blue"]["num_errors"] == 1
    assert metrics["ds/1:blue"]["avg_inference_time"] == pytest.approx(0.03)


_CLIP = "clip/ViT-B-16"
_CLIP_SERIES = "clip_ViT_B_16"


def _embedding_gateway(clock, script) -> FakeGateway:
    def _embed(image, params):
        duration = script.pop(0)
        if duration is None:
            raise RuntimeError("model failed")
        clock[0] += duration
        rows = len(params["texts"]) if "texts" in params else 1
        return np.array([[1.0, 0.0]] * rows)

    return FakeGateway(
        predictions={(_CLIP, "embed_text"): _embed, (_CLIP, "embed_images"): _embed},
        model_info={_CLIP: {"actions": {"embed_text": {}, "embed_images": {}}}},
    )


def _clip_requests(monkeypatch, gateway, path: str, bodies: list) -> tuple:
    module = _reloaded_app(monkeypatch, gateway)
    try:
        with TestClient(module.app, raise_server_exceptions=False) as client:
            statuses = [client.post(path, json=body).status_code for body in bodies]
            scraped = client.get("/metrics").text
    finally:
        monkeypatch.undo()
        importlib.reload(module)
    return statuses, scraped


def _compare_text_body(**extra) -> dict:
    return {
        "subject": "a",
        "subject_type": "text",
        "prompt": ["b"],
        "prompt_type": "text",
        **extra,
    }


def test_core_route_disable_model_monitoring_records_nothing(monkeypatch, clock):
    gateway = _embedding_gateway(clock, [0.01, None])
    body = {"text": "cat", "disable_model_monitoring": True}

    statuses, scraped = _clip_requests(
        monkeypatch, gateway, "/clip/embed_text", [body, body]
    )

    assert statuses[0] == 200
    assert statuses[1] >= 500
    assert _gauge_value(scraped, f"num_inferences_{_CLIP_SERIES}") == 0.0
    assert _gauge_value(scraped, f"num_errors_{_CLIP_SERIES}") == 0.0


def test_compare_disable_model_monitoring_records_nothing(monkeypatch, clock):
    gateway = _embedding_gateway(clock, [0.01, 0.1, 0.01, None])
    body = _compare_text_body(disable_model_monitoring=True)

    statuses, scraped = _clip_requests(
        monkeypatch, gateway, "/clip/compare", [body, body]
    )

    assert statuses[0] == 200
    assert statuses[1] >= 500
    assert _gauge_value(scraped, f"num_inferences_{_CLIP_SERIES}") == 0.0
    assert _gauge_value(scraped, f"num_errors_{_CLIP_SERIES}") == 0.0


def test_compare_records_one_event_spanning_every_internal_call(monkeypatch, clock):
    gateway = _embedding_gateway(clock, [0.01, 0.1])

    statuses, scraped = _clip_requests(
        monkeypatch, gateway, "/clip/compare", [_compare_text_body()]
    )

    assert statuses == [200]
    assert len([c for c in gateway.calls if c[0] == "infer"]) == 2
    assert _gauge_value(scraped, f"num_inferences_{_CLIP_SERIES}") == 1.0
    assert _gauge_value(scraped, f"num_errors_{_CLIP_SERIES}") == 0.0
    assert _gauge_value(scraped, f"avg_inference_time_{_CLIP_SERIES}") == pytest.approx(
        0.11
    )
    assert _gauge_value(scraped, "num_inferences_total") == 1.0


def test_compare_with_failing_internal_call_records_one_error(monkeypatch, clock):
    gateway = _embedding_gateway(clock, [0.01, None])

    statuses, scraped = _clip_requests(
        monkeypatch, gateway, "/clip/compare", [_compare_text_body()]
    )

    assert statuses[0] >= 500
    assert _gauge_value(scraped, f"num_inferences_{_CLIP_SERIES}") == 0.0
    assert _gauge_value(scraped, f"num_errors_{_CLIP_SERIES}") == 1.0
    assert _gauge_value(scraped, "num_inferences_total") == 0.0
    assert _gauge_value(scraped, "num_errors_total") == 1.0


def test_image_compare_records_one_event_for_every_image_call(monkeypatch, clock):
    gateway = _embedding_gateway(clock, [0.01, 0.02, 0.03])
    image = {"type": "base64", "value": _jpeg_b64()}
    body = {
        "subject": image,
        "subject_type": "image",
        "prompt": [image, image],
        "prompt_type": "image",
    }

    statuses, scraped = _clip_requests(monkeypatch, gateway, "/clip/compare", [body])

    assert statuses == [200]
    assert len([c for c in gateway.calls if c[0] == "infer"]) == 3
    assert _gauge_value(scraped, f"num_inferences_{_CLIP_SERIES}") == 1.0
    assert _gauge_value(scraped, f"avg_inference_time_{_CLIP_SERIES}") == pytest.approx(
        0.06
    )


class _FakeStreamManagerClient:
    def __init__(self, reports: dict, fail: bool = False):
        self.reports = reports
        self.fail = fail

    async def list_pipelines(self):
        if self.fail:
            raise ConnectionError("stream manager down")
        return SimpleNamespace(pipelines=list(self.reports))

    async def get_status(self, pipeline_id):
        return SimpleNamespace(report=self.reports[pipeline_id])


_STREAM_REPORTS = {
    "p1": {
        "inference_throughput": 12.5,
        "sources_metadata": [
            {"source_reference": "rtsp://cam/1", "source_properties": {"fps": 30.0}},
            {"source_reference": "rtsp://cam/2", "source_properties": {"fps": 20.0}},
            {"source_reference": None, "source_properties": {"fps": 0}},
        ],
        "latency_reports": [
            {
                "frame_decoding_latency": 0.25,
                "inference_latency": 0.5,
                "e2e_latency": 1.0,
            },
            {
                "frame_decoding_latency": 0.75,
                "inference_latency": 1.5,
                "e2e_latency": None,
            },
        ],
    },
    "p2": {},
}


def _scrape(stream_manager_client=None) -> str:
    app = FastAPI()
    install_prometheus_metrics(app)
    if stream_manager_client is not None:
        app.state.stream_manager_client = stream_manager_client
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    return response.text


def test_stream_gauges_when_client_present():
    body = _scrape(_FakeStreamManagerClient(_STREAM_REPORTS))

    p1 = 'pipeline_id="p1",source="rtsp://cam/1,rtsp://cam/2"'
    p2 = 'pipeline_id="p2",source=""'
    for help_line in (
        "# HELP inference_pipeline_inference_fps Inference throughput FPS",
        "# HELP inference_pipeline_camera_fps Camera source FPS",
        "# HELP inference_pipeline_frame_decoding_latency Average frame decoding "
        "latency (seconds)",
        "# HELP inference_pipeline_inference_latency Average inference latency "
        "(seconds)",
        "# HELP inference_pipeline_e2e_latency Average end-to-end latency (seconds)",
        "# HELP inference_pipeline_active_streams Number of active inference "
        "pipelines",
        "# TYPE inference_pipeline_active_streams gauge",
    ):
        assert help_line in body, help_line
    assert f"inference_pipeline_inference_fps{{{p1}}} 12.5" in body
    assert f"inference_pipeline_camera_fps{{{p1}}} 25.0" in body
    assert f"inference_pipeline_frame_decoding_latency{{{p1}}} 0.5" in body
    assert f"inference_pipeline_inference_latency{{{p1}}} 1.0" in body
    assert f"inference_pipeline_e2e_latency{{{p1}}} 1.0" in body
    assert f"inference_pipeline_inference_fps{{{p2}}} 0.0" in body
    assert f"inference_pipeline_camera_fps{{{p2}}} 0.0" in body
    assert f"inference_pipeline_e2e_latency{{{p2}}} 0.0" in body
    assert "inference_pipeline_active_streams 2.0" in body


def test_stream_gauges_omit_source_label_values_when_disabled(monkeypatch):
    from inference_server import configuration

    monkeypatch.setattr(configuration, "METRICS_INCLUDE_SOURCE_LABELS", False)

    body = _scrape(_FakeStreamManagerClient(_STREAM_REPORTS))

    assert 'inference_pipeline_inference_fps{pipeline_id="p1",source=""} 12.5' in body
    assert "rtsp://" not in body


def test_stream_gauges_empty_without_client():
    body = _scrape()

    for name in (
        "inference_pipeline_inference_fps",
        "inference_pipeline_camera_fps",
        "inference_pipeline_frame_decoding_latency",
        "inference_pipeline_inference_latency",
        "inference_pipeline_e2e_latency",
    ):
        assert f"# TYPE {name} gauge" in body, name
    assert 'pipeline_id="' not in body
    assert "inference_pipeline_active_streams 0.0" in body
    assert "num_inferences_total" in body


@pytest.mark.parametrize(
    "reference,expected",
    [
        ("rtsp://user:secret@camera/live?token=abc", "rtsp://camera/live"),
        ("/data/videos/cam1.mp4", "/data/videos/cam1.mp4"),
        ("rtsp://user:pa/ss@cam:554/live", UNPARSEABLE_SOURCE),
    ],
)
def test_sanitize_source_reference(reference, expected):
    assert sanitize_source_reference(reference) == expected


def test_stream_source_labels_are_sanitized():
    reports = {
        "p1": {
            "sources_metadata": [
                {"source_reference": "rtsp://user:secret@camera/live?token=abc"},
                {"source_reference": "/data/videos/cam1.mp4"},
                {"source_reference": "rtsp://user:pa/ss@cam:554/live"},
            ]
        }
    }

    body = _scrape(_FakeStreamManagerClient(reports))

    assert (
        'inference_pipeline_camera_fps{pipeline_id="p1",source="rtsp://camera/live,'
        f'/data/videos/cam1.mp4,{UNPARSEABLE_SOURCE}"}} 0.0'
    ) in body
    assert "secret" not in body
    assert "token" not in body


def test_stream_gauges_empty_when_client_fails():
    body = _scrape(_FakeStreamManagerClient(_STREAM_REPORTS, fail=True))

    assert "# HELP inference_pipeline_inference_fps" in body
    assert 'pipeline_id="' not in body
    assert "inference_pipeline_active_streams 0.0" in body
