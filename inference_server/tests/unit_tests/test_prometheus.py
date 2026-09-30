import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_server.prometheus import (
    ModelMetricsCollector,
    install_prometheus_metrics,
)
from tests.unit_tests.legacy.conftest import FakeGateway


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
    gateway = _StatsGateway(
        {"coco/3": {"inference_count": 4, "error_count": 1, "latency_p50_ms": 20.0}}
    )
    module = _reloaded_app(monkeypatch, gateway, ENABLE_PROMETHEUS=True)
    try:
        with TestClient(module.app) as client:
            client.get("/v2/server/health")
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


def _stats(count: int, errors: int = 0, p50_ms: float = 0.0) -> dict:
    return {
        "models": {
            "m/1": {
                "inference_count": count,
                "error_count": errors,
                "latency_p50_ms": p50_ms,
            }
        }
    }


def _collected(collector: ModelMetricsCollector) -> dict:
    return {m.name: m.samples[0].value for m in collector.collect()}


def test_model_gauges_report_zero_without_baseline():
    collector = ModelMetricsCollector(time_window=10)
    collector.observe(_stats(100, 3, 50.0), now=1000.0)

    values = _collected(collector)

    assert values["num_inferences_m_1"] == 0.0
    assert values["num_errors_m_1"] == 0.0
    assert values["avg_inference_time_m_1"] == 0.0
    assert values["num_inferences_total"] == 0.0


def test_model_gauges_are_deltas_over_the_window():
    collector = ModelMetricsCollector(time_window=10)
    collector.observe(_stats(100, 3, 50.0), now=1000.0)
    collector.observe(_stats(120, 5, 50.0), now=1010.0)

    values = _collected(collector)

    assert values["num_inferences_m_1"] == 20.0
    assert values["num_errors_m_1"] == 2.0
    assert values["avg_inference_time_m_1"] == 0.05
    assert values["num_inferences_total"] == 20.0
    assert values["num_errors_total"] == 2.0
    assert values["avg_inference_time_total"] == 0.05


def test_model_gauges_rescale_sparse_scrapes_to_window():
    collector = ModelMetricsCollector(time_window=10)
    collector.observe(_stats(0), now=1000.0)
    collector.observe(_stats(30), now=1015.0)

    assert _collected(collector)["num_inferences_m_1"] == 20.0


def test_model_gauges_use_oldest_sample_inside_window_on_frequent_scrapes():
    collector = ModelMetricsCollector(time_window=10)
    for i, ts in enumerate((1000.0, 1004.0, 1008.0, 1012.0)):
        collector.observe(_stats(i * 4), now=ts)

    # Baseline = newest sample at/before 1002 -> (1000, 0); 12 in 12s -> 10/10s.
    assert _collected(collector)["num_inferences_m_1"] == 10.0


def test_model_gauges_handle_counter_reset_and_unloaded_models():
    collector = ModelMetricsCollector(time_window=10)
    collector.observe(_stats(500), now=1000.0)
    collector.observe(_stats(7), now=1010.0)

    assert _collected(collector)["num_inferences_m_1"] == 7.0

    collector.observe({"models": {}}, now=1020.0)
    names = set(_collected(collector))

    assert "num_inferences_m_1" not in names
    assert "num_inferences_total" in names


def test_model_gauges_cap_number_of_models():
    collector = ModelMetricsCollector(time_window=10, max_models=2)
    collector.observe(
        {"models": {f"m{i}": {"inference_count": 1} for i in range(5)}}, now=1.0
    )

    per_model = [n for n in _collected(collector) if n.startswith("num_errors_m")]

    assert len(per_model) == 2
