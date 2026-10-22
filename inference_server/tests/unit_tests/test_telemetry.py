import importlib
import logging
import os
import subprocess
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

import inference_server.app as app_mod
import inference_server.telemetry as telemetry_module
from inference_server import configuration
from inference_server.cors import PathAwareCORSMiddleware
from inference_server.hosted import serverless_auth
from inference_server.hosted.serverless_auth import ServerlessAuthMiddleware
from inference_server.legacy.errors import _BodyLimitMiddleware
from inference_server.middlewares.correlation_id import CorrelationIdMiddleware
from inference_server.middlewares.model_load import ModelLoadHeadersMiddleware

FAKE_MODULE_NAMES = (
    "opentelemetry",
    "opentelemetry.context",
    "opentelemetry.trace",
    "opentelemetry.trace.propagation",
    "opentelemetry.trace.propagation.tracecontext",
    "opentelemetry.propagate",
    "opentelemetry.propagators",
    "opentelemetry.propagators.composite",
    "opentelemetry.metrics",
    "opentelemetry.sdk",
    "opentelemetry.sdk.resources",
    "opentelemetry.sdk.trace",
    "opentelemetry.sdk.trace.export",
    "opentelemetry.sdk.trace.sampling",
    "opentelemetry.sdk.metrics",
    "opentelemetry.sdk.metrics.export",
    "opentelemetry.exporter",
    "opentelemetry.exporter.otlp",
    "opentelemetry.exporter.otlp.proto",
    "opentelemetry.exporter.otlp.proto.grpc",
    "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
    "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
    "opentelemetry.exporter.otlp.proto.http",
    "opentelemetry.exporter.otlp.proto.http.trace_exporter",
    "opentelemetry.exporter.otlp.proto.http.metric_exporter",
    "opentelemetry.instrumentation",
    "opentelemetry.instrumentation.fastapi",
    "opentelemetry.instrumentation.requests",
)
TRACE_ID = 0x5B8AA5A2D2C872E8321CF37308D69DF2
SPAN_ID = 0x051581BF3CB55C13


def _span(trace_id: int = TRACE_ID, span_id: int = SPAN_ID, recording: bool = True):
    return SimpleNamespace(
        get_span_context=lambda: SimpleNamespace(trace_id=trace_id, span_id=span_id),
        is_recording=lambda: recording,
        record_exception=MagicMock(),
        set_status=MagicMock(),
        set_attribute=MagicMock(),
    )


class _FakeContextModule:
    """Faithful stand-in for ``opentelemetry.context``: dict contexts, one current."""

    def __init__(self) -> None:
        self.current: dict = {}
        self.Context = dict

    def create_key(self, name: str) -> str:
        return f"{name}-key"

    def get_value(self, key, context=None):
        source = self.current if context is None else context
        return source.get(key)

    def set_value(self, key, value, context=None):
        source = self.current if context is None else context
        updated = dict(source)
        updated[key] = value
        return updated

    def get_current(self):
        return self.current

    def attach(self, context):
        previous = self.current
        self.current = context
        return previous

    def detach(self, token):
        self.current = token


class _Decision:
    DROP = "DROP"
    RECORD_AND_SAMPLE = "RECORD_AND_SAMPLE"


class _SamplingResult:
    def __init__(self, decision, attributes=None, trace_state=None) -> None:
        self.decision = decision
        self.attributes = attributes or {}
        self.trace_state = trace_state


class _StaticSampler:
    def __init__(self, decision) -> None:
        self._decision = decision

    def should_sample(self, *args, **kwargs):
        return _SamplingResult(self._decision)

    def get_description(self) -> str:
        return f"Static{{{self._decision}}}"


class _ParentBased:
    """Faithful stand-in for the SDK ``ParentBased``: parent wins, else root."""

    def __init__(self, root, **kwargs) -> None:
        self._root = root

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        links=None,
        trace_state=None,
    ):
        parent = (parent_context or {}).get("parent_span")
        if parent is not None and parent.is_valid:
            decision = _Decision.RECORD_AND_SAMPLE if parent.sampled else _Decision.DROP
            return _SamplingResult(decision, attributes, trace_state)
        return self._root.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )


class _DictGetter:
    def get(self, carrier, key):
        value = carrier.get(key)
        return None if value is None else [value]


class _ScopeGetter:
    def get(self, scope, key):
        wanted = key.lower().encode("latin-1")
        values = [
            value.decode("latin-1")
            for name, value in scope.get("headers", [])
            if name.lower() == wanted
        ]
        return values or None


class _CompositePropagator:
    def __init__(self, propagators) -> None:
        self.propagators = list(propagators)

    def extract(self, carrier, context=None, getter=None):
        for propagator in self.propagators:
            context = propagator.extract(carrier, context, getter=getter)
        return context


def _faithful_sampling(mocks: dict) -> None:
    sampling = mocks["sampling"]
    sampling.ALWAYS_OFF = _StaticSampler(_Decision.DROP)
    sampling.ALWAYS_ON = _StaticSampler(_Decision.RECORD_AND_SAMPLE)
    sampling.ParentBased = _ParentBased
    sampling.Decision = _Decision
    sampling.SamplingResult = _SamplingResult
    for name, value in vars(sampling).items():
        setattr(sys.modules["opentelemetry.sdk.trace.sampling"], name, value)
    mocks["composite_propagator_cls"].side_effect = _CompositePropagator
    mocks["tracecontext_propagator_cls"].return_value.extract.side_effect = (
        lambda carrier, context=None, getter=None: {} if context is None else context
    )


def _build_fake_modules(patch: pytest.MonkeyPatch) -> dict:
    modules = {}
    for name in FAKE_MODULE_NAMES:
        modules[name] = types.ModuleType(name)
        patch.setitem(sys.modules, name, modules[name])

    mocks = {
        "get_current_span": MagicMock(return_value=_span(trace_id=0, span_id=0)),
        "get_tracer": MagicMock(),
        "set_tracer_provider": MagicMock(),
        "status_code": SimpleNamespace(ERROR="ERROR"),
        "context": _FakeContextModule(),
        "inject": MagicMock(),
        "set_global_textmap": MagicMock(),
        "composite_propagator_cls": MagicMock(),
        "tracecontext_propagator_cls": MagicMock(),
        "resource_cls": MagicMock(),
        "provider_cls": MagicMock(),
        "batch_processor_cls": MagicMock(),
        "sampling": SimpleNamespace(
            ALWAYS_OFF=MagicMock(name="ALWAYS_OFF"),
            ALWAYS_ON=MagicMock(name="ALWAYS_ON"),
            ParentBased=MagicMock(),
            TraceIdRatioBased=MagicMock(),
            Decision=SimpleNamespace(RECORD_AND_SAMPLE="RECORD_AND_SAMPLE"),
            SamplingResult=MagicMock(),
        ),
        "grpc_exporter_cls": MagicMock(),
        "http_exporter_cls": MagicMock(),
        "grpc_metric_exporter_cls": MagicMock(),
        "http_metric_exporter_cls": MagicMock(),
        "set_meter_provider": MagicMock(),
        "meter_provider_cls": MagicMock(),
        "metric_reader_cls": MagicMock(),
        "fastapi_instrumentor_cls": MagicMock(),
        "requests_instrumentor_cls": MagicMock(),
    }

    trace_module = modules["opentelemetry.trace"]
    trace_module.get_current_span = mocks["get_current_span"]
    trace_module.get_tracer = mocks["get_tracer"]
    trace_module.set_tracer_provider = mocks["set_tracer_provider"]
    trace_module.StatusCode = mocks["status_code"]
    modules[
        "opentelemetry.trace.propagation.tracecontext"
    ].TraceContextTextMapPropagator = mocks["tracecontext_propagator_cls"]
    context_module = modules["opentelemetry.context"]
    for name in ("Context", "create_key", "get_value", "set_value", "get_current"):
        setattr(context_module, name, getattr(mocks["context"], name))
    context_module.attach = mocks["context"].attach
    context_module.detach = mocks["context"].detach
    modules["opentelemetry.propagate"].inject = mocks["inject"]
    modules["opentelemetry.propagate"].set_global_textmap = mocks["set_global_textmap"]
    modules["opentelemetry.propagators.composite"].CompositePropagator = mocks[
        "composite_propagator_cls"
    ]
    modules["opentelemetry.metrics"].set_meter_provider = mocks["set_meter_provider"]
    modules["opentelemetry.sdk.resources"].Resource = mocks["resource_cls"]
    modules["opentelemetry.sdk.trace"].TracerProvider = mocks["provider_cls"]
    modules["opentelemetry.sdk.trace.export"].BatchSpanProcessor = mocks[
        "batch_processor_cls"
    ]
    for name, value in vars(mocks["sampling"]).items():
        setattr(modules["opentelemetry.sdk.trace.sampling"], name, value)
    modules["opentelemetry.sdk.metrics"].MeterProvider = mocks["meter_provider_cls"]
    modules["opentelemetry.sdk.metrics.export"].PeriodicExportingMetricReader = mocks[
        "metric_reader_cls"
    ]
    modules[
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    ].OTLPSpanExporter = mocks["grpc_exporter_cls"]
    modules[
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    ].OTLPSpanExporter = mocks["http_exporter_cls"]
    modules[
        "opentelemetry.exporter.otlp.proto.grpc.metric_exporter"
    ].OTLPMetricExporter = mocks["grpc_metric_exporter_cls"]
    modules[
        "opentelemetry.exporter.otlp.proto.http.metric_exporter"
    ].OTLPMetricExporter = mocks["http_metric_exporter_cls"]
    modules["opentelemetry.instrumentation.fastapi"].FastAPIInstrumentor = mocks[
        "fastapi_instrumentor_cls"
    ]
    modules["opentelemetry.instrumentation.requests"].RequestsInstrumentor = mocks[
        "requests_instrumentor_cls"
    ]

    return mocks


@pytest.fixture
def fake_otel():
    patch = pytest.MonkeyPatch()
    mocks = _build_fake_modules(patch)
    telemetry = importlib.reload(telemetry_module)
    assert telemetry._OTEL_AVAILABLE is True
    patch.setattr(configuration, "OTEL_TRACING_ENABLED", True)
    patch.setattr(configuration, "OTEL_METRICS_ENABLED", True)
    try:
        yield telemetry, mocks, patch
    finally:
        patch.undo()
        importlib.reload(telemetry_module)


@pytest.fixture
def no_otel():
    patch = pytest.MonkeyPatch()
    patch.setitem(sys.modules, "opentelemetry", None)
    telemetry = importlib.reload(telemetry_module)
    assert telemetry._OTEL_AVAILABLE is False
    try:
        yield telemetry, patch
    finally:
        patch.undo()
        importlib.reload(telemetry_module)


def _probe_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe() -> Response:
        return Response(content=b"ok")

    return app


def test_helpers_are_noops_without_opentelemetry(no_otel):
    telemetry, patch = no_otel
    patch.setattr(configuration, "OTEL_TRACING_ENABLED", True)

    with telemetry.start_span("model.infer", {"model.id": "m"}) as span:
        assert span is None
    telemetry.record_error(ValueError("boom"))
    telemetry.set_span_attribute("k", "v")
    assert telemetry.get_trace_id() is None
    assert telemetry.trace_context_fields() == {}
    assert telemetry.capture_context() is None
    assert telemetry.attach_context("ctx") is None
    telemetry.detach_context(None)
    assert telemetry.inject_trace_context(None) == {}
    headers = {"a": "b"}
    assert telemetry.inject_trace_context(headers) is headers
    telemetry.record_model_loaded("m", 1.0)
    telemetry.record_model_unloaded("m")
    telemetry.record_inference("m", 0.1)
    telemetry.record_api_call("get_model", 0.1)
    telemetry.record_error_metric("ValueError")
    telemetry.shutdown_telemetry()


def test_setup_telemetry_is_noop_when_disabled(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_TRACING_ENABLED", False)
    app = MagicMock()

    telemetry.setup_telemetry(app)

    app.add_middleware.assert_not_called()
    mocks["provider_cls"].assert_not_called()
    assert telemetry.get_trace_id() is None


def test_setup_telemetry_warns_when_packages_missing(no_otel, caplog):
    telemetry, patch = no_otel
    patch.setattr(configuration, "OTEL_TRACING_ENABLED", True)
    app = MagicMock()

    with caplog.at_level(logging.WARNING, logger="inference_server.telemetry"):
        telemetry.setup_telemetry(app)

    app.add_middleware.assert_not_called()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "opentelemetry packages are not installed" in warnings[0].getMessage()


@pytest.mark.parametrize(
    "missing_module",
    [
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        "opentelemetry.instrumentation.fastapi",
        "opentelemetry.instrumentation.requests",
        "opentelemetry.sdk.trace",
    ],
)
def test_setup_telemetry_warns_when_a_tracing_package_is_missing(
    fake_otel, caplog, missing_module
):
    telemetry, mocks, patch = fake_otel
    patch.setitem(sys.modules, missing_module, None)
    app = _probe_app()

    with caplog.at_level(logging.WARNING, logger="inference_server.telemetry"):
        telemetry.setup_telemetry(app)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "tracing is disabled" in warnings[0].getMessage()
    assert app.user_middleware == []
    assert telemetry._provider is None
    mocks["set_tracer_provider"].assert_not_called()
    assert TestClient(app).get("/probe").status_code == 200


def test_setup_telemetry_imports_only_the_selected_protocol_exporter(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_EXPORTER_PROTOCOL", "grpc")
    patch.setitem(
        sys.modules, "opentelemetry.exporter.otlp.proto.http.trace_exporter", None
    )
    patch.setitem(
        sys.modules, "opentelemetry.exporter.otlp.proto.http.metric_exporter", None
    )

    telemetry.setup_telemetry(_probe_app())

    mocks["grpc_exporter_cls"].assert_called_once()
    mocks["grpc_metric_exporter_cls"].assert_called_once()
    assert telemetry._provider is mocks["provider_cls"].return_value


def test_setup_telemetry_configures_tracing_from_env(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_EXPORTER_PROTOCOL", "grpc")
    patch.setattr(configuration, "OTEL_EXPORTER_ENDPOINT", "collector:4317")
    patch.setattr(configuration, "OTEL_SERVICE_NAME", "my-service")
    patch.setattr(configuration, "OTEL_SAMPLING_RATE", 0.5)
    patch.setattr(configuration, "OTEL_TRACE_EXPORT_INTERVAL_MS", 1234)
    patch.setattr(configuration, "OTEL_METRICS_ENABLED", False)
    patch.setattr(configuration, "INFERENCE_SERVER_ID", "server-1")
    app = _probe_app()

    telemetry.setup_telemetry(app)

    mocks["grpc_exporter_cls"].assert_called_once_with(
        endpoint="collector:4317", insecure=True
    )
    mocks["http_exporter_cls"].assert_not_called()
    mocks["sampling"].TraceIdRatioBased.assert_called_once_with(0.5)
    mocks["resource_cls"].create.assert_called_once_with(
        {"service.name": "my-service", "service.instance.id": "server-1"}
    )
    provider = mocks["provider_cls"].return_value
    mocks["provider_cls"].assert_called_once_with(
        resource=mocks["resource_cls"].create.return_value,
        sampler=mocks["sampling"].ParentBased.return_value,
    )
    mocks["batch_processor_cls"].assert_called_once_with(
        mocks["grpc_exporter_cls"].return_value, schedule_delay_millis=1234
    )
    provider.add_span_processor.assert_called_once_with(
        mocks["batch_processor_cls"].return_value
    )
    mocks["set_tracer_provider"].assert_called_once_with(provider)
    mocks["composite_propagator_cls"].assert_called_once()
    propagators = mocks["composite_propagator_cls"].call_args.args[0]
    assert propagators[0] is mocks["tracecontext_propagator_cls"].return_value
    assert isinstance(propagators[1], telemetry._ForceTracePropagator)
    mocks["set_global_textmap"].assert_called_once_with(
        mocks["composite_propagator_cls"].return_value
    )
    mocks["fastapi_instrumentor_cls"].instrument_app.assert_called_once_with(app)
    mocks["requests_instrumentor_cls"].return_value.instrument.assert_called_once_with()
    assert [entry.cls for entry in app.user_middleware] == [
        telemetry.TraceIdResponseMiddleware
    ]
    mocks["meter_provider_cls"].assert_not_called()
    assert telemetry._metrics is None


def test_setup_telemetry_uses_http_exporters_for_http_protocol(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_EXPORTER_PROTOCOL", "http")
    patch.setattr(configuration, "OTEL_EXPORTER_ENDPOINT", "collector:4318")

    telemetry.setup_telemetry(_probe_app())

    mocks["http_exporter_cls"].assert_called_once_with(
        endpoint="http://collector:4318/v1/traces"
    )
    mocks["grpc_exporter_cls"].assert_not_called()
    mocks["http_metric_exporter_cls"].assert_called_once_with(
        endpoint="http://collector:4318/v1/metrics"
    )
    mocks["grpc_metric_exporter_cls"].assert_not_called()


@pytest.mark.parametrize(
    "rate,expected",
    [(0.0, "ALWAYS_OFF"), (-1.0, "ALWAYS_OFF"), (1.0, "ALWAYS_ON"), (2.0, "ALWAYS_ON")],
)
def test_setup_telemetry_picks_constant_samplers_at_the_bounds(
    fake_otel, rate, expected
):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_SAMPLING_RATE", rate)

    telemetry.setup_telemetry(_probe_app())

    root = mocks["sampling"].ParentBased.call_args.kwargs["root"]
    assert isinstance(root, telemetry._ForceTraceRootSampler)
    assert root._delegate is getattr(mocks["sampling"], expected)
    mocks["sampling"].TraceIdRatioBased.assert_not_called()


def test_setup_telemetry_installs_metrics_provider_from_env(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_EXPORTER_PROTOCOL", "grpc")
    patch.setattr(configuration, "OTEL_EXPORTER_ENDPOINT", "collector:4317")
    patch.setattr(configuration, "OTEL_METRIC_EXPORTER_ENDPOINT", "")
    patch.setattr(configuration, "OTEL_METRIC_EXPORT_INTERVAL_MS", 777)

    telemetry.setup_telemetry(_probe_app())

    mocks["grpc_metric_exporter_cls"].assert_called_once_with(
        endpoint="collector:4317", insecure=True
    )
    mocks["metric_reader_cls"].assert_called_once_with(
        mocks["grpc_metric_exporter_cls"].return_value, export_interval_millis=777
    )
    mocks["meter_provider_cls"].assert_called_once_with(
        resource=mocks["resource_cls"].create.return_value,
        metric_readers=[mocks["metric_reader_cls"].return_value],
    )
    meter_provider = mocks["meter_provider_cls"].return_value
    mocks["set_meter_provider"].assert_called_once_with(meter_provider)
    meter = meter_provider.get_meter.return_value
    assert set(telemetry._metrics) == {
        "models_loaded",
        "model_loads",
        "model_unloads",
        "model_load_duration",
        "model_infer_count",
        "model_infer_duration",
        "api_call_duration",
        "errors",
    }
    assert meter.create_counter.call_count == 4
    assert meter.create_histogram.call_count == 3
    meter.create_up_down_counter.assert_called_once()


def test_setup_telemetry_metrics_endpoint_falls_back_to_dedicated_endpoint(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_EXPORTER_ENDPOINT", "collector:4317")
    patch.setattr(configuration, "OTEL_METRIC_EXPORTER_ENDPOINT", "metrics:4317")

    telemetry.setup_telemetry(_probe_app())

    mocks["grpc_metric_exporter_cls"].assert_called_once_with(
        endpoint="metrics:4317", insecure=True
    )


def test_setup_telemetry_skips_metrics_when_disabled(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_METRICS_ENABLED", False)

    telemetry.setup_telemetry(_probe_app())

    mocks["meter_provider_cls"].assert_not_called()
    mocks["set_meter_provider"].assert_not_called()
    assert telemetry._metrics is None
    assert telemetry._provider is mocks["provider_cls"].return_value


def test_setup_telemetry_keeps_tracing_when_metrics_packages_are_missing(
    fake_otel, caplog
):
    telemetry, mocks, patch = fake_otel
    patch.setitem(sys.modules, "opentelemetry.sdk.metrics", None)

    with caplog.at_level(logging.WARNING, logger="inference_server.telemetry"):
        telemetry.setup_telemetry(_probe_app())

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "metrics are disabled" in warnings[0].getMessage()
    assert telemetry._metrics is None
    mocks["set_tracer_provider"].assert_called_once()


def test_metric_recorders_write_to_the_instruments(fake_otel):
    telemetry, mocks, patch = fake_otel
    meter = mocks["meter_provider_cls"].return_value.get_meter.return_value
    for factory in ("create_counter", "create_histogram", "create_up_down_counter"):
        getattr(meter, factory).side_effect = lambda *args, **kwargs: MagicMock()
    telemetry.setup_telemetry(_probe_app())
    instruments = telemetry._metrics

    telemetry.record_model_loaded("ds/1", 2.5)
    telemetry.record_model_unloaded("ds/1")
    telemetry.record_inference("ds/1", 0.25)
    telemetry.record_api_call("get_model", 0.5)
    telemetry.record_error_metric("ValueError")

    instruments["models_loaded"].add.assert_any_call(1)
    instruments["models_loaded"].add.assert_any_call(-1)
    instruments["model_loads"].add.assert_called_once_with(1, {"model.id": "ds/1"})
    instruments["model_load_duration"].record.assert_called_once_with(
        2.5, {"model.id": "ds/1"}
    )
    instruments["model_unloads"].add.assert_called_once_with(1, {"model.id": "ds/1"})
    instruments["model_infer_count"].add.assert_called_once_with(
        1, {"model.id": "ds/1"}
    )
    instruments["model_infer_duration"].record.assert_called_once_with(
        0.25, {"model.id": "ds/1"}
    )
    instruments["api_call_duration"].record.assert_called_once_with(
        0.5, {"roboflow_api.function": "get_model"}
    )
    instruments["errors"].add.assert_called_once_with(1, {"error.type": "ValueError"})


def test_setup_telemetry_installs_providers_once_per_process(fake_otel):
    telemetry, mocks, patch = fake_otel
    first = _probe_app()
    second = _probe_app()

    telemetry.setup_telemetry(first)
    telemetry.setup_telemetry(second)

    mocks["provider_cls"].assert_called_once()
    mocks["meter_provider_cls"].assert_called_once()
    mocks["requests_instrumentor_cls"].return_value.instrument.assert_called_once()
    assert mocks["fastapi_instrumentor_cls"].instrument_app.call_count == 2
    assert len(first.user_middleware) == 1
    assert len(second.user_middleware) == 1


@pytest.mark.asyncio
async def test_lifespan_shutdown_calls_shutdown_telemetry(monkeypatch):
    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    calls = []

    class _StubProxy:
        async def start(self):
            pass

        async def shutdown(self):
            calls.append("proxy")

    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: _StubProxy()
    )
    monkeypatch.setattr(
        app_mod, "shutdown_telemetry", lambda: calls.append("telemetry")
    )

    async with app_mod._lifespan(app_mod.app):
        assert calls == []

    assert calls == ["telemetry", "proxy"]


def test_shutdown_telemetry_shuts_down_both_providers(fake_otel):
    telemetry, mocks, patch = fake_otel
    telemetry.setup_telemetry(_probe_app())

    telemetry.shutdown_telemetry()

    mocks["provider_cls"].return_value.shutdown.assert_called_once_with()
    mocks["meter_provider_cls"].return_value.shutdown.assert_called_once_with()


def test_span_helpers_use_the_current_span(fake_otel):
    telemetry, mocks, patch = fake_otel
    span = _span()
    mocks["get_current_span"].return_value = span
    tracer = mocks["get_tracer"].return_value
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    error = ValueError("boom")

    with telemetry.start_span("model.infer", {"model.id": "m"}) as started:
        assert started is span
    telemetry.record_error(error)
    telemetry.set_span_attribute("k", "v")
    headers = telemetry.inject_trace_context({"a": "b"})
    mocks["context"].current = {"k": "outer"}
    captured = telemetry.capture_context()
    mocks["context"].current = {"k": "inner"}
    token = telemetry.attach_context(captured)
    attached = mocks["context"].get_current()
    telemetry.detach_context(token)

    tracer.start_as_current_span.assert_called_once_with(
        "model.infer", attributes={"model.id": "m"}
    )
    span.record_exception.assert_called_once_with(error)
    span.set_status.assert_called_once_with("ERROR", "boom")
    span.set_attribute.assert_called_once_with("k", "v")
    mocks["inject"].assert_called_once_with({"a": "b"})
    assert headers == {"a": "b"}
    assert captured == {"k": "outer"}
    assert attached == {"k": "outer"}
    assert token == {"k": "inner"}
    assert mocks["context"].get_current() == {"k": "inner"}
    assert telemetry.get_trace_id() == format(TRACE_ID, "032x")
    assert telemetry.trace_context_fields() == {
        "trace_id": format(TRACE_ID, "032x"),
        "span_id": format(SPAN_ID, "016x"),
    }


def test_span_helpers_skip_non_recording_spans(fake_otel):
    telemetry, mocks, patch = fake_otel
    span = _span(recording=False)
    mocks["get_current_span"].return_value = span

    telemetry.record_error(ValueError("boom"))
    telemetry.set_span_attribute("k", "v")

    span.record_exception.assert_not_called()
    span.set_attribute.assert_not_called()


def test_offline_mode_forces_tracing_and_metrics_off():
    code = (
        "from inference_server import configuration as c; "
        "assert c.OTEL_TRACING_ENABLED is False; "
        "assert c.OTEL_METRICS_ENABLED is False"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "OFFLINE_MODE": "true",
            "OTEL_TRACING_ENABLED": "true",
            "OTEL_METRICS_ENABLED": "true",
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_helpers_work_on_an_external_span_without_the_tracing_flag(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OTEL_TRACING_ENABLED", False)
    span = _span()
    mocks["get_current_span"].return_value = span
    error = ValueError("boom")

    telemetry.record_error(error)
    telemetry.set_span_attribute("k", "v")
    headers = telemetry.inject_trace_context({})

    assert telemetry.get_trace_id() == format(TRACE_ID, "032x")
    assert telemetry.trace_context_fields()["span_id"] == format(SPAN_ID, "016x")
    span.record_exception.assert_called_once_with(error)
    span.set_attribute.assert_called_once_with("k", "v")
    mocks["inject"].assert_called_once_with(headers)
    assert telemetry.capture_context() is mocks["context"].get_current()
    app = _probe_app()
    telemetry.setup_telemetry(app)
    assert app.user_middleware == []
    mocks["provider_cls"].assert_not_called()


def test_helpers_are_noops_under_offline_mode(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(configuration, "OFFLINE_MODE", True)
    span = _span()
    mocks["get_current_span"].return_value = span

    telemetry.record_error(ValueError("boom"))
    with telemetry.start_span("x") as started:
        assert started is None

    assert telemetry.get_trace_id() is None
    assert telemetry.trace_context_fields() == {}
    assert telemetry.capture_context() is None
    span.record_exception.assert_not_called()


@pytest.mark.parametrize(
    "carrier,expected",
    [
        ({}, None),
        ({"x-force-trace": "true"}, True),
        ({"x-force-trace": " TRUE "}, True),
        ({"x-force-trace": "yes"}, None),
    ],
)
def test_force_trace_propagator_reads_the_header_into_the_context(
    fake_otel, carrier, expected
):
    telemetry, mocks, patch = fake_otel
    propagator = telemetry._ForceTracePropagator()

    extracted = propagator.extract(carrier, {"other": 1}, getter=_DictGetter())

    assert extracted.get("other") == 1
    assert extracted.get(telemetry._FORCE_TRACE_KEY) == expected
    assert propagator.extract(carrier, getter=_DictGetter()) is not None
    assert propagator.fields == set()
    assert propagator.inject({}, None, None) is None


def test_force_trace_sampler_delegates_unforced_root_spans(fake_otel):
    telemetry, mocks, patch = fake_otel
    delegate = MagicMock()
    delegate.get_description.return_value = "TraceIdRatioBased{0.5}"
    sampler = telemetry._ForceTraceRootSampler(delegate, mocks["sampling"])

    result = sampler.should_sample({}, TRACE_ID, "GET /x", "kind", {"a": 1}, [], "ts")

    delegate.should_sample.assert_called_once_with(
        {}, TRACE_ID, "GET /x", "kind", {"a": 1}, [], "ts"
    )
    assert result is delegate.should_sample.return_value
    assert sampler.get_description() == "ForceTraceRootSampler(TraceIdRatioBased{0.5})"


def _installed_sampler_and_propagator(telemetry, mocks, patch, *, rate: float):
    _faithful_sampling(mocks)
    patch.setattr(configuration, "OTEL_SAMPLING_RATE", rate)
    app = _probe_app()
    telemetry.setup_telemetry(app)
    sampler = mocks["provider_cls"].call_args.kwargs["sampler"]
    propagator = mocks["set_global_textmap"].call_args.args[0]

    return app, sampler, propagator


@pytest.mark.parametrize(
    "rate,carrier,parent,expected",
    [
        (0.0, {"x-force-trace": "true"}, None, "RECORD_AND_SAMPLE"),
        (0.0, {}, None, "DROP"),
        (0.0, {"x-force-trace": "true"}, (True, False), "DROP"),
        (0.0, {"x-force-trace": "true"}, (True, True), "RECORD_AND_SAMPLE"),
        (0.0, {}, (True, True), "RECORD_AND_SAMPLE"),
        (1.0, {}, None, "RECORD_AND_SAMPLE"),
        (1.0, {}, (True, False), "DROP"),
    ],
)
def test_force_trace_sampling_decisions(fake_otel, rate, carrier, parent, expected):
    telemetry, mocks, patch = fake_otel
    _, sampler, propagator = _installed_sampler_and_propagator(
        telemetry, mocks, patch, rate=rate
    )
    context = propagator.extract(carrier, {}, getter=_DictGetter())
    if parent is not None:
        is_valid, sampled = parent
        context["parent_span"] = SimpleNamespace(is_valid=is_valid, sampled=sampled)

    result = sampler.should_sample(context, TRACE_ID, "GET /x")

    assert result.decision == expected
    if carrier and parent is None and expected == "RECORD_AND_SAMPLE":
        assert result.attributes == {"sampling.forced": True}


def test_force_trace_header_is_sampled_at_the_request_level(fake_otel):
    telemetry, mocks, patch = fake_otel
    decisions = []

    def _instrument(app):
        sampler = mocks["provider_cls"].call_args.kwargs["sampler"]
        propagator = mocks["set_global_textmap"].call_args.args[0]

        class _FakeOtelMiddleware:
            def __init__(self, inner) -> None:
                self.inner = inner

            async def __call__(self, scope, receive, send):
                context = propagator.extract(scope, {}, getter=_ScopeGetter())
                token = mocks["context"].attach(context)
                try:
                    decisions.append(
                        sampler.should_sample(context, TRACE_ID, "GET").decision
                    )
                    await self.inner(scope, receive, send)
                finally:
                    mocks["context"].detach(token)

        app.add_middleware(_FakeOtelMiddleware)

    mocks["fastapi_instrumentor_cls"].instrument_app.side_effect = _instrument
    app, _, _ = _installed_sampler_and_propagator(telemetry, mocks, patch, rate=0.0)
    client = TestClient(app)

    assert client.get("/probe").status_code == 200
    assert client.get("/probe", headers={"X-Force-Trace": "true"}).status_code == 200
    assert client.get("/probe", headers={"X-Force-Trace": "no"}).status_code == 200

    assert decisions == ["DROP", "RECORD_AND_SAMPLE", "DROP"]
    assert mocks["context"].get_current() == {}


def _early_response_app(telemetry, *, serverless: bool = False) -> FastAPI:
    app = _probe_app()
    app.add_middleware(_BodyLimitMiddleware)
    app.add_middleware(app_mod._AuthMiddleware)
    app.add_middleware(
        PathAwareCORSMiddleware,
        match_paths=r"^(?!/build).*",
        allow_origins=["https://app.roboflow.com"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if serverless:
        app.add_middleware(ServerlessAuthMiddleware)
    app.add_middleware(ModelLoadHeadersMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    telemetry.setup_telemetry(app)

    return app


def test_trace_id_header_on_normal_response_with_active_span(fake_otel):
    telemetry, mocks, patch = fake_otel
    mocks["get_current_span"].return_value = _span()
    app = _probe_app()
    telemetry.setup_telemetry(app)

    response = TestClient(app).get("/probe")

    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == format(TRACE_ID, "032x")


def test_trace_id_header_absent_without_active_span(fake_otel):
    telemetry, mocks, patch = fake_otel
    mocks["get_current_span"].return_value = _span(trace_id=0, span_id=0)
    app = _probe_app()
    telemetry.setup_telemetry(app)

    response = TestClient(app).get("/probe")

    assert response.status_code == 200
    assert "x-trace-id" not in response.headers


def test_trace_id_header_on_auth_denial(fake_otel):
    telemetry, mocks, patch = fake_otel
    mocks["get_current_span"].return_value = _span()
    app = _early_response_app(telemetry)

    response = TestClient(app).post("/v2/models/infer")

    assert response.status_code == 401
    assert response.headers["X-Trace-Id"] == format(TRACE_ID, "032x")


def test_trace_id_header_on_body_limit_rejection(fake_otel):
    telemetry, mocks, patch = fake_otel
    mocks["get_current_span"].return_value = _span()
    patch.setattr(configuration, "MAX_BODY_BYTES", 8)
    app = _early_response_app(telemetry)

    response = TestClient(app).post(
        "/infer/object_detection?api_key=k", content=b"x" * 64
    )

    assert response.status_code == 413
    assert response.headers["X-Trace-Id"] == format(TRACE_ID, "032x")


def test_trace_id_header_on_cors_preflight(fake_otel):
    telemetry, mocks, patch = fake_otel
    mocks["get_current_span"].return_value = _span()
    app = _early_response_app(telemetry)

    response = TestClient(app).options(
        "/infer/object_detection",
        headers={
            "Origin": "https://app.roboflow.com",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://app.roboflow.com"
    assert response.headers["X-Trace-Id"] == format(TRACE_ID, "032x")


def test_trace_id_header_on_serverless_denial(fake_otel):
    telemetry, mocks, patch = fake_otel
    mocks["get_current_span"].return_value = _span()
    patch.setattr(serverless_auth, "_cache", {})
    app = _early_response_app(telemetry, serverless=True)

    response = TestClient(app).post("/infer/object_detection", json={})

    assert response.status_code == 401
    assert response.headers["X-Trace-Id"] == format(TRACE_ID, "032x")
    assert float(response.headers["X-Processing-Time"]) >= 0.0


def test_app_adds_telemetry_middlewares_outside_every_other_middleware(fake_otel):
    telemetry, mocks, patch = fake_otel
    patch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    try:
        module = importlib.reload(app_mod)
        classes = [entry.cls for entry in module.app.user_middleware]

        assert classes[:3] == [
            telemetry.TraceIdResponseMiddleware,
            CorrelationIdMiddleware,
            ModelLoadHeadersMiddleware,
        ]
        mocks["fastapi_instrumentor_cls"].instrument_app.assert_called_once_with(
            module.app
        )
    finally:
        patch.undo()
        importlib.reload(telemetry_module)
        importlib.reload(app_mod)
