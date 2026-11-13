import json
import os
import subprocess
import sys
import textwrap

import pytest
import requests
import requests_mock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

import inference.core.telemetry as telemetry

PRELOAD_KEY = "synthetic-preload-span-key"
SERVICE_SECRET = "synthetic-service-span-secret"
CALLER_KEY = "synthetic-caller-span-key"
CREDENTIALS = (PRELOAD_KEY, SERVICE_SECRET, CALLER_KEY)
PLATFORM_URL = "https://api.roboflow.com/demo-workspace/demo-model/1"
MASKED_PRELOAD_KEY = "sy***ey"


@pytest.fixture
def configured_credentials(monkeypatch):
    monkeypatch.setattr(telemetry, "PRELOAD_API_KEY", PRELOAD_KEY)
    monkeypatch.setattr(telemetry, "ROBOFLOW_INTERNAL_SERVICE_SECRET", SERVICE_SECRET)


@pytest.fixture
def provider_and_exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        yield provider, exporter
    finally:
        provider.shutdown()


def _mocked_session() -> requests.Session:
    # A mounted adapter leaves Session.send, which the instrumentation wraps, alone.
    adapter = requests_mock.Adapter()
    adapter.register_uri("GET", PLATFORM_URL, json={})
    session = requests.Session()
    session.mount("https://", adapter)

    return session


def _assert_no_credentials(text: str) -> None:
    for credential in CREDENTIALS:
        assert credential not in text


def test_requests_hook_redacts_outbound_platform_url(
    configured_credentials, provider_and_exporter
):
    provider, exporter = provider_and_exporter
    instrumentor = RequestsInstrumentor()
    instrumentor.instrument(
        tracer_provider=provider, request_hook=telemetry._requests_request_hook
    )
    try:
        session = _mocked_session()
        session.get(
            PLATFORM_URL,
            params={"api_key": PRELOAD_KEY, "service_secret": SERVICE_SECRET},
            timeout=1,
        )
    finally:
        instrumentor.uninstrument()

    (span,) = exporter.get_finished_spans()
    url = span.attributes.get("http.url") or span.attributes.get("url.full")
    assert url == f"{PLATFORM_URL}?api_key={MASKED_PRELOAD_KEY}&service_secret=***"
    _assert_no_credentials(span.to_json())


def test_server_hook_redacts_inbound_query(
    configured_credentials, provider_and_exporter
):
    provider, exporter = provider_and_exporter
    app = FastAPI()

    @app.get("/model/{name}")
    def route(name: str) -> dict:
        return {"name": name}

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=provider,
        server_request_hook=telemetry._server_request_hook,
    )
    try:
        response = TestClient(app).get(f"/model/demo?api_key={CALLER_KEY}&x=1")
    finally:
        FastAPIInstrumentor.uninstrument_app(app)

    assert response.status_code == 200
    (server_span,) = [
        span for span in exporter.get_finished_spans() if span.kind.name == "SERVER"
    ]
    assert server_span.attributes["http.url"].endswith(
        "/model/demo?api_key=sy***ey&x=1"
    )
    _assert_no_credentials(server_span.to_json())


def test_redacting_exporter_masks_every_string_on_the_span(configured_credentials):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(telemetry.RedactingSpanExporter(exporter))
    )
    tracer = provider.get_tracer(__name__)

    with tracer.start_as_current_span(f"lookup {PRELOAD_KEY}") as span:
        span.set_attribute("custom.note", f"used {PRELOAD_KEY}")
        span.set_attribute("custom.urls", (f"{PLATFORM_URL}?api_key={CALLER_KEY}",))
        span.set_attribute("custom.count", 3)
        span.add_event("retry", {"detail": f"service_secret={SERVICE_SECRET}"})
        span.set_status(Status(StatusCode.ERROR, f"denied {SERVICE_SECRET}"))
    provider.shutdown()

    (exported,) = exporter.get_finished_spans()
    _assert_no_credentials(exported.to_json())
    assert exported.name == "lookup [REDACTED]"
    assert exported.attributes["custom.note"] == "used [REDACTED]"
    assert exported.attributes["custom.urls"] == (f"{PLATFORM_URL}?api_key=sy***ey",)
    assert exported.attributes["custom.count"] == 3
    assert exported.events[0].attributes["detail"] == "service_secret=***"
    assert exported.status.status_code == StatusCode.ERROR
    assert exported.status.description == "denied [REDACTED]"
    assert exported.context == span.get_span_context()


def test_redacting_exporter_passes_clean_spans_through(configured_credentials):
    exporter = InMemorySpanExporter()
    redacting = telemetry.RedactingSpanExporter(exporter)
    tracer = TracerProvider().get_tracer(__name__)
    span = tracer.start_span("model.load", attributes={"model.id": "a/1"})
    span.end()

    redacting.export([span])

    (exported,) = exporter.get_finished_spans()
    assert exported is span


@pytest.mark.parametrize("semconv", ["", "http/dup"], ids=["default", "http-dup"])
def test_setup_telemetry_keeps_credentials_out_of_exported_spans(semconv):
    script = textwrap.dedent(f"""
        import json

        import opentelemetry.exporter.otlp.proto.grpc.trace_exporter as grpc_exporter
        import requests
        import requests_mock
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        import inference.core.telemetry as telemetry

        memory = InMemorySpanExporter()
        grpc_exporter.OTLPSpanExporter = lambda **kwargs: memory
        params = {{"api_key": {PRELOAD_KEY!r}, "service_secret": {SERVICE_SECRET!r}}}
        adapter = requests_mock.Adapter()
        adapter.register_uri("GET", {PLATFORM_URL!r}, json={{}})
        failing = requests.Request("GET", {PLATFORM_URL!r} + "/fail", params=params)
        adapter.register_uri(
            "GET",
            {PLATFORM_URL!r} + "/fail",
            exc=requests.ConnectionError("cannot reach " + failing.prepare().url),
        )
        session = requests.Session()
        session.mount("https://", adapter)

        app = FastAPI()

        @app.get("/model/{{name}}")
        def route(name: str) -> dict:
            session.get({PLATFORM_URL!r}, params=params, timeout=1)
            try:
                session.get({PLATFORM_URL!r} + "/fail", params=params, timeout=1)
            except requests.ConnectionError:
                pass
            return {{}}

        telemetry.setup_telemetry(app)
        status = TestClient(app).get("/model/demo?api_key={CALLER_KEY}").status_code
        telemetry.shutdown_telemetry()
        print(json.dumps({{
            "status": status,
            "spans": [json.loads(span.to_json()) for span in memory.get_finished_spans()],
        }}))
        """)

    result = subprocess.run(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "OFFLINE_MODE": "false",
            "OTEL_TRACING_ENABLED": "true",
            "OTEL_METRICS_ENABLED": "false",
            "OTEL_EXPORTER_PROTOCOL": "grpc",
            "OTEL_SAMPLING_RATE": "1.0",
            "OTEL_SEMCONV_STABILITY_OPT_IN": semconv,
            "PRELOAD_API_KEY": PRELOAD_KEY,
            "ROBOFLOW_INTERNAL_SERVICE_SECRET": SERVICE_SECRET,
        },
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout.strip().splitlines()[-1]
    payload = json.loads(output)
    assert payload["status"] == 200
    kinds = [span["kind"] for span in payload["spans"]]
    assert kinds.count("SpanKind.CLIENT") == 2
    assert "SpanKind.SERVER" in kinds
    failed = [
        span
        for span in payload["spans"]
        if span["kind"] == "SpanKind.CLIENT"
        and span["status"]["status_code"] == "ERROR"
    ]
    assert failed
    assert "cannot reach" in failed[0]["status"]["description"]
    urls = [
        span["attributes"].get("url.full") or span["attributes"].get("http.url")
        for span in payload["spans"]
        if span["kind"] == "SpanKind.CLIENT"
    ]
    # The exporter masks the hook's "sy***ey" once more, as it is under 8 characters.
    assert f"{PLATFORM_URL}?api_key=***&service_secret=***" in urls
    _assert_no_credentials(output)
