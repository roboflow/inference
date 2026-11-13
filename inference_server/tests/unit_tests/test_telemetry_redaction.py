import json
from functools import partial
from urllib.parse import quote, quote_plus

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from inference_server import configuration, platform_http, telemetry
from inference_server.legacy.errors import with_legacy_errors


@pytest.fixture
def exported_spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)
    monkeypatch.setattr(configuration, "OFFLINE_MODE", False)
    monkeypatch.setattr(telemetry, "_tracer", tracer)
    try:
        yield tracer, exporter
    finally:
        provider.shutdown()


@pytest.mark.parametrize("automatic", [False, True], ids=["record_error", "start_span"])
@pytest.mark.parametrize(
    "encode",
    [
        str,
        partial(quote, safe=""),
        partial(quote_plus, safe=""),
        lambda s: json.dumps(s)[1:-1],
    ],
    ids=["plain", "url", "query", "json"],
)
def test_exported_errors_redact_configured_credentials(
    monkeypatch, exported_spans, automatic, encode
):
    tracer, exporter = exported_spans
    api_key = 'preload key+/"\\'
    service_secret = 'service secret+/"\\'
    monkeypatch.setattr(configuration, "PRELOAD_API_KEY", api_key)
    monkeypatch.setattr(
        configuration, "ROBOFLOW_INTERNAL_SERVICE_SECRET", service_secret
    )
    message = f"model demo/1 failed: {encode(api_key)} {encode(service_secret)}"
    cause = ValueError(f"upstream rejected {encode(service_secret)} {encode(api_key)}")
    error = RuntimeError(message)
    scope = (
        telemetry.start_span
        if automatic
        else partial(
            tracer.start_as_current_span,
            record_exception=False,
            set_status_on_exception=False,
        )
    )

    with pytest.raises(RuntimeError) as caught:
        with scope("model.load", attributes={"model.id": "demo/1"}):
            try:
                raise cause
            except ValueError:
                try:
                    raise error
                except RuntimeError as recorded:
                    if not automatic:
                        telemetry.record_error(recorded)
                    raise

    assert caught.value is error
    assert str(error) == message
    assert error.__context__ is cause
    (span,) = exporter.get_finished_spans()
    assert span.attributes["model.id"] == "demo/1"
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "model demo/1 failed: [REDACTED] [REDACTED]"
    (event,) = span.events
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "RuntimeError"
    assert event.attributes["exception.message"] == span.status.description
    stacktrace = event.attributes["exception.stacktrace"]
    assert "ValueError: upstream rejected [REDACTED] [REDACTED]" in stacktrace
    assert "RuntimeError: model demo/1 failed: [REDACTED] [REDACTED]" in stacktrace
    assert "test_exported_errors_redact_configured_credentials" in stacktrace
    for credential in (api_key, service_secret):
        assert credential not in stacktrace
        assert encode(credential) not in stacktrace


@pytest.mark.asyncio
async def test_invalid_platform_url_does_not_export_configured_key(
    monkeypatch, exported_spans
):
    tracer, exporter = exported_spans
    api_key = "synthetic-preload-key"
    monkeypatch.setattr(configuration, "PRELOAD_API_KEY", api_key)
    monkeypatch.setattr(configuration, "ROBOFLOW_INTERNAL_SERVICE_SECRET", None)

    @with_legacy_errors
    async def route():
        return platform_http._platform_request("get", "bad-host/?api_key=" + api_key)

    with tracer.start_as_current_span("route"):
        response = await route()

    assert response.status_code == 503
    assert json.loads(response.body) == {
        "message": "Internal error. Could not connect to Roboflow API."
    }
    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert span.status.status_code == StatusCode.ERROR
    assert event.attributes["exception.type"] == (
        "inference_server.legacy.errors.LegacyHTTPError"
    )
    assert "MissingSchema" in event.attributes["exception.stacktrace"]
    assert "api_key=[REDACTED]" in event.attributes["exception.stacktrace"]
    assert api_key not in span.to_json()


def test_unconfigured_credentials_preserve_error_diagnostics(
    monkeypatch, exported_spans
):
    tracer, exporter = exported_spans
    monkeypatch.setattr(configuration, "PRELOAD_API_KEY", None)
    monkeypatch.setattr(configuration, "ROBOFLOW_INTERNAL_SERVICE_SECRET", "")

    with tracer.start_as_current_span("model.load"):
        telemetry.record_error(ValueError("model demo/1 unavailable"))

    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert span.status.description == "model demo/1 unavailable"
    assert event.attributes["exception.type"] == "ValueError"
    assert event.attributes["exception.message"] == "model demo/1 unavailable"
    assert event.attributes["exception.stacktrace"] == (
        "ValueError: model demo/1 unavailable\n"
    )
