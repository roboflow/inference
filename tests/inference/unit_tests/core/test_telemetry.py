from unittest.mock import MagicMock


def test_start_span_does_not_use_global_otel_provider_in_offline_mode(
    monkeypatch,
) -> None:
    import inference.core.telemetry as telemetry

    get_tracer_mock = MagicMock()
    monkeypatch.setattr(telemetry, "OFFLINE_MODE", True)
    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_get_tracer", get_tracer_mock)

    with telemetry.start_span("offline-operation") as span:
        assert span is None

    get_tracer_mock.assert_not_called()


def test_setup_telemetry_does_not_create_exporters_in_offline_mode(
    monkeypatch,
) -> None:
    import inference.core.telemetry as telemetry

    app = MagicMock()
    monkeypatch.setattr(telemetry, "OFFLINE_MODE", True)

    telemetry.setup_telemetry(app)

    app.add_middleware.assert_not_called()


def test_error_trace_redacts_preload_credentials_and_exception_chain(monkeypatch):
    import inference.core.telemetry as telemetry

    key = "synthetic-preload-trace-credential"
    service_secret = "synthetic-service-trace-credential"
    span = MagicMock()
    monkeypatch.setattr(telemetry, "_otel_enabled", lambda: True)
    monkeypatch.setattr(telemetry, "PRELOAD_API_KEY", key)
    monkeypatch.setattr(telemetry, "ROBOFLOW_INTERNAL_SERVICE_SECRET", service_secret)
    monkeypatch.setattr(telemetry.trace, "get_current_span", lambda: span)
    try:
        try:
            raise ConnectionError(f"request failed: api_key={key}")
        except ConnectionError as cause:
            raise RuntimeError(
                f"sam3/sam3_final: upstream failure {service_secret}"
            ) from cause
    except RuntimeError as error:
        telemetry.record_error(error)

    exception = span.record_exception.call_args.args[0]
    attributes = span.record_exception.call_args.kwargs["attributes"]
    status = span.set_status.call_args.args[1]
    for value in (str(exception), str(attributes), status):
        assert key not in value
        assert service_secret not in value
    assert attributes["exception.type"] == "RuntimeError"
    assert "sam3/sam3_final: upstream failure" in attributes["exception.message"]
    assert "ConnectionError" in attributes["exception.stacktrace"]
    assert "[REDACTED]" in attributes["exception.stacktrace"]


def test_span_errors_use_filtered_events_and_preserve_the_exception(monkeypatch):
    import pytest

    import inference.core.telemetry as telemetry

    tracer = MagicMock()
    span = tracer.start_as_current_span.return_value.__enter__.return_value
    monkeypatch.setattr(telemetry, "_otel_enabled", lambda: True)
    monkeypatch.setattr(telemetry, "_get_tracer", lambda: tracer)
    monkeypatch.setattr(telemetry.trace, "get_current_span", lambda: span)
    key = "synthetic-preload-span-credential"
    monkeypatch.setattr(telemetry, "PRELOAD_API_KEY", key)
    error = RuntimeError(f"sam3/sam3_final: denied {key}")

    with pytest.raises(RuntimeError) as raised:
        with telemetry.start_span("model.load", {"model.id": "sam3/sam3_final"}):
            raise error

    assert raised.value is error
    tracer.start_as_current_span.assert_called_once_with(
        "model.load",
        attributes={"model.id": "sam3/sam3_final"},
        record_exception=False,
        set_status_on_exception=False,
    )
    assert key not in str(span.record_exception.call_args)
    assert key not in str(span.set_status.call_args)


def test_exported_span_does_not_include_preload_key_in_exception_chain(monkeypatch):
    import json

    import pytest
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import inference.core.telemetry as telemetry

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_otel_enabled", lambda: True)
    monkeypatch.setattr(
        telemetry, "_get_tracer", lambda: provider.get_tracer("preload-test")
    )
    key = "synthetic-exported-preload-credential"
    monkeypatch.setattr(telemetry, "PRELOAD_API_KEY", key)
    try:
        with pytest.raises(RuntimeError):
            with telemetry.start_span("model.load"):
                try:
                    raise ConnectionError(f"request URL includes api_key={key}")
                except ConnectionError as cause:
                    raise RuntimeError("sam3/sam3_final: connection failed") from cause

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        payload = json.dumps(
            [
                {
                    "status": span.status.description,
                    "events": [dict(event.attributes) for event in span.events],
                }
                for span in spans
            ]
        )
        assert key not in payload
        assert "sam3/sam3_final: connection failed" in payload
        assert "ConnectionError" in payload
        assert "[REDACTED]" in payload
    finally:
        provider.shutdown()
