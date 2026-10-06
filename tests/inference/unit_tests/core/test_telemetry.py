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
