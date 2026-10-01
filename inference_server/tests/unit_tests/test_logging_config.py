import importlib
import io
import json
import logging
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient
from inference_sdk.config import execution_id
from uvicorn.logging import AccessFormatter, DefaultFormatter

import inference_server.app as app_mod
from inference_server import configuration, logging_config, telemetry
from inference_server.middlewares.correlation_id import (
    CorrelationIdMiddleware,
    correlation_id,
)
from inference_server.middlewares.model_load import (
    ModelLoadHeadersMiddleware,
    record_model_load,
)

OWNED_LOGGERS = (
    logging_config.APPLICATION_LOGGER_NAME,
    logging_config.UVICORN_LOGGER_NAME,
    logging_config.UVICORN_ERROR_LOGGER_NAME,
    logging_config.UVICORN_ACCESS_LOGGER_NAME,
)


@pytest.fixture(autouse=True)
def _restore_logging_state():
    root = logging.getLogger()
    saved = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in OWNED_LOGGERS
    }
    root_handlers = list(root.handlers)
    yield
    for name, (handlers, level, propagate) in saved.items():
        target = logging.getLogger(name)
        target.handlers = handlers
        target.setLevel(level)
        target.propagate = propagate
    root.handlers = root_handlers


def _owned(name: str) -> list:
    return [
        handler
        for handler in logging.getLogger(name).handlers
        if isinstance(handler, logging_config._OwnedStreamHandler)
    ]


def _handlers(name: str) -> list:
    return [
        handler
        for handler in logging.getLogger(name).handlers
        if handler.__class__.__module__ != "_pytest.logging"
    ]


def _capture(name: str = logging_config.APPLICATION_LOGGER_NAME) -> SimpleNamespace:
    handler = _owned(name)[0]
    stream = io.StringIO()
    handler.setStream(stream)

    return SimpleNamespace(handler=handler, stream=stream)


def _lines(captured: SimpleNamespace) -> list:
    return [line for line in captured.stream.getvalue().splitlines() if line]


def _payloads(captured: SimpleNamespace) -> list:
    return [json.loads(line) for line in _lines(captured)]


@pytest.fixture
def structured(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", True)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", True)
    monkeypatch.setattr(configuration, "LOG_LEVEL", "INFO")
    monkeypatch.setattr(configuration, "CORRELATION_ID_LOG_KEY", "request_id")
    logging_config.configure_logging()

    return _capture()


def _uvicorn_access_record(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.getLogger(logging_config.UVICORN_ACCESS_LOGGER_NAME).makeRecord(
        logging_config.UVICORN_ACCESS_LOGGER_NAME,
        logging.INFO,
        "httptools_impl.py",
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:54321", method, path, "1.1", status),
        None,
    )


def _access_app(**route_headers) -> FastAPI:
    app = FastAPI()
    cold_start = route_headers.pop("cold_start", False)

    @app.api_route("/{full_path:path}", methods=["GET", "POST"])
    async def _probe() -> Response:
        if cold_start:
            record_model_load("ds/1", cold_start=True, load_time_s=1.5)
        return Response(content=b"ok", headers=route_headers)

    app.add_middleware(ModelLoadHeadersMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    logging_config.install_structured_access_log(app)

    return app


def _record(logger_name: str, level: int, message: str, exc_info=None):
    logger = logging.getLogger(logger_name)
    record = logger.makeRecord(
        logger.name, level, "gateway.py", 10, message, ("m",), exc_info
    )
    return record


def test_default_configuration_keeps_the_plain_uvicorn_access_line(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", False)
    monkeypatch.setattr(configuration, "LOG_LEVEL", "WARNING")

    logging_config.configure_logging()

    access_logger = logging.getLogger(logging_config.UVICORN_ACCESS_LOGGER_NAME)
    access_handlers = _owned("uvicorn.access")
    assert len(access_handlers) == 1
    assert isinstance(access_handlers[0].formatter, AccessFormatter)
    assert access_logger.level == logging.INFO
    assert access_logger.propagate is False
    line = access_handlers[0].format(
        _uvicorn_access_record("POST", "/v2/models/infer", 200)
    )
    assert line.endswith('127.0.0.1:54321 - "POST /v2/models/infer HTTP/1.1" 200 OK')
    assert '"POST' in line
    health_line = access_handlers[0].format(
        _uvicorn_access_record("GET", "/healthz", 200)
    )
    assert health_line.startswith("INFO")
    assert logging.getLogger("inference_server").level == logging.WARNING
    application_formatter = _owned("inference_server")[0].formatter
    assert isinstance(application_formatter, logging.Formatter)
    assert not isinstance(application_formatter, logging_config.JsonFormatter)


def test_plain_mode_adds_no_access_middleware(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", False)
    app = FastAPI()

    logging_config.install_structured_access_log(app)

    assert app.user_middleware == []


def test_api_logging_alone_does_not_replace_the_access_log(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", True)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", False)
    app = FastAPI()

    logging_config.configure_logging()
    logging_config.install_structured_access_log(app)

    assert isinstance(_owned("uvicorn.access")[0].formatter, AccessFormatter)
    assert app.user_middleware == []
    assert isinstance(
        _owned("inference_server")[0].formatter, logging_config.JsonFormatter
    )


def test_configure_logging_replaces_only_its_own_handlers(structured):
    foreign = logging.StreamHandler(io.StringIO())
    application_logger = logging.getLogger("inference_server")
    application_logger.addHandler(foreign)
    uvicorn_foreign = logging.StreamHandler(io.StringIO())
    logging.getLogger("uvicorn").addHandler(uvicorn_foreign)
    first_owned = _owned("inference_server")[0]
    first_uvicorn_owned = _owned("uvicorn")[0]

    logging_config.configure_logging()

    assert foreign in application_logger.handlers
    assert uvicorn_foreign in logging.getLogger("uvicorn").handlers
    assert len(_owned("inference_server")) == 1
    assert _owned("inference_server")[0] is not first_owned
    assert first_owned not in application_logger.handlers
    assert len(_owned("uvicorn")) == 1
    assert first_uvicorn_owned not in logging.getLogger("uvicorn").handlers
    assert _owned("uvicorn.error") == []
    assert _handlers("uvicorn.access") == []


def test_handlers_survive_the_entry_point_uvicorn_configuration(structured):
    application_handler = structured.handler
    uvicorn_handler = _owned("uvicorn")[0]

    uvicorn.Config("inference_server.app:app", log_config=None).configure_logging()

    assert _owned("inference_server") == [application_handler]
    assert _owned("uvicorn") == [uvicorn_handler]
    assert _handlers("uvicorn.access") == []


def test_uvicorn_default_configuration_would_replace_the_handlers(structured):
    uvicorn_handler = _owned("uvicorn")[0]

    uvicorn.Config("inference_server.app:app").configure_logging()

    assert uvicorn_handler not in logging.getLogger("uvicorn").handlers
    assert _handlers("uvicorn.access") != []


def test_entry_point_runs_uvicorn_without_its_logging_config(monkeypatch):
    run = MagicMock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setenv(configuration.PORT_ENV, "9123")

    app_mod.main()

    run.assert_called_once_with(
        "inference_server.app:app",
        host="0.0.0.0",
        port=9123,
        workers=configuration.NUM_WORKERS,
        log_config=None,
    )


def test_log_level_applies_to_the_application_logger_only(monkeypatch):
    monkeypatch.setattr(configuration, "LOG_LEVEL", "INFO")

    logging_config.configure_logging()

    assert logging.getLogger("inference_server").isEnabledFor(logging.INFO)
    assert logging.getLogger("inference_server.gateway").isEnabledFor(logging.INFO)
    assert not logging.getLogger("inference_server").isEnabledFor(logging.DEBUG)
    assert logging.getLogger("uvicorn.error").isEnabledFor(logging.INFO)

    monkeypatch.setattr(configuration, "LOG_LEVEL", "ERROR")
    logging_config.configure_logging()

    assert not logging.getLogger("inference_server").isEnabledFor(logging.WARNING)
    assert logging.getLogger("uvicorn.error").isEnabledFor(logging.INFO)
    assert logging.getLogger("uvicorn.access").isEnabledFor(logging.INFO)


@pytest.mark.parametrize("api_logging_enabled", [False, True])
def test_uvicorn_loggers_keep_uvicorn_defaults(monkeypatch, api_logging_enabled):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", api_logging_enabled)
    monkeypatch.setattr(configuration, "LOG_LEVEL", "WARNING")

    logging_config.configure_logging()

    uvicorn_logger = logging.getLogger("uvicorn")
    error_logger = logging.getLogger("uvicorn.error")
    assert uvicorn_logger.level == logging.INFO
    assert uvicorn_logger.propagate is False
    assert error_logger.level == logging.INFO
    assert error_logger.propagate is True
    assert _owned("uvicorn.error") == []
    captured = _capture("uvicorn")
    assert isinstance(captured.handler.formatter, DefaultFormatter)

    error_logger.info("Started server process")

    assert _lines(captured) == ["INFO:     Started server process"]
    assert captured.handler not in logging.getLogger("inference_server").handlers


def test_application_records_do_not_propagate_to_the_root_logger(structured):
    root_stream = io.StringIO()
    root_handler = logging.StreamHandler(root_stream)
    logging.getLogger().addHandler(root_handler)

    logging.getLogger("inference_server.gateway").warning("Preload of %s failed", "m")

    assert logging.getLogger("inference_server").propagate is False
    assert root_stream.getvalue() == ""
    payloads = _payloads(structured)
    assert len(payloads) == 1
    assert payloads[0]["message"] == "Preload of m failed"


def test_unknown_log_level_falls_back_to_warning(monkeypatch):
    monkeypatch.setattr(configuration, "LOG_LEVEL", "LOUD")

    logging_config.configure_logging()

    assert logging.getLogger("inference_server").level == logging.WARNING


def test_json_application_log_carries_context_and_callsite_fields(
    structured, monkeypatch
):
    monkeypatch.setattr(
        telemetry,
        "trace_context_fields",
        lambda: {"trace_id": "t" * 32, "span_id": "s" * 16},
    )
    id_token = correlation_id.set("abc-123")
    exec_token = execution_id.set("exec-7")
    try:
        logging.getLogger("inference_server.gateway").warning("load %s", "m")
    finally:
        execution_id.reset(exec_token)
        correlation_id.reset(id_token)

    payload = _payloads(structured)[0]

    assert payload["message"] == "load m"
    assert payload["severity"] == "WARNING"
    assert payload["logger"] == "inference_server.gateway"
    assert payload["request_id"] == "abc-123"
    assert payload["execution_id"] == "exec-7"
    assert payload["trace_id"] == "t" * 32
    assert payload["span_id"] == "s" * 16
    assert payload["filename"] == "test_logging_config.py"
    assert (
        payload["func_name"]
        == "test_json_application_log_carries_context_and_callsite_fields"
    )
    assert isinstance(payload["lineno"], int)
    assert "timestamp" in payload


def test_json_application_log_uses_the_configured_correlation_key(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", True)
    monkeypatch.setattr(configuration, "CORRELATION_ID_LOG_KEY", "correlation")
    logging_config.configure_logging()
    captured = _capture()
    token = correlation_id.set("abc-123")
    try:
        logging.getLogger("inference_server.x").error("m")
    finally:
        correlation_id.reset(token)

    payload = _payloads(captured)[0]

    assert payload["correlation"] == "abc-123"
    assert "request_id" not in payload


def test_json_application_log_omits_absent_context_and_formats_exceptions(
    structured,
):
    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("inference_server.x").exception("failed")

    payload = _payloads(structured)[0]

    assert "request_id" not in payload
    assert "execution_id" not in payload
    assert "trace_id" not in payload
    assert payload["severity"] == "ERROR"
    exception = payload["exception"]
    assert exception["type"] == "ValueError"
    assert exception["message"] == "boom"
    frame = exception["stacktrace"][-1]
    assert frame["filename"].endswith("test_logging_config.py")
    assert (
        frame["function"]
        == "test_json_application_log_omits_absent_context_and_formats_exceptions"
    )
    assert frame["code"] == 'raise ValueError("boom")'
    assert isinstance(frame["lineno"], int)


def test_structured_access_log_emits_one_json_object_with_legacy_fields(structured):
    app = _access_app(
        **{
            "X-Processing-Time": "0.25",
            "X-Workflow-Id": "wf-1",
            "X-Workspace-Id": "ws-1",
            "X-Trace-Id": "t" * 32,
            "execution_id": "exec-7",
            "cold_start": True,
        }
    )

    response = TestClient(app).post("/v2/models/infer?api_key=secret-key", json={})

    assert response.status_code == 200
    lines = _lines(structured)
    assert len(lines) == 1
    assert "secret-key" not in lines[0]
    payload = json.loads(lines[0])
    assert (
        payload["message"] == 'testclient:50000 - "POST /v2/models/infer HTTP/1.1" 200'
    )
    assert '"POST' in payload["message"]
    assert payload["method"] == "POST"
    assert payload["path"] == "/v2/models/infer"
    assert payload["status"] == 200
    assert payload["status_code"] == 200
    assert payload["duration_ms"] >= 0
    assert payload["request_id"] == response.headers["X-Request-ID"]
    assert payload["severity"] == "INFO"
    assert payload["logger"] == logging_config.ACCESS_LOGGER_NAME


def test_structured_access_log_takes_fields_from_response_headers(structured):
    app = _access_app(
        **{
            "X-Processing-Time": "0.25",
            "X-Workflow-Id": "wf-1",
            "X-Workspace-Id": "ws-1",
            "X-Trace-Id": "t" * 32,
            "execution_id": "exec-7",
            "cold_start": True,
        }
    )

    response = TestClient(app).get("/workflows/run")

    assert response.status_code == 200
    payload = _payloads(structured)[0]
    assert payload["processing_time"] == "0.25"
    assert payload["model_cold_start"] == "true"
    assert payload["model_cold_start_count"] == "1"
    assert payload["model_load_time"] == "1.5"
    assert payload["model_id"] == "ds/1"
    assert payload["workflow_id"] == "wf-1"
    assert payload["workspace_id"] == "ws-1"
    assert payload["trace_id"] == "t" * 32
    assert payload["execution_id"] == "exec-7"
    assert payload["request_id"] == response.headers["X-Request-ID"]


def test_structured_access_log_omits_absent_optional_fields(structured):
    app = _access_app()

    TestClient(app).get("/v2/models")

    payload = _payloads(structured)[0]
    assert payload["model_cold_start"] == "false"
    assert payload["model_cold_start_count"] == "0"
    for absent in (
        "processing_time",
        "model_load_time",
        "model_id",
        "workflow_id",
        "workspace_id",
        "trace_id",
        "execution_id",
    ):
        assert absent not in payload


def test_structured_access_log_survives_a_reserved_correlation_key(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", True)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", True)
    monkeypatch.setattr(configuration, "LOG_LEVEL", "INFO")
    monkeypatch.setattr(configuration, "CORRELATION_ID_LOG_KEY", "message")
    logging_config.configure_logging()
    captured = _capture()
    app = _access_app()

    response = TestClient(app).get("/v2/models")

    assert response.status_code == 200
    assert response.content == b"ok"
    payload = _payloads(captured)[0]
    assert payload["message"] == 'testclient:50000 - "GET /v2/models HTTP/1.1" 200'
    assert payload["status"] == 200
    assert not hasattr(
        logging.makeLogRecord({}), logging_config.STRUCTURED_FIELDS_ATTRIBUTE
    )


def test_structured_access_log_honours_log_level_for_ordinary_paths(
    structured, monkeypatch
):
    monkeypatch.setattr(configuration, "LOG_LEVEL", "ERROR")
    logging_config.configure_logging()
    captured = _capture()
    app = _access_app()

    TestClient(app).get("/v2/models")

    assert _lines(captured) == []
    assert not logging.getLogger(logging_config.ACCESS_LOGGER_NAME).isEnabledFor(
        logging.INFO
    )


@pytest.mark.parametrize("path", sorted(logging_config.HEALTH_LOG_PATHS))
def test_structured_access_log_demotes_health_paths_to_debug(
    structured, monkeypatch, path
):
    app = _access_app()

    TestClient(app).get(path)
    assert _lines(structured) == []

    monkeypatch.setattr(configuration, "LOG_LEVEL", "DEBUG")
    logging_config.configure_logging()
    captured = _capture()
    TestClient(app).get(path)

    payloads = _payloads(captured)
    assert len(payloads) == 1
    assert payloads[0]["severity"] == "DEBUG"
    assert payloads[0]["path"] == path


def test_structured_access_log_measures_the_whole_stack(structured):
    app = FastAPI()

    @app.get("/slow")
    async def _slow() -> Response:
        time.sleep(0.02)
        return Response(content=b"ok")

    logging_config.install_structured_access_log(app)

    TestClient(app).get("/slow")

    assert _payloads(structured)[0]["duration_ms"] >= 15


def test_app_adds_the_access_log_middleware_outside_every_other_middleware(
    structured, monkeypatch
):
    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    try:
        module = importlib.reload(app_mod)
        classes = [entry.cls for entry in module.app.user_middleware]

        assert classes[:3] == [
            logging_config.StructuredAccessLogMiddleware,
            CorrelationIdMiddleware,
            ModelLoadHeadersMiddleware,
        ]
        assert _handlers("uvicorn.access") == []
    finally:
        monkeypatch.undo()
        importlib.reload(app_mod)


def test_plain_application_log_writes_the_plain_line(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    monkeypatch.setattr(configuration, "LOG_LEVEL", "WARNING")
    logging_config.configure_logging()
    captured = _capture()

    logging.getLogger("inference_server.gateway").warning("Preload of %s failed", "m")

    lines = _lines(captured)
    assert len(lines) == 1
    assert lines[0].endswith(" WARNING inference_server.gateway: Preload of m failed")
    assert sys.stderr is not captured.stream
