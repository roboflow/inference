"""uvicorn's access log must not carry API keys from the query string.

The server is run with real uvicorn so the access line comes from uvicorn
itself, with the default logging configuration (``API_LOGGING_ENABLED`` and
``STRUCTURED_API_LOGGING`` unset).
"""

import io
import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import pytest
import requests
import uvicorn
from fastapi import FastAPI, Response

from inference_server import configuration, logging_config
from inference_server.ops import memory_logs

API_KEY = "TESTKEY123"
SERVICE_SECRET = "SERVICESECRET987"
SERVER_START_TIMEOUT_S = 10.0
WATCHED_LOGGERS = (
    None,
    logging_config.APPLICATION_LOGGER_NAME,
    logging_config.UVICORN_LOGGER_NAME,
    logging_config.UVICORN_ERROR_LOGGER_NAME,
    logging_config.UVICORN_ACCESS_LOGGER_NAME,
)


@pytest.fixture(autouse=True)
def _isolated_logging(monkeypatch):
    monkeypatch.setattr(configuration, "API_LOGGING_ENABLED", False)
    monkeypatch.setattr(configuration, "STRUCTURED_API_LOGGING", False)
    monkeypatch.setattr(configuration, "LOG_LEVEL", "INFO")
    saved = {
        name: (
            list(logging.getLogger(name).handlers),
            list(logging.getLogger(name).filters),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in WATCHED_LOGGERS
    }
    memory_logs._log_entries.clear()
    yield
    memory_logs._log_entries.clear()
    for name, (handlers, filters, level, propagate) in saved.items():
        target = logging.getLogger(name)
        target.handlers = handlers
        target.filters = filters
        target.setLevel(level)
        target.propagate = propagate


def _access_stream() -> io.StringIO:
    access_logger = logging.getLogger(logging_config.UVICORN_ACCESS_LOGGER_NAME)
    owned = [
        handler
        for handler in access_logger.handlers
        if isinstance(handler, logging_config._OwnedStreamHandler)
    ]
    stream = io.StringIO()
    owned[0].setStream(stream)

    return stream


def _probe_app() -> FastAPI:
    app = FastAPI()

    @app.get("/probe")
    async def _probe() -> Response:
        return Response(content=b"ok")

    return app


@contextmanager
def _serve(app: FastAPI) -> Iterator[str]:
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_config=None, lifespan="off"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + SERVER_START_TIMEOUT_S
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.01)

    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=SERVER_START_TIMEOUT_S)


@pytest.mark.parametrize(
    "path",
    [
        f"/probe?api_key={API_KEY}",
        f"/probe?foo=bar&api_key={API_KEY}",
        f"/probe?api%5Fkey={API_KEY}",
        f"/probe?service_secret={SERVICE_SECRET}",
    ],
)
def test_redact_path_with_query_string_removes_credentials(path):
    redacted = logging_config.redact_path_with_query_string(path)

    assert API_KEY not in redacted
    assert SERVICE_SECRET not in redacted
    assert redacted.startswith("/probe?")


def test_redact_path_with_query_string_keeps_other_parameters_encoded():
    path = f"/model/1?confidence=0.5&name=a%20b&api_key={API_KEY}"

    redacted = logging_config.redact_path_with_query_string(path)

    assert redacted == "/model/1?confidence=0.5&name=a%20b&api_key=***"


def test_redact_path_with_query_string_strips_paths_of_embedded_urls():
    path = "/model/1?image=https%3A%2F%2Fexample.com%2Fsigned%3Ftoken%3Dabc"

    redacted = logging_config.redact_path_with_query_string(path)

    assert redacted == "/model/1?image=https://example.com/***"


def test_default_uvicorn_access_log_carries_no_api_key():
    logging_config.configure_logging()
    stream = _access_stream()

    with _serve(_probe_app()) as base_url:
        requests.get(f"{base_url}/probe?api_key={API_KEY}", timeout=5)
        requests.get(f"{base_url}/probe?api%5Fkey={API_KEY}", timeout=5)

    lines = stream.getvalue()
    assert "/probe?api_key=***" in lines
    assert API_KEY not in lines


def test_memory_logs_hold_no_api_key_from_the_access_log(monkeypatch):
    monkeypatch.setattr(configuration, "ENABLE_IN_MEMORY_LOGS", True)
    logging_config.configure_logging()
    memory_logs.setup_memory_logging()
    _access_stream()

    with _serve(_probe_app()) as base_url:
        requests.get(f"{base_url}/probe?api_key={API_KEY}", timeout=5)

    messages = [entry["message"] for entry in memory_logs.get_recent_logs(0)]
    assert any("/probe?api_key=" in message for message in messages)
    assert all(API_KEY not in message for message in messages)


def test_redaction_filter_is_installed_once():
    logging_config.configure_logging()
    logging_config.configure_logging()

    access_logger = logging.getLogger(logging_config.UVICORN_ACCESS_LOGGER_NAME)
    installed = [
        existing
        for existing in access_logger.filters
        if isinstance(existing, logging_config.AccessLogRedactionFilter)
    ]
    assert len(installed) == 1
