"""uvicorn's access log must not carry API keys from the query string.

The server is run with real uvicorn so the access line comes from uvicorn
itself, with the default logging configuration (``STRUCTURED_API_LOGGING``
unset), and the in-memory ``/logs`` buffer is read back through the route.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
import uvicorn

from inference.core.logging import access_log_redaction, memory_handler
from inference.core.logging.access_log_redaction import (
    AccessLogRedactionFilter,
    redact_path_with_query_string,
)

API_KEY = "TESTKEY123"
SERVICE_SECRET = "SERVICESECRET987"
SERVER_START_TIMEOUT_S = 10.0


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app

    def set_stream_manager_client(self, stream_manager_client) -> None:
        pass


@pytest.fixture
def access_logger() -> Iterator[logging.Logger]:
    target = logging.getLogger(access_log_redaction.UVICORN_ACCESS_LOGGER_NAME)
    saved = (list(target.handlers), list(target.filters), target.level)
    target.filters = [
        existing
        for existing in target.filters
        if not isinstance(existing, AccessLogRedactionFilter)
    ]
    target.setLevel(logging.INFO)
    yield target
    target.handlers, target.filters, level = saved
    target.setLevel(level)


@pytest.fixture
def in_memory_logs(monkeypatch) -> Iterator[None]:
    monkeypatch.setattr(memory_handler, "ENABLE_IN_MEMORY_LOGS", True)
    root = logging.getLogger()
    watched = [root, logging.getLogger("inference")] + [
        logging.getLogger(name) for name in memory_handler.EXTRACTED_PACKAGE_LOGGERS
    ]
    saved = {target: list(target.handlers) for target in watched}
    memory_handler._log_entries.clear()
    yield
    memory_handler._log_entries.clear()
    for target, handlers in saved.items():
        target.handlers = handlers


def _build_app(monkeypatch):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    interface = http_api.HttpInterface(model_manager=model_manager)

    return interface.app


@contextmanager
def _serve(app) -> Iterator[str]:
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
        f"/info?api_key={API_KEY}",
        f"/info?foo=bar&api_key={API_KEY}",
        f"/info?api%5Fkey={API_KEY}",
        f"/info?service_secret={SERVICE_SECRET}",
    ],
)
def test_redact_path_with_query_string_removes_credentials(path):
    redacted = redact_path_with_query_string(path)

    assert API_KEY not in redacted
    assert SERVICE_SECRET not in redacted
    assert redacted.startswith("/info?")


def test_redact_path_with_query_string_keeps_other_parameters_encoded():
    path = f"/model/1?image=https%3A%2F%2Fexample.com%2Fa.jpg&api_key={API_KEY}"

    redacted = redact_path_with_query_string(path)

    assert (
        redacted == "/model/1?image=https%3A%2F%2Fexample.com%2Fa.jpg&api_key=TE***23"
    )


def test_redact_path_without_query_string_is_unchanged():
    assert redact_path_with_query_string("/model/1") == "/model/1"


def test_uvicorn_access_log_and_memory_logs_carry_no_api_key(
    monkeypatch, caplog, access_logger, in_memory_logs
):
    app = _build_app(monkeypatch)
    access_logger.addHandler(caplog.handler)

    with caplog.at_level(logging.INFO, logger=access_logger.name):
        with _serve(app) as base_url:
            requests.get(f"{base_url}/info?api_key={API_KEY}", timeout=5)
            requests.get(f"{base_url}/info?api%5Fkey={API_KEY}", timeout=5)
            logs = requests.get(f"{base_url}/logs?limit=1000", timeout=5).json()

    access_lines = [
        record.getMessage()
        for record in caplog.records
        if record.name == access_logger.name
    ]
    assert any("/info?api_key=" in line for line in access_lines)
    assert all(API_KEY not in line for line in access_lines)
    memory_messages = [entry["message"] for entry in logs["logs"]]
    assert any("/info?api_key=" in message for message in memory_messages)
    assert all(API_KEY not in message for message in memory_messages)


def test_redaction_filter_is_installed_once(monkeypatch, access_logger):
    _build_app(monkeypatch)
    _build_app(monkeypatch)

    installed = [
        existing
        for existing in access_logger.filters
        if isinstance(existing, AccessLogRedactionFilter)
    ]
    assert len(installed) == 1
