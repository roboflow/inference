"""Transport failures must not expose credentials through logs or spans."""

import asyncio
import logging
import traceback
from typing import Generator, Type

import aiohttp
import pytest
import requests
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from inference.core import roboflow_api, telemetry
from inference.core.exceptions import (
    RetryRequestError,
    RoboflowAPIConnectionError,
    RoboflowAPITimeoutError,
)

API_KEY = "synthetic-transport-api-key"
SERVICE_SECRET = "synthetic-transport-service-secret"
SENSITIVE_URL = (
    f"https://example.invalid/endpoint?api_key={API_KEY}"
    f"&service_secret={SERVICE_SECRET}"
)
TIMEOUT_MESSAGE = "Timeout when attempting to connect to Roboflow API."
CONNECTION_MESSAGE = "Could not connect to Roboflow API."
INVALID_URL_MESSAGE = "Could not build a valid Roboflow API URL. Check API_BASE_URL."
LOGGER = logging.getLogger(__name__)


@pytest.fixture
def span_exporter(monkeypatch) -> Generator[InMemorySpanExporter, None, None]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "OFFLINE_MODE", False)
    monkeypatch.setattr(telemetry, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer(__name__))

    try:
        yield exporter
    finally:
        provider.shutdown()


def _raise_transport_error(error_type: Type[Exception], *, retried: bool) -> None:
    try:
        try:
            raise RuntimeError("Underlying connection failed for " + SENSITIVE_URL)
        except RuntimeError as cause:
            raise error_type("Request failed for " + SENSITIVE_URL) from cause
    except error_type as error:
        if retried:
            raise RetryRequestError("Retries exhausted", inner_error=error) from error

        raise


def _assert_safe_failure(
    error: Exception,
    *,
    message: str,
    caplog,
    exporter: InMemorySpanExporter,
) -> None:
    assert str(error) == message
    formatted_traceback = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    # Cover both explicit record_error() and automatic span-exit recording.
    events = [event for event in span.events if event.name == "exception"]
    assert len(events) == 2

    for event in events:
        assert event.attributes["exception.message"] == message
        assert event.attributes["exception.type"].endswith(type(error).__name__)

    serialized_span = span.to_json()
    for secret in (API_KEY, SERVICE_SECRET):
        assert secret not in formatted_traceback
        assert secret not in caplog.text
        assert secret not in serialized_span

    assert type(error).__name__ in caplog.text


@pytest.mark.parametrize("retried", [False, True])
@pytest.mark.parametrize(
    "transport_error,mapped_error,message",
    [
        (requests.Timeout, RoboflowAPITimeoutError, TIMEOUT_MESSAGE),
        (requests.ConnectTimeout, RoboflowAPITimeoutError, TIMEOUT_MESSAGE),
        (requests.ReadTimeout, RoboflowAPITimeoutError, TIMEOUT_MESSAGE),
        (requests.ConnectionError, RoboflowAPIConnectionError, CONNECTION_MESSAGE),
        (ConnectionError, RoboflowAPIConnectionError, CONNECTION_MESSAGE),
    ],
)
def test_sync_transport_errors_keep_credentials_out_of_diagnostics(
    transport_error, mapped_error, message, retried, span_exporter, caplog
) -> None:
    @roboflow_api.wrap_roboflow_api_errors()
    def fail() -> None:
        _raise_transport_error(transport_error, retried=retried)

    with pytest.raises(mapped_error) as caught:
        try:
            fail()
        except mapped_error:
            LOGGER.exception("Platform request failed")
            raise

    _assert_safe_failure(
        caught.value, message=message, caplog=caplog, exporter=span_exporter
    )


@pytest.mark.parametrize(
    "url",
    [
        # No scheme: requests raises MissingSchema with the whole URL in its text.
        f"api.example.invalid/endpoint?api_key={API_KEY}&service_secret={SERVICE_SECRET}",
        f"ftp://api.example.invalid/endpoint?api_key={API_KEY}&service_secret={SERVICE_SECRET}",
    ],
)
def test_invalid_url_keeps_credentials_out_of_diagnostics(
    url, monkeypatch, span_exporter, caplog
) -> None:
    monkeypatch.setattr(roboflow_api, "OFFLINE_MODE", False)

    with pytest.raises(RoboflowAPIConnectionError) as caught:
        try:
            roboflow_api.get_from_url(url)
        except RoboflowAPIConnectionError:
            LOGGER.exception("Platform request failed")
            raise

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True
    _assert_safe_failure(
        caught.value,
        message=INVALID_URL_MESSAGE,
        caplog=caplog,
        exporter=span_exporter,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("retried", [False, True])
@pytest.mark.parametrize(
    "transport_error,mapped_error,message",
    [
        (asyncio.TimeoutError, RoboflowAPITimeoutError, TIMEOUT_MESSAGE),
        (aiohttp.ServerTimeoutError, RoboflowAPITimeoutError, TIMEOUT_MESSAGE),
        (aiohttp.ClientConnectionError, RoboflowAPIConnectionError, CONNECTION_MESSAGE),
        (ConnectionError, RoboflowAPIConnectionError, CONNECTION_MESSAGE),
    ],
)
async def test_async_transport_errors_keep_credentials_out_of_diagnostics(
    transport_error, mapped_error, message, retried, span_exporter, caplog
) -> None:
    @roboflow_api.wrap_roboflow_api_errors_async()
    async def fail() -> None:
        _raise_transport_error(transport_error, retried=retried)

    with pytest.raises(mapped_error) as caught:
        try:
            await fail()
        except mapped_error:
            LOGGER.exception("Platform request failed")
            raise

    _assert_safe_failure(
        caught.value, message=message, caplog=caplog, exporter=span_exporter
    )
