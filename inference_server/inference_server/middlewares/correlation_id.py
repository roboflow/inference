"""Correlation id and execution id of every HTTP request.

``CorrelationIdMiddleware`` reads the request header named by
``configuration.CORRELATION_ID_HEADER``. With ``API_LOGGING_ENABLED`` any
non-empty value is accepted; otherwise a value that is not a UUID is replaced.
A missing or replaced value becomes a new uuid4 hex. The final id is written
back into the request headers, published on the ``correlation_id`` contextvar
and echoed on the response.

Under ``GCP_SERVERLESS`` it also prepares the execution id before any
authorization runs: the request header named by
``configuration.EXECUTION_ID_HEADER``, or a generated
``<time_ns>_<4 hex>``, written back into the request headers and published on
``inference_sdk.config.execution_id``. The request start is published on
``request_start_time`` so a middleware answering before any route runs can
report its processing time.
"""

import contextvars
import time
import uuid
from typing import List, Optional, Tuple

from inference_sdk.config import execution_id

from inference_server import configuration

correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)
request_start_time: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "request_start_time", default=None
)


def _is_valid_uuid4(value: str) -> bool:
    try:
        uuid.UUID(value, version=4)
    except ValueError:
        return False
    return True


def _header(headers: List[Tuple[bytes, bytes]], name: bytes) -> str:
    return next(
        (value.decode("latin-1") for key, value in headers if key.lower() == name),
        "",
    )


def _with_header(
    headers: List[Tuple[bytes, bytes]], name: bytes, value: str
) -> List[Tuple[bytes, bytes]]:
    replaced = [(key, current) for key, current in headers if key.lower() != name]
    replaced.append((name, value.encode("latin-1")))
    return replaced


def _resolve_request_id(incoming: str) -> str:
    if not incoming:
        return uuid.uuid4().hex
    if not configuration.API_LOGGING_ENABLED and not _is_valid_uuid4(incoming):
        return uuid.uuid4().hex
    return incoming


class CorrelationIdMiddleware:
    """Raw ASGI middleware assigning the request correlation and execution ids."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()
        header_name = configuration.CORRELATION_ID_HEADER.lower().encode("latin-1")
        headers = list(scope.get("headers", []))
        request_id = _resolve_request_id(_header(headers, header_name))
        headers = _with_header(headers, header_name, request_id)

        execution_id_value = None
        if configuration.GCP_SERVERLESS and configuration.EXECUTION_ID_HEADER:
            execution_header = configuration.EXECUTION_ID_HEADER.lower().encode(
                "latin-1"
            )
            execution_id_value = (
                _header(headers, execution_header)
                or f"{time.time_ns()}_{uuid.uuid4().hex[:4]}"
            )
            headers = _with_header(headers, execution_header, execution_id_value)
        scope["headers"] = headers

        async def _send(message) -> None:
            if message["type"] == "http.response.start":
                message = {
                    **message,
                    "headers": _with_header(
                        list(message.get("headers", [])), header_name, request_id
                    ),
                }
            await send(message)

        id_token = correlation_id.set(request_id)
        start_token = request_start_time.set(started_at)
        execution_token = (
            execution_id.set(execution_id_value)
            if execution_id_value is not None
            else None
        )
        try:
            await self.app(scope, receive, _send)
        finally:
            if execution_token is not None:
                execution_id.reset(execution_token)
            request_start_time.reset(start_token)
            correlation_id.reset(id_token)
