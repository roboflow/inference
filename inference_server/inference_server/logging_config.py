"""Logging configuration for the inference server.

``configure_logging()`` installs the application log handler on the
``inference_server`` logger at ``LOG_LEVEL``. With ``API_LOGGING_ENABLED`` the
handler writes one JSON object per line carrying the correlation key named by
``CORRELATION_ID_LOG_KEY`` and the trace and execution ids when present;
otherwise a plain line, and does not propagate to the root logger. uvicorn's
own loggers keep uvicorn's default handlers,
formats and ``INFO`` level, installed here because the runner starts uvicorn
with ``log_config=None``. With ``STRUCTURED_API_LOGGING`` set as well, uvicorn's
access log is silenced and ``install_structured_access_log(app)`` adds a
middleware that logs every response through the application logger with the
request, model and trace fields of the legacy structured access log, health
paths at ``DEBUG``. With both flags unset the access log is uvicorn's default
line: ``<client> - "<METHOD> <path> HTTP/1.1" <status>``. Every call replaces
only the handlers it installed before, so the ``python -m inference_server.app``
runner, which imports the application twice, ends with one owned handler per
logger and keeps handlers installed by anyone else.
"""

import json
import logging
import sys
import time
import traceback
import warnings
from typing import Any, Dict, List

from inference_sdk.config import execution_id
from uvicorn.config import LOGGING_CONFIG as UVICORN_LOGGING_CONFIG
from uvicorn.logging import AccessFormatter, DefaultFormatter
from uvicorn.protocols.utils import get_client_addr

from inference_server import configuration, telemetry
from inference_server.middlewares.correlation_id import correlation_id
from inference_server.middlewares.headers import (
    MODEL_COLD_START_COUNT_HEADER,
    MODEL_COLD_START_HEADER,
    MODEL_ID_HEADER,
    MODEL_LOAD_TIME_HEADER,
    PROCESSING_TIME_HEADER,
    TRACE_ID_HEADER,
    WORKFLOW_ID_HEADER,
    WORKSPACE_ID_HEADER,
)

APPLICATION_LOGGER_NAME = "inference_server"
ACCESS_LOGGER_NAME = "inference_server.access"
UVICORN_LOGGER_NAME = "uvicorn"
UVICORN_ERROR_LOGGER_NAME = "uvicorn.error"
UVICORN_ACCESS_LOGGER_NAME = "uvicorn.access"

PLAIN_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
UVICORN_DEFAULT_LOG_FORMAT = UVICORN_LOGGING_CONFIG["formatters"]["default"]["fmt"]
UVICORN_ACCESS_LOG_FORMAT = UVICORN_LOGGING_CONFIG["formatters"]["access"]["fmt"]

HEALTH_LOG_PATHS = frozenset(
    {
        "/healthz",
        "/readiness",
        "/",
        "/info",
        "/metrics",
        "/v2/server/health",
        "/v2/server/ready",
    }
)

ACCESS_LOG_HEADER_FIELDS = {
    "processing_time": PROCESSING_TIME_HEADER,
    "model_cold_start": MODEL_COLD_START_HEADER,
    "model_cold_start_count": MODEL_COLD_START_COUNT_HEADER,
    "model_load_time": MODEL_LOAD_TIME_HEADER,
    "model_id": MODEL_ID_HEADER,
    "workflow_id": WORKFLOW_ID_HEADER,
    "workspace_id": WORKSPACE_ID_HEADER,
    "trace_id": TRACE_ID_HEADER,
}

STRUCTURED_FIELDS_ATTRIBUTE = "structured_fields"


def _resolve_level(name: str) -> int:
    level = logging.getLevelName(name.upper())
    if not isinstance(level, int):
        return logging.WARNING
    return level


def structured_access_log_enabled() -> bool:
    """Return whether the structured access log replaces uvicorn's.

    Returns:
        True when ``API_LOGGING_ENABLED`` and ``STRUCTURED_API_LOGGING`` are
        both set, as the legacy server requires.
    """
    enabled = bool(
        configuration.API_LOGGING_ENABLED and configuration.STRUCTURED_API_LOGGING
    )

    return enabled


class _OwnedStreamHandler(logging.StreamHandler):
    """Stream handler installed by ``configure_logging``; replaced on re-run."""


def _format_exception(exc_info: Any) -> Dict[str, Any]:
    exc_type, exc_value, exc_tb = exc_info
    stacktrace = [
        {
            "filename": frame.filename,
            "lineno": frame.lineno,
            "function": frame.name,
            "code": frame.line,
        }
        for frame in traceback.extract_tb(exc_tb)
    ]
    formatted = {
        "type": exc_type.__name__ if exc_type else "N/A",
        "message": str(exc_value) if exc_value else "N/A",
        "stacktrace": stacktrace,
    }

    return formatted


class JsonFormatter(logging.Formatter):
    """Stdlib JSON formatter adding the correlation, trace and execution ids.

    Dynamic fields travel in the record attribute named by
    ``STRUCTURED_FIELDS_ATTRIBUTE`` (a dict passed through ``extra``) and are
    merged after the fixed fields, which keep their names; the context fields
    are added only when the record does not carry them. Exceptions are
    rendered as the legacy object with ``type``, ``message`` and
    ``stacktrace`` entries.
    """

    def __init__(self, correlation_key: str) -> None:
        super().__init__()
        self._correlation_key = correlation_key

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "message": record.getMessage(),
            "severity": record.levelname,
            "logger": record.name,
            "timestamp": self.formatTime(record),
            "filename": record.filename,
            "func_name": record.funcName,
            "lineno": record.lineno,
        }
        structured_fields = getattr(record, STRUCTURED_FIELDS_ATTRIBUTE, None)
        if isinstance(structured_fields, dict):
            for key, value in structured_fields.items():
                payload.setdefault(key, value)

        request_id = correlation_id.get()
        if request_id:
            payload.setdefault(self._correlation_key, request_id)
        for key, value in telemetry.trace_context_fields().items():
            payload.setdefault(key, value)
        execution_id_value = execution_id.get()
        if execution_id_value:
            payload.setdefault("execution_id", execution_id_value)
        if record.exc_info:
            payload["exception"] = _format_exception(record.exc_info)

        line = json.dumps(payload, default=str)

        return line


def _access_log_fields(scope: dict, message: dict, started_at: float) -> Dict[str, Any]:
    status_code = int(message["status"])
    fields: Dict[str, Any] = {
        "method": scope.get("method", ""),
        "path": scope.get("path", ""),
        "status": status_code,
        "status_code": status_code,
        "duration_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
    }

    response_headers: Dict[str, str] = {}
    for name, value in message.get("headers", []):
        response_headers.setdefault(
            name.decode("latin-1").lower(), value.decode("latin-1")
        )
    header_fields = dict(ACCESS_LOG_HEADER_FIELDS)
    if configuration.EXECUTION_ID_HEADER:
        header_fields["execution_id"] = configuration.EXECUTION_ID_HEADER
    for field_name, header_name in header_fields.items():
        value = response_headers.get(header_name.lower())
        if value is not None:
            fields[field_name] = value
    request_id = correlation_id.get()
    if request_id:
        fields[configuration.CORRELATION_ID_LOG_KEY] = request_id

    return fields


def _log_access(scope: dict, message: dict, started_at: float) -> None:
    fields = _access_log_fields(scope, message, started_at)
    request_line = (
        f'{get_client_addr(scope)} - "{fields["method"]} {fields["path"]} '
        f'HTTP/{scope.get("http_version", "1.1")}" {fields["status"]}'
    )
    level = logging.DEBUG if fields["path"] in HEALTH_LOG_PATHS else logging.INFO
    logging.getLogger(ACCESS_LOGGER_NAME).log(
        level, request_line, extra={STRUCTURED_FIELDS_ATTRIBUTE: fields}
    )


class StructuredAccessLogMiddleware:
    """Raw ASGI middleware logging every response as a structured entry.

    Added last, so it wraps every other middleware: the entry is logged when
    the response starts, from the response headers the inner middlewares have
    stamped, and ``duration_ms`` covers the whole stack.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = time.perf_counter()

        async def _send(message: dict) -> None:
            if message["type"] == "http.response.start":
                _log_access(scope, message, started_at)
            await send(message)

        await self.app(scope, receive, _send)


def _replace_owned_handlers(
    target: logging.Logger, handlers: List[logging.Handler]
) -> None:
    foreign = [
        handler
        for handler in target.handlers
        if not isinstance(handler, _OwnedStreamHandler)
    ]
    target.handlers = foreign + handlers


def _install_unless_foreign(
    target: logging.Logger, handler: _OwnedStreamHandler
) -> None:
    foreign = [
        existing
        for existing in target.handlers
        if not isinstance(existing, _OwnedStreamHandler)
    ]
    if any(
        isinstance(existing.formatter, type(handler.formatter)) for existing in foreign
    ):
        target.handlers = foreign
    else:
        target.handlers = foreign + [handler]


def _owned_handler(stream: Any, formatter: logging.Formatter) -> _OwnedStreamHandler:
    handler = _OwnedStreamHandler(stream)
    handler.setFormatter(formatter)

    return handler


def _configure_uvicorn_loggers() -> None:
    uvicorn_logger = logging.getLogger(UVICORN_LOGGER_NAME)
    _install_unless_foreign(
        uvicorn_logger,
        _owned_handler(sys.stderr, DefaultFormatter(UVICORN_DEFAULT_LOG_FORMAT)),
    )
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_logger.propagate = False

    error_logger = logging.getLogger(UVICORN_ERROR_LOGGER_NAME)
    _replace_owned_handlers(error_logger, [])
    error_logger.setLevel(logging.INFO)
    error_logger.propagate = True

    access_logger = logging.getLogger(UVICORN_ACCESS_LOGGER_NAME)
    access_logger.propagate = False
    access_logger.setLevel(logging.INFO)
    if structured_access_log_enabled():
        access_logger.handlers = []
        return

    _install_unless_foreign(
        access_logger,
        _owned_handler(sys.stdout, AccessFormatter(UVICORN_ACCESS_LOG_FORMAT)),
    )


def configure_logging() -> None:
    """Configure application and access logging from the environment.

    Installs the application handler on the ``inference_server`` logger at
    ``LOG_LEVEL``, JSON-formatted with ``API_LOGGING_ENABLED`` and plain
    otherwise, and stops it propagating to the root logger like legacy. Gives
    uvicorn's loggers uvicorn's default handlers and ``INFO`` level, silencing
    the access line only when the structured access log replaces it. A later
    call replaces only the handlers an earlier call installed and keeps any
    other handler in place; a uvicorn logger that already has another handler
    gets no second one; the silenced access logger drops every handler, as
    the legacy server does.
    """
    level = _resolve_level(configuration.LOG_LEVEL)
    if configuration.LOG_LEVEL.upper() in ("ERROR", "FATAL"):
        warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime.*")

    if configuration.API_LOGGING_ENABLED:
        formatter: logging.Formatter = JsonFormatter(
            configuration.CORRELATION_ID_LOG_KEY
        )
    else:
        formatter = logging.Formatter(PLAIN_LOG_FORMAT)
    application_logger = logging.getLogger(APPLICATION_LOGGER_NAME)
    _replace_owned_handlers(application_logger, [_owned_handler(sys.stderr, formatter)])
    application_logger.setLevel(level)
    application_logger.propagate = False

    _configure_uvicorn_loggers()


def install_structured_access_log(app: Any) -> None:
    """Add the structured access log middleware when the flags enable it.

    Must be called after every other ``app.add_middleware`` call so the
    middleware wraps the whole stack.

    Args:
        app: The FastAPI application.
    """
    if not structured_access_log_enabled():
        return

    app.add_middleware(StructuredAccessLogMiddleware)
