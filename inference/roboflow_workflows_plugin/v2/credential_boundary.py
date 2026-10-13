"""Fixed diagnostics for synchronous V2 calls into the shared host libraries.

The permanently registered filters do nothing outside a context-local V2 call.
They run on audited emitters before direct, propagated or queued handlers. This
covers ordinary host/urllib3 diagnostics, not arbitrary application loggers,
callbacks on other threads, or tools that dump traceback frame locals.
"""

import logging
from contextvars import ContextVar
from enum import Enum
from threading import Lock
from typing import Any, Callable

from inference.core import exceptions
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError


class HostOperation(Enum):
    """Fixed operation labels allowed in V2 host diagnostics."""

    WORKSPACE = "workspace lookup"
    METADATA = "custom metadata upload"
    SEARCH = "project image search"
    IMAGE = "candidate image download"
    WORKFLOW = "saved workflow retrieval"


_OPERATION: ContextVar[HostOperation | None] = ContextVar(
    "v2_host_operation", default=None
)
_REGISTRATION_LOCK = Lock()
# Logger filters do not apply to descendant emissions. These are the actual
# emitters used by the pinned synchronous host and urllib3 request/retry paths.
AUDITED_LOGGERS = (
    "inference",
    "urllib3.connectionpool",
    "urllib3.connection",
    "urllib3.poolmanager",
    "urllib3.util.retry",
    "urllib3.response",
    "backoff",
)
_SAFE_RECORD_FIELDS = frozenset(
    {
        "name",
        "levelno",
        "levelname",
        "pathname",
        "filename",
        "module",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }
)


class _ScopedDiagnosticFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        operation = _OPERATION.get()
        if operation is None:
            return True

        # Do not format or traverse arbitrary payloads: __str__, encoded URLs,
        # exception graphs and extras can all carry credentials. Keep only
        # ordinary log metadata, replacing all diagnostic-bearing fields.
        metadata = {
            key: value
            for key, value in record.__dict__.items()
            if key in _SAFE_RECORD_FIELDS
        }
        record.__dict__.clear()
        record.__dict__.update(metadata)
        # Normalize even structlog event dicts to a safe stdlib message, and
        # discard their _logger/_name flags with the original payload. This
        # makes ProcessorFormatter use its foreign-record path both directly
        # and after QueueHandler.prepare() has converted the message to str.
        record.msg = f"V2 host diagnostic: {operation.value}"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        record.v2_operation = operation.value
        return True


_FILTER = _ScopedDiagnosticFilter()


def register_diagnostic_filters() -> None:
    """Register once per emitter, including concurrent adapter construction."""
    with _REGISTRATION_LOCK:
        for name in AUDITED_LOGGERS:
            logger = logging.getLogger(name)
            if _FILTER not in logger.filters:
                logger.addFilter(_FILTER)


# Order matters: subclasses must precede their common request-error base.
# Each category names an imported, trusted class, never a dynamic host type.
_ERROR_CATEGORIES = (
    (exceptions.RoboflowAPINotAuthorizedError, "authorization failed"),
    (exceptions.RoboflowAPIForbiddenError, "access forbidden"),
    (exceptions.RoboflowAPINotNotFoundError, "resource not found"),
    (exceptions.RoboflowAPITimeoutError, "request timed out"),
    (exceptions.RoboflowAPIConnectionError, "connection failed"),
    (exceptions.RoboflowAPIUnsuccessfulRequestError, "request unsuccessful"),
    (exceptions.MalformedWorkflowResponseError, "malformed workflow response"),
    (exceptions.MalformedRoboflowAPIResponseError, "malformed API response"),
    (exceptions.RoboflowAPIRequestError, "request failed"),
    (exceptions.WorkspaceLoadError, "workspace unavailable"),
    (exceptions.InputImageLoadError, "image could not be loaded"),
    (exceptions.CacheUnavailableError, "cache unavailable"),
    (FileNotFoundError, "local workflow file not found"),
    (TimeoutError, "request timed out"),
    (
        ConnectionError,
        "connection unavailable (check host availability and OFFLINE_MODE)",
    ),
    (ValueError, "invalid host data"),
)


def _safe_failure(
    error: Exception, operation: HostOperation, platform: bool
) -> Exception:
    error_class, category = RuntimeError, "unexpected host failure"
    for expected, description in _ERROR_CATEGORIES:
        if isinstance(error, expected):
            error_class, category = expected, description
            break
    message = f"{operation.value} failed: {category}"
    if platform:
        failure = PlatformRequestError(
            message, error_type=f"{error_class.__module__}.{error_class.__qualname__}"
        )

        return failure
    failure = error_class(message)

    return failure


def call_host(
    operation: HostOperation,
    function: Callable[..., Any],
    *,
    platform: bool = False,
    **arguments: Any,
) -> Any:
    """Keep host success behavior; replace failures without retaining objects.

    Only a fixed operation marker enters the ContextVar. Nested calls restore
    the preceding marker using tokens; unrelated threads remain unaffected.

    Args:
        operation: Fixed category for this synchronous host call.
        function: Host function to invoke with unchanged arguments.
        platform: Wrap failures in PlatformRequestError when true; otherwise
            preserve allowlisted host exception classes using fresh instances.
        **arguments: Keyword arguments forwarded unchanged to the host.

    Returns:
        The host function's unmodified successful result.

    Raises:
        PlatformRequestError: A platform call failed and platform is true.
        Exception: A fresh allowlisted host error, or RuntimeError for an
            unexpected failure, when platform is false.
    """
    register_diagnostic_filters()
    token = _OPERATION.set(operation)
    try:
        try:
            result = function(**arguments)
        except Exception as error:
            failure = _safe_failure(error, operation, platform)
        else:
            return result
    finally:
        _OPERATION.reset(token)
    # Raising after our except removes the host's active context. Clear once
    # more on a bare re-raise: callers may themselves be handling an exception.
    try:
        raise failure
    except Exception:
        failure.__cause__ = None
        failure.__context__ = None
        raise
