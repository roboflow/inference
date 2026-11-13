"""Redaction of credentials from uvicorn's access log.

uvicorn's ``uvicorn.access`` logger writes the request line with its full query
string, so a client calling ``POST /model/1?api_key=...`` would get its key
printed to stdout and stored by the in-memory ``/logs`` handler. The filter in
this module rewrites the record arguments before any handler formats them.
"""

import logging
from typing import Any
from urllib.parse import unquote_plus

from inference.core.utils.requests import deduct_api_key_from_string

UVICORN_ACCESS_LOGGER_NAME = "uvicorn.access"


def redact_path_with_query_string(path: str) -> str:
    """Redact API keys and service secrets from a request path and query string.

    Each query parameter is checked in its decoded form, so a percent-encoded
    name such as ``api%5Fkey`` is caught as well. Parameters carrying no
    credential keep their original encoding.

    Args:
        path: The path with its raw query string, as uvicorn logs it.

    Returns:
        The path with every credential value redacted.
    """
    base, separator, query = path.partition("?")
    if not separator:
        redacted_path = deduct_api_key_from_string(path)

        return redacted_path

    parameters = []
    for parameter in query.split("&"):
        decoded = unquote_plus(parameter)
        redacted = deduct_api_key_from_string(decoded)
        parameters.append(redacted if redacted != decoded else parameter)
    redacted_path = (
        f"{deduct_api_key_from_string(base)}{separator}{'&'.join(parameters)}"
    )

    return redacted_path


def _redact_argument(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    return redact_path_with_query_string(value)


class AccessLogRedactionFilter(logging.Filter):
    """Logger filter that strips credentials from uvicorn access log records.

    uvicorn logs ``(client, method, path_with_query, http_version, status)``;
    every string argument is redacted so the filter does not depend on that
    order. The filter never drops a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact the record's string arguments in place.

        Args:
            record: The access log record.

        Returns:
            Always True, so the record is still emitted.
        """
        try:
            if isinstance(record.args, tuple):
                record.args = tuple(_redact_argument(arg) for arg in record.args)
            elif isinstance(record.msg, str) and not record.args:
                record.msg = redact_path_with_query_string(record.msg)
        except Exception:
            pass

        return True


def install_access_log_redaction() -> None:
    """Attach the redaction filter to the ``uvicorn.access`` logger once.

    A logger-level filter runs before every handler of that logger, including
    the in-memory ``/logs`` handler.
    """
    access_logger = logging.getLogger(UVICORN_ACCESS_LOGGER_NAME)
    if any(
        isinstance(existing, AccessLogRedactionFilter)
        for existing in access_logger.filters
    ):
        return

    access_logger.addFilter(AccessLogRedactionFilter())
