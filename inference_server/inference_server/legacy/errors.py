from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
from typing import Any, Callable, List, Optional, Tuple

import pydantic
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from inference_model_manager.pipelines import InvalidPipelineIdError
from starlette.exceptions import HTTPException

from inference_models.errors import (
    EnvironmentConfigurationError,
    FileHashSumMissmatch,
    InvalidEnvVariable,
    InvalidParameterError,
    MissingDependencyError,
    ModelInputError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageNegotiationError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
    UntrustedFileError,
)
from inference_server import configuration
from inference_server.errors import PayloadTooLargeError, ServerBusyError
from inference_server.gateway import _redact_secrets
from inference_server.legacy.telemetry_recording import (
    record_route_error,
    request_telemetry_scope,
)

logger = logging.getLogger(__name__)

PAYLOAD_TOO_LARGE_MESSAGE = "Request payload too large."
UNAUTHORIZED_MESSAGE = (
    "Unauthorized access to roboflow API - check API key and make sure the key is "
    "valid for workspace you use. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)
NOT_FOUND_MESSAGE = (
    "Requested Roboflow resource not found. Make sure that workspace, project or "
    "model you referred in request exists."
)
MODEL_RESTRICTED_MESSAGE = (
    "Model loading failed due to restrictions of server configuration - usually due "
    "to excessive runtime memory requirement of the model (for instance caused by "
    "large input size)."
)
MODEL_PACKAGE_BROKEN_MESSAGE = "Model package is broken."
REGISTRY_UNREACHABLE_MESSAGE = "Internal error. Could not connect to Roboflow API."
REGISTRY_REQUEST_FAILED_MESSAGE = "Internal error. Request to Roboflow API failed."
REGISTRY_TIMEOUT_MESSAGE = "Timeout when attempting to connect to Roboflow API."
SERVICE_MISCONFIGURATION_MESSAGE = "Service misconfiguration."
INFERENCE_TIMEOUT_MESSAGE = "Timed out waiting for inference result."
INTERNAL_ERROR_MESSAGE = "Internal error."
INVALID_MODEL_ID_MESSAGE = "Invalid Model ID sent in request."
MODEL_NOT_READY_MESSAGE = "Model is temporarily not ready - retry request."

MIN_REDACTED_VALUE_LENGTH = 6
REDACTED_VALUE = "***"
UNPRINTABLE_ERROR_MESSAGE = "<unprintable>"
MAX_REDACTED_CHAIN_LENGTH = 8

REDACTION_VALUES: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "legacy_redaction_values", default=None
)

MODEL_ACCESS_ERROR_MESSAGES = {
    402: "Not enough credits to perform this request. Verify your workspace billing page.",
    403: "Unauthorized access to roboflow API - check API key and make sure the key is valid and "
    "have required scopes. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one.",
    423: "Roboflow API usage is paused. Please contact your workspace administrator to re-enable api keys.",
}


class BodyTooLargeHTTPException(HTTPException):
    def __init__(self, detail: str) -> None:
        super().__init__(status_code=413, detail=detail)


class LegacyHTTPError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        extra: Optional[dict] = None,
        headers: Optional[dict] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.extra = extra or {}
        self.headers = headers or {}


class ModelNotReadyError(LegacyHTTPError):
    def __init__(self) -> None:
        super().__init__(503, MODEL_NOT_READY_MESSAGE, headers={"Retry-After": "1"})


class ImageFetchError(LegacyHTTPError):
    pass


class MissingServiceSecretError(LegacyHTTPError):
    def __init__(self) -> None:
        super().__init__(500, SERVICE_MISCONFIGURATION_MESSAGE)


def add_redaction_values(*values: Any) -> None:
    """Make credentials of the request being served known to error logging.

    Values shorter than ``MIN_REDACTED_VALUE_LENGTH`` characters are ignored so
    ordinary text is not blanked. Without an active request scope nothing is
    stored.

    Args:
        *values: Credentials of the request, such as its API key.
    """
    held = REDACTION_VALUES.get()
    if held is None:
        return

    for value in values:
        if (
            isinstance(value, str)
            and len(value) >= MIN_REDACTED_VALUE_LENGTH
            and value not in held
        ):
            held.append(value)


def redact_text(text: str, values: Any = ()) -> str:
    """Remove credentials from text, first by value and then by pattern.

    Args:
        text: Text that may carry credentials.
        values: Credential values to replace wherever they occur; those shorter
            than ``MIN_REDACTED_VALUE_LENGTH`` characters are skipped.

    Returns:
        The text with each searched value replaced by ``***`` and labelled or
        URL-embedded credentials redacted.
    """
    searched = sorted(
        {
            value
            for value in values
            if isinstance(value, str) and len(value) >= MIN_REDACTED_VALUE_LENGTH
        },
        key=len,
        reverse=True,
    )
    for value in searched:
        text = text.replace(value, REDACTED_VALUE)

    return _redact_secrets(text)


def _redacted_message(error: BaseException) -> str:
    try:
        return redact_text(str(error), REDACTION_VALUES.get() or ())
    except Exception:
        return UNPRINTABLE_ERROR_MESSAGE


def _redacted_copy(error: BaseException, depth: int = 0) -> BaseException:
    error_type = type(
        type(error).__name__,
        (Exception,),
        {"__module__": type(error).__module__},
    )
    copy = error_type(_redacted_message(error))
    copy.__traceback__ = error.__traceback__
    if depth >= MAX_REDACTED_CHAIN_LENGTH:
        return copy
    if error.__cause__ is not None:
        copy.__cause__ = _redacted_copy(error.__cause__, depth + 1)
    elif error.__context__ is not None and not error.__suppress_context__:
        copy.__context__ = _redacted_copy(error.__context__, depth + 1)

    return copy


def _redacted_exc_info(error: BaseException) -> tuple:
    copy = _redacted_copy(error)

    return type(copy), copy, copy.__traceback__


def legacy_error_response(error: BaseException) -> JSONResponse:
    if isinstance(error, LegacyHTTPError):
        return JSONResponse(
            status_code=error.status_code,
            content={"message": error.message, **error.extra},
            headers=error.headers or None,
        )
    if isinstance(error, (PayloadTooLargeError, BodyTooLargeHTTPException)):
        return JSONResponse(
            status_code=413, content={"message": PAYLOAD_TOO_LARGE_MESSAGE}
        )
    if isinstance(error, ServerBusyError):
        return JSONResponse(
            status_code=503,
            content={"message": str(error)},
            headers={"Retry-After": "1"},
        )
    if isinstance(error, asyncio.TimeoutError):
        return JSONResponse(
            status_code=504, content={"message": INFERENCE_TIMEOUT_MESSAGE}
        )

    answer = _mapped_answer(error)
    if answer is None:
        logger.error("Unhandled legacy route error", exc_info=_redacted_exc_info(error))
        return JSONResponse(
            status_code=500, content={"message": INTERNAL_ERROR_MESSAGE}
        )

    status_code, content = answer
    if status_code == 402 and isinstance(error, RuntimeError):
        logger.warning("%s: %s", type(error).__name__, _redacted_message(error))
    else:
        logger.error(
            "%s: %s",
            type(error).__name__,
            _redacted_message(error),
            exc_info=_redacted_exc_info(error),
        )

    return JSONResponse(status_code=status_code, content=content)


def _mapped_answer(error: BaseException) -> Optional[Tuple[int, dict]]:
    cause = error.__cause__
    if isinstance(error, ModelInputError):
        return 400, {
            "message": f"Error with model input. Cause: {error}",
            "help_url": error.help_url,
        }
    if isinstance(error, UnauthorizedModelAccessError) or (
        isinstance(error, PermissionError)
        and isinstance(cause, UnauthorizedModelAccessError)
    ):
        return 401, {"message": UNAUTHORIZED_MESSAGE}
    if isinstance(error, ModelNotFoundError) or (
        isinstance(error, LookupError) and isinstance(cause, ModelNotFoundError)
    ):
        return 404, {"message": NOT_FOUND_MESSAGE}
    if isinstance(error, LookupError) and isinstance(cause, InvalidPipelineIdError):
        return 400, {"message": INVALID_MODEL_ID_MESSAGE}
    if isinstance(error, ModelPackageNegotiationError):
        return 500, {
            "message": f"Could not negotiate model package - {error}",
            "help_url": error.help_url,
        }
    if isinstance(
        error,
        (
            EnvironmentConfigurationError,
            InvalidEnvVariable,
            MissingDependencyError,
            InvalidParameterError,
        ),
    ):
        return 500, {"message": SERVICE_MISCONFIGURATION_MESSAGE}
    if isinstance(error, ModelPackageRestrictedError):
        return 507, {"message": MODEL_RESTRICTED_MESSAGE}
    if isinstance(error, ModelPackageAlternativesExhaustedError):
        if any(
            isinstance(alternative_error, ModelPackageRestrictedError)
            for alternative_error in error.alternatives_errors or []
        ):
            return 507, {
                "message": MODEL_RESTRICTED_MESSAGE,
                "help_url": error.help_url,
            }
        return 500, {
            "message": f"Model loading failed: {error}",
            "help_url": error.help_url,
        }
    if isinstance(error, ModelLoadingError):
        return 500, {
            "message": f"Model loading failed: {error}",
            "help_url": error.help_url,
        }
    if isinstance(error, (UntrustedFileError, FileHashSumMissmatch)):
        return 500, {
            "message": f"Issue with model package file: {error}",
            "help_url": error.help_url,
        }
    if isinstance(error, ModelRetrievalError):
        access_answer = _model_access_answer(error)
        if access_answer is not None:
            return access_answer
        return 500, {
            "message": f"Could not retrieve model {error}",
            "help_url": error.help_url,
        }
    if isinstance(error, RuntimeError) and isinstance(
        cause, (RetryError, ModelRetrievalError, OSError)
    ):
        registry_answer = _registry_failure_answer(cause)
        return registry_answer
    return None


def _registry_failure_answer(cause: BaseException) -> Optional[Tuple[int, dict]]:
    access_answer = _model_access_answer(cause)
    if access_answer is not None:
        return access_answer
    if isinstance(cause, OSError):
        return _network_failure_answer(cause)
    if isinstance(cause, RetryError):
        if cause.__cause__ is not None:
            return _network_failure_answer(cause.__cause__)
        return 502, {"message": REGISTRY_REQUEST_FAILED_MESSAGE}
    if isinstance(cause.__cause__, (KeyError, pydantic.ValidationError)):
        return None
    return 502, {"message": REGISTRY_REQUEST_FAILED_MESSAGE}


def _network_failure_answer(error: BaseException) -> Optional[Tuple[int, dict]]:
    if isinstance(error, requests.exceptions.Timeout):
        return 504, {"message": REGISTRY_TIMEOUT_MESSAGE}
    if isinstance(error, (requests.exceptions.ConnectionError, ConnectionError)):
        return 503, {"message": REGISTRY_UNREACHABLE_MESSAGE}
    return None


def _model_access_answer(error: BaseException) -> Optional[Tuple[int, dict]]:
    status_code = getattr(error, "status_code", None)
    if status_code not in MODEL_ACCESS_ERROR_MESSAGES:
        return None

    return status_code, {"message": MODEL_ACCESS_ERROR_MESSAGES[status_code]}


def with_legacy_errors(fn: Callable) -> Callable:
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        token = REDACTION_VALUES.set([])
        try:
            with request_telemetry_scope():
                try:
                    try:
                        return await fn(*args, **kwargs)
                    except Exception as error:
                        record_route_error(error)
                        raise
                except HTTPException:
                    raise
                except Exception as error:
                    return legacy_error_response(error)
        finally:
            REDACTION_VALUES.reset(token)

    return wrapper


def install_legacy_exception_handlers(app: FastAPI) -> None:
    async def _handle(_request: Any, error: Exception) -> JSONResponse:
        return legacy_error_response(error)

    app.add_exception_handler(LegacyHTTPError, _handle)
    app.add_exception_handler(PayloadTooLargeError, _handle)
    app.add_exception_handler(BodyTooLargeHTTPException, _handle)


def _declared_content_length(scope: dict) -> Optional[int]:
    for name, value in scope.get("headers") or ():
        if name == b"content-length":
            raw = value.decode("latin-1").strip()
            if raw.isdigit():
                return int(raw)
            return None
    return None


async def _send_json(send: Callable, status_code: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


class _BodyLimitMiddleware:
    """Raw ASGI. For http scopes whose path does not start with /v2/:
    Content-Length > configuration.MAX_BODY_BYTES -> immediate 413
    {"message": "Request payload too large."}; otherwise wraps `receive` counting
    http.request bodies and raises BodyTooLargeHTTPException past the cap (same
    logic as framework/dispatch.py::_cap_request_body)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http" or scope.get("path", "").startswith("/v2/"):
            await self.app(scope, receive, send)
            return

        max_bytes = configuration.MAX_BODY_BYTES
        declared = _declared_content_length(scope)
        if declared is not None and declared > max_bytes:
            await _send_json(send, 413, {"message": PAYLOAD_TOO_LARGE_MESSAGE})
            return

        total = 0

        async def capped_receive():
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > max_bytes:
                    raise BodyTooLargeHTTPException(
                        f"request body exceeds {max_bytes} byte limit"
                    )
            return message

        await self.app(scope, capped_receive, send)
