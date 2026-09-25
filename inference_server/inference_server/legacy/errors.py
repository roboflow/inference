from __future__ import annotations

import asyncio
import functools
import json
import logging
from typing import Any, Callable, Optional

import pydantic
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from inference_models.errors import (
    ModelInputError,
    ModelNotFoundError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    RetryError,
    UnauthorizedModelAccessError,
)
from inference_server import configuration
from inference_server.errors import PayloadTooLargeError, ServerBusyError

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
INFERENCE_TIMEOUT_MESSAGE = "Timed out waiting for inference result."
INTERNAL_ERROR_MESSAGE = "Internal error."

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
        self, status_code: int, message: str, extra: Optional[dict] = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.extra = extra or {}


def legacy_error_response(error: BaseException) -> JSONResponse:
    if isinstance(error, LegacyHTTPError):
        return JSONResponse(
            status_code=error.status_code,
            content={"message": error.message, **error.extra},
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
    if isinstance(error, (PermissionError, UnauthorizedModelAccessError)):
        return JSONResponse(status_code=401, content={"message": UNAUTHORIZED_MESSAGE})
    if isinstance(error, (LookupError, ModelNotFoundError)):
        return JSONResponse(status_code=404, content={"message": NOT_FOUND_MESSAGE})
    if isinstance(error, ModelPackageRestrictedError):
        return JSONResponse(
            status_code=507, content={"message": MODEL_RESTRICTED_MESSAGE}
        )
    if isinstance(error, ModelRetrievalError):
        access_response = _model_access_response(error)
        if access_response is not None:
            return access_response
    if isinstance(error, (ModelInputError, ValueError, pydantic.ValidationError)):
        return JSONResponse(status_code=400, content={"message": str(error)})
    if isinstance(error, asyncio.TimeoutError):
        return JSONResponse(
            status_code=504, content={"message": INFERENCE_TIMEOUT_MESSAGE}
        )
    if isinstance(error, RuntimeError):
        cause = error.__cause__
        if isinstance(cause, (RetryError, ModelRetrievalError, OSError)):
            access_response = _model_access_response(cause)
            if access_response is not None:
                return access_response
            return JSONResponse(
                status_code=503, content={"message": REGISTRY_UNREACHABLE_MESSAGE}
            )
    logger.error("Unhandled legacy route error", exc_info=error)
    return JSONResponse(status_code=500, content={"message": INTERNAL_ERROR_MESSAGE})


def _model_access_response(error: BaseException) -> Optional[JSONResponse]:
    status_code = getattr(error, "status_code", None)
    if status_code not in MODEL_ACCESS_ERROR_MESSAGES:
        return None
    return JSONResponse(
        status_code=status_code,
        content={"message": MODEL_ACCESS_ERROR_MESSAGES[status_code]},
    )


def with_legacy_errors(fn: Callable) -> Callable:
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as error:
            return legacy_error_response(error)

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
