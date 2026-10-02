from __future__ import annotations

import functools
import logging
from typing import Callable, Optional, Tuple

from fastapi.responses import JSONResponse
from roboflow_workflows.errors import WorkflowsInvalidEnvironmentValueError
from roboflow_workflows.http_contract.errors import workflow_error_payload
from roboflow_workflows.prototypes.platform_errors import (
    FeatureDeprecatedError,
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIRequestError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)
from starlette.exceptions import HTTPException

from inference_server.legacy.errors import (
    MODEL_ACCESS_ERROR_MESSAGES,
    NOT_FOUND_MESSAGE,
    REGISTRY_REQUEST_FAILED_MESSAGE,
    REGISTRY_TIMEOUT_MESSAGE,
    REGISTRY_UNREACHABLE_MESSAGE,
    SERVICE_MISCONFIGURATION_MESSAGE,
    UNAUTHORIZED_MESSAGE,
    legacy_error_response,
)

logger = logging.getLogger(__name__)


class PaymentRequiredError(RoboflowAPIUnsuccessfulRequestError):
    pass


class RoboflowAPIUsagePausedError(RoboflowAPIUnsuccessfulRequestError):
    pass


class MalformedRoboflowAPIResponseError(RoboflowAPIRequestError):
    pass


class WorkspaceLoadError(RoboflowAPIRequestError):
    pass


class ModelDeploymentNotSupportedError(Exception):
    pass


def with_workflow_errors(fn: Callable) -> Callable:
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as error:
            payload = workflow_error_payload(error) or _platform_error_payload(error)
            if payload is None:
                return legacy_error_response(error)

            status_code, content = payload
            if isinstance(error, FeatureDeprecatedError):
                logger.warning("%s: %s", type(error).__name__, error)
            else:
                logger.error("%s: %s", type(error).__name__, error, exc_info=error)

            return JSONResponse(status_code=status_code, content=content)

    return wrapper


def _platform_error_payload(error: BaseException) -> Optional[Tuple[int, dict]]:
    if isinstance(error, RoboflowAPINotAuthorizedError):
        return 401, {"message": UNAUTHORIZED_MESSAGE}
    if isinstance(error, PaymentRequiredError):
        return 402, {"message": MODEL_ACCESS_ERROR_MESSAGES[402]}
    if isinstance(error, RoboflowAPIForbiddenError):
        return 403, {"message": MODEL_ACCESS_ERROR_MESSAGES[403]}
    if isinstance(error, RoboflowAPIUsagePausedError):
        return 423, {"message": MODEL_ACCESS_ERROR_MESSAGES[423]}
    if isinstance(error, RoboflowAPINotNotFoundError):
        return 404, {"message": NOT_FOUND_MESSAGE}
    if isinstance(error, WorkflowsInvalidEnvironmentValueError):
        return 500, {"message": SERVICE_MISCONFIGURATION_MESSAGE}
    if isinstance(
        error,
        (
            MalformedRoboflowAPIResponseError,
            RoboflowAPIUnsuccessfulRequestError,
            WorkspaceLoadError,
        ),
    ):
        return 502, {"message": REGISTRY_REQUEST_FAILED_MESSAGE}
    if isinstance(error, RoboflowAPIConnectionError):
        return 503, {"message": REGISTRY_UNREACHABLE_MESSAGE}
    if isinstance(error, RoboflowAPITimeoutError):
        return 504, {"message": REGISTRY_TIMEOUT_MESSAGE}
    return None
