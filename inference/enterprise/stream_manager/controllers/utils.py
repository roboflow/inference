from typing import Optional

from inference.enterprise.stream_manager.controllers.entities import (
    ERROR_TYPE_KEY,
    STATUS_KEY,
    ErrorType,
    OperationStatus,
)


def describe_error(
    exception: Optional[Exception] = None,
    error_type: ErrorType = ErrorType.INTERNAL_ERROR,
) -> dict:
    payload = {
        STATUS_KEY: OperationStatus.FAILURE,
        ERROR_TYPE_KEY: error_type,
    }
    if exception is not None:
        payload["error_class"] = exception.__class__.__name__
        payload["error_message"] = str(exception)
    return payload
