import json
from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from inference.core.interfaces.stream_manager.manager_app.entities import (
    ENCODING,
    ERROR_TYPE_KEY,
    PIPELINE_ID_KEY,
    REQUEST_ID_KEY,
    RESPONSE_KEY,
    STATUS_KEY,
    ErrorType,
    OperationStatus,
)


def serialise_to_json(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if issubclass(type(obj), Enum):
        return obj.value
    raise TypeError(f"Type {type(obj)} not serializable")


def describe_error(
    exception: Optional[Exception] = None,
    error_type: ErrorType = ErrorType.INTERNAL_ERROR,
    public_error_message: Optional[str] = None,
) -> dict:
    payload = {
        STATUS_KEY: OperationStatus.FAILURE,
        ERROR_TYPE_KEY: error_type,
    }
    if exception is not None:
        payload["error_class"] = exception.__class__.__name__
        payload["error_message"] = str(exception)
    if public_error_message is not None:
        payload["public_error_message"] = public_error_message
    return payload


def prepare_error_response(
    request_id: str, error: Exception, error_type: ErrorType, pipeline_id: Optional[str]
) -> bytes:
    error_description = describe_error(exception=error, error_type=error_type)
    return prepare_response(
        request_id=request_id, response=error_description, pipeline_id=pipeline_id
    )


def prepare_response(
    request_id: str, response: dict, pipeline_id: Optional[str]
) -> bytes:
    payload = json.dumps(
        {
            REQUEST_ID_KEY: request_id,
            RESPONSE_KEY: response,
            PIPELINE_ID_KEY: pipeline_id,
        },
        default=serialise_to_json,
    )
    return payload.encode(ENCODING)
