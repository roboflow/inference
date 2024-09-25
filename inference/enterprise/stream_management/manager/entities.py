from enum import Enum

STATUS_KEY = "status"
TYPE_KEY = "type"
ERROR_TYPE_KEY = "error_type"
REQUEST_ID_KEY = "request_id"
PIPELINE_ID_KEY = "pipeline_id"
COMMAND_KEY = "command"
RESPONSE_KEY = "response"
ENCODING = "utf-8"


class OperationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class ErrorType(str, Enum):
    INTERNAL_ERROR = "internal_error"
    INVALID_PAYLOAD = "invalid_payload"
    NOT_FOUND = "not_found"
    OPERATION_ERROR = "operation_error"
    AUTHORISATION_ERROR = "authorisation_error"


class CommandType(str, Enum):
    INIT = "init"
    MUTE = "mute"
    RESUME = "resume"
    STATUS = "status"
    TERMINATE = "terminate"
    LIST_PIPELINES = "list_pipelines"
