import traceback
from typing import Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from inference.core import logger
from inference.core.exceptions import (
    ContentTypeInvalid,
    ContentTypeMissing,
    InferenceModelNotFound,
    InputImageLoadError,
    InvalidEnvironmentVariableError,
    InvalidMaskDecodeArgument,
    InvalidModelIDError,
    MalformedRoboflowAPIResponseError,
    MalformedWorkflowResponseError,
    MissingApiKeyError,
    MissingServiceSecretError,
    ModelArtefactError,
    OnnxProviderNotAvailable,
    PostProcessingError,
    PreProcessingError,
    RoboflowAPIConnectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIUnsuccessfulRequestError,
    ServiceConfigurationError,
    WorkspaceLoadError,
)
from inference.core.interfaces.stream_manager.api.errors import (
    ProcessesManagerAuthorisationError,
    ProcessesManagerClientError,
    ProcessesManagerInvalidPayload,
    ProcessesManagerNotFoundError,
)
from inference.core.interfaces.stream_manager.manager_app.errors import (
    CommunicationProtocolError,
    MalformedPayloadError,
    MessageToBigError,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationTypeNotRecognisedError,
)
from inference.core.workflows.errors import (
    DynamicBlockError,
    ExecutionGraphStructureError,
    InvalidReferenceTargetError,
    NotSupportedExecutionEngineError,
    ReferenceTypeError,
    RuntimeInputError,
    WorkflowDefinitionError,
    WorkflowError,
    WorkflowExecutionEngineVersionError,
)


def setup_error_handling(app: FastAPI) -> None:
    for exception, handler in ERROR_TO_HANDLER_MAPPING:
        app.add_exception_handler(exc_class_or_status_code=exception, handler=handler)


def handle_content_type_invalid(
    request: Request, exc: ContentTypeInvalid
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"message": "Invalid Content-Type header provided with request."},
    )


def handle_content_type_missing(
    request: Request, exc: ContentTypeMissing
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"message": "Content-Type header not provided with request."},
    )


def handle_input_image_load_error(
    request: Request, exc: InputImageLoadError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "message": f"Could not load input image. Cause: {exc.get_public_error_details()}"
        },
    )


def handle_invalid_model_id_error(
    request: Request, exc: InvalidModelIDError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"message": "Invalid Model ID sent in request."},
    )


def handle_invalid_mask_decode_argument(
    request: Request, exc: InvalidMaskDecodeArgument
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "message": "Invalid mask decode argument sent. tradeoff_factor must be in [0.0, 1.0], "
            "mask_decode_mode: must be one of ['accurate', 'fast', 'tradeoff']"
        },
    )


def handle_missing_api_key(request: Request, exc: MissingApiKeyError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "message": "Required Roboflow API key is missing. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
            "to learn how to retrieve one."
        },
    )


WorkflowBadRequestError = Union[
    WorkflowDefinitionError,
    ExecutionGraphStructureError,
    ReferenceTypeError,
    InvalidReferenceTargetError,
    RuntimeInputError,
    InvalidInputTypeError,
    OperationTypeNotRecognisedError,
    DynamicBlockError,
    WorkflowExecutionEngineVersionError,
    NotSupportedExecutionEngineError,
]


def handle_workflows_bad_request(
    request: Request, exc: WorkflowBadRequestError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "message": exc.public_message,
            "error_type": exc.__class__.__name__,
            "context": exc.context,
            "inner_error_type": exc.inner_error_type,
            "inner_error_message": str(exc.inner_error),
        },
    )


VideoManagementAPIBadRequestError = Union[
    ProcessesManagerInvalidPayload,
    MalformedPayloadError,
    MessageToBigError,
]


def handle_video_management_api_bad_request(
    request: Request, exc: VideoManagementAPIBadRequestError
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "message": exc.public_message,
            "error_type": exc.__class__.__name__,
            "inner_error_type": exc.inner_error_type,
        },
    )


def handle_not_authorised_error(
    request: Request,
    exc: Union[RoboflowAPINotAuthorizedError, ProcessesManagerAuthorisationError],
) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "message": "Unauthorized access to roboflow API - check API key and make sure the key is valid for "
            "workspace you use. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
            "to learn how to retrieve one."
        },
    )


def handle_not_found_error(
    request: Request, exc: Union[RoboflowAPINotNotFoundError, InferenceModelNotFound]
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "message": "Requested Roboflow resource not found. Make sure that workspace, project or model "
            "you referred in request exists."
        },
    )


def handle_video_management_api_not_found(
    request: Request, exc: ProcessesManagerNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "message": exc.public_message,
            "error_type": exc.__class__.__name__,
            "inner_error_type": exc.inner_error_type,
        },
    )


def handle_service_miss_configuration_error(
    request: Request,
    exc: Union[
        InvalidEnvironmentVariableError,
        MissingServiceSecretError,
        ServiceConfigurationError,
    ],
) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(
        status_code=500, content={"message": "Service misconfiguration."}
    )


def handle_model_processing_error(
    request: Request, exc: Union[PreProcessingError, PostProcessingError]
) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(
        status_code=500,
        content={
            "message": "Model configuration related to pre- or post-processing is invalid."
        },
    )


def handle_model_artefact_error(
    request: Request, exc: ModelArtefactError
) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(
        status_code=500, content={"message": "Model package is broken."}
    )


def handle_onnx_provider_error(
    request: Request, exc: OnnxProviderNotAvailable
) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(
        status_code=501,
        content={
            "message": "Could not find requested ONNX Runtime Provider. Check that you are using "
            "the correct docker image on a supported device."
        },
    )


def handle_roboflow_api_request_fail_error(
    request: Request,
    exc: Union[
        MalformedRoboflowAPIResponseError,
        RoboflowAPIUnsuccessfulRequestError,
        WorkspaceLoadError,
        MalformedWorkflowResponseError,
    ],
) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(
        status_code=502,
        content={"message": "Internal error. Request to Roboflow API failed."},
    )


def handle_roboflow_api_connection_error(
    request: Request, exc: RoboflowAPIConnectionError
) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(
        status_code=503,
        content={"message": "Internal error. Could not connect to Roboflow API."},
    )


def handle_workflow_error(request: Request, exc: WorkflowError) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(
        status_code=500,
        content={
            "message": exc.public_message,
            "error_type": exc.__class__.__name__,
            "context": exc.context,
            "inner_error_type": exc.inner_error_type,
            "inner_error_message": str(exc.inner_error),
        },
    )


def handle_video_management_api_internal_error(
    request: Request,
    exc: Union[ProcessesManagerClientError, CommunicationProtocolError],
) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(
        status_code=500,
        content={
            "message": exc.public_message,
            "error_type": exc.__class__.__name__,
            "inner_error_type": exc.inner_error_type,
        },
    )


def handle_unknown_error(request: Request, exc: Exception) -> JSONResponse:
    _log_error_message(exception=exc)
    return JSONResponse(status_code=500, content={"message": "Internal error."})


def _log_error_message(exception: Exception) -> None:
    logger.error(exception)
    traceback.print_tb(tb=exception.__traceback__)


ERROR_TO_HANDLER_MAPPING = [
    (ContentTypeInvalid, handle_content_type_invalid),
    (ContentTypeMissing, handle_content_type_missing),
    (InputImageLoadError, handle_input_image_load_error),
    (InvalidModelIDError, handle_invalid_model_id_error),
    (InvalidMaskDecodeArgument, handle_invalid_mask_decode_argument),
    (MissingApiKeyError, handle_missing_api_key),
    (WorkflowDefinitionError, handle_workflows_bad_request),
    (ExecutionGraphStructureError, handle_workflows_bad_request),
    (ReferenceTypeError, handle_workflows_bad_request),
    (InvalidReferenceTargetError, handle_workflows_bad_request),
    (RuntimeInputError, handle_workflows_bad_request),
    (InvalidInputTypeError, handle_workflows_bad_request),
    (OperationTypeNotRecognisedError, handle_workflows_bad_request),
    (DynamicBlockError, handle_workflows_bad_request),
    (WorkflowExecutionEngineVersionError, handle_workflows_bad_request),
    (NotSupportedExecutionEngineError, handle_workflows_bad_request),
    (ProcessesManagerInvalidPayload, handle_video_management_api_bad_request),
    (MalformedPayloadError, handle_video_management_api_bad_request),
    (MessageToBigError, handle_video_management_api_bad_request),
    (RoboflowAPINotAuthorizedError, handle_not_authorised_error),
    (ProcessesManagerAuthorisationError, handle_not_authorised_error),
    (RoboflowAPINotNotFoundError, handle_not_found_error),
    (InferenceModelNotFound, handle_not_found_error),
    (ProcessesManagerNotFoundError, handle_video_management_api_not_found),
    (InvalidEnvironmentVariableError, handle_service_miss_configuration_error),
    (MissingServiceSecretError, handle_service_miss_configuration_error),
    (ServiceConfigurationError, handle_service_miss_configuration_error),
    (PreProcessingError, handle_model_processing_error),
    (PostProcessingError, handle_model_processing_error),
    (ModelArtefactError, handle_model_artefact_error),
    (OnnxProviderNotAvailable, handle_onnx_provider_error),
    (MalformedRoboflowAPIResponseError, handle_roboflow_api_request_fail_error),
    (RoboflowAPIUnsuccessfulRequestError, handle_roboflow_api_request_fail_error),
    (WorkspaceLoadError, handle_roboflow_api_request_fail_error),
    (MalformedWorkflowResponseError, handle_roboflow_api_request_fail_error),
    (RoboflowAPIConnectionError, handle_roboflow_api_connection_error),
    (WorkflowError, handle_workflow_error),
    (ProcessesManagerClientError, handle_video_management_api_internal_error),
    (CommunicationProtocolError, handle_video_management_api_internal_error),
    (Exception, handle_unknown_error),
]
