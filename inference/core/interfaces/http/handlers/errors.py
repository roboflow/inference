import traceback
from typing import Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from inference.core.exceptions import ContentTypeInvalid, ContentTypeMissing, InputImageLoadError, InvalidModelIDError, \
    InvalidMaskDecodeArgument, MissingApiKeyError, RoboflowAPINotAuthorizedError, RoboflowAPINotNotFoundError, \
    InferenceModelNotFound, InvalidEnvironmentVariableError, MissingServiceSecretError, ServiceConfigurationError, \
    PostProcessingError, PreProcessingError, ModelArtefactError, OnnxProviderNotAvailable
from inference.core.interfaces.stream_manager.api.errors import ProcessesManagerInvalidPayload, \
    ProcessesManagerAuthorisationError, ProcessesManagerNotFoundError
from inference.core.interfaces.stream_manager.manager_app.errors import MalformedPayloadError, MessageToBigError
from inference.core.workflows.core_steps.common.query_language.errors import InvalidInputTypeError, \
    OperationTypeNotRecognisedError
from inference.core.workflows.errors import WorkflowDefinitionError, ExecutionGraphStructureError, ReferenceTypeError, \
    InvalidReferenceTargetError, RuntimeInputError, DynamicBlockError, WorkflowExecutionEngineVersionError, \
    NotSupportedExecutionEngineError


def setup_error_handling(app: FastAPI) -> None:
    pass


def handle_content_type_invalid(request: Request, exc: ContentTypeInvalid) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "message": "Invalid Content-Type header provided with request."
        },
    )


def handle_content_type_missing(request: Request, exc: ContentTypeMissing) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"message": "Content-Type header not provided with request."},
    )


def handle_input_image_load_error(request: Request, exc: InputImageLoadError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "message": f"Could not load input image. Cause: {e.get_public_error_details()}"
        },
    )


def handle_invalid_model_id_error(request: Request, exc: InvalidModelIDError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"message": "Invalid Model ID sent in request."},
    )


def handle_invalid_mask_decode_argument(request: Request, exc: InvalidMaskDecodeArgument) -> JSONResponse:
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


def handle_workflows_bad_request(request: Request, exc: WorkflowBadRequestError) -> JSONResponse:
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


def handle_video_management_api_bad_request(request: Request, exc: VideoManagementAPIBadRequestError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "message": exc.public_message,
            "error_type": exc.__class__.__name__,
            "inner_error_type": exc.inner_error_type,
        },
    )


def handle_not_authorised_error(request: Request, exc: Union[RoboflowAPINotAuthorizedError, ProcessesManagerAuthorisationError]) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "message": "Unauthorized access to roboflow API - check API key and make sure the key is valid for "
                       "workspace you use. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                       "to learn how to retrieve one."
        },
    )


def handle_not_found_error(request: Request, exc: Union[RoboflowAPINotNotFoundError, InferenceModelNotFound]) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "message": "Requested Roboflow resource not found. Make sure that workspace, project or model "
            "you referred in request exists."
        },
    )


def handle_video_management_api_not_found(request: Request, exc: ProcessesManagerNotFoundError) -> JSONResponse:
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
    exc: Union[InvalidEnvironmentVariableError, MissingServiceSecretError, ServiceConfigurationError],
) -> JSONResponse:
    traceback.print_exc()
    return JSONResponse(
        status_code=500, content={"message": "Service misconfiguration."}
    )


def handle_model_processing_error(request: Request, exc: Union[PreProcessingError, PostProcessingError]) -> JSONResponse:
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "message": "Model configuration related to pre- or post-processing is invalid."
        },
    )


def handle_model_artefact_error(request: Request, exc: ModelArtefactError) -> JSONResponse:
    traceback.print_exc()
    return JSONResponse(
        status_code=500, content={"message": "Model package is broken."}
    )


def handle_onnx_provider_error(request: Request, exc: OnnxProviderNotAvailable) -> JSONResponse:
    pass
