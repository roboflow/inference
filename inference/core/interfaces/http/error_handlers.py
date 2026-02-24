from functools import wraps

from starlette.responses import JSONResponse

from inference.core import logger
from inference.core.entities.responses.workflows import WorkflowErrorResponse
from inference.core.exceptions import (
    ContentTypeInvalid,
    ContentTypeMissing,
    CreditsExceededError,
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
    ModelManagerLockAcquisitionError,
    OnnxProviderNotAvailable,
    PostProcessingError,
    PreProcessingError,
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
    ServiceConfigurationError,
    WebRTCConfigurationError,
    WorkspaceLoadError,
    WorkspaceStreamQuotaError,
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
    ClientCausedStepExecutionError,
    DynamicBlockError,
    ExecutionGraphStructureError,
    InvalidReferenceTargetError,
    NotSupportedExecutionEngineError,
    ReferenceTypeError,
    RuntimeInputError,
    StepExecutionError,
    StepInputDimensionalityError,
    WorkflowBlockError,
    WorkflowDefinitionError,
    WorkflowError,
    WorkflowExecutionEngineVersionError,
    WorkflowSyntaxError,
)
from inference_models.errors import (
    EnvironmentConfigurationError,
    FileHashSumMissmatch,
    InvalidEnvVariable,
    InvalidParameterError,
    JetsonTypeResolutionError,
    MissingDependencyError,
    ModelInputError,
    ModelLoadingError,
    ModelNotFoundError,
    ModelPackageNegotiationError,
    ModelRetrievalError,
    UnauthorizedModelAccessError,
    UntrustedFileError,
)


def with_route_exceptions(route):
    """
    A decorator that wraps a FastAPI route to handle specific exceptions. If an exception
    is caught, it returns a JSON response with the error message.

    Args:
        route (Callable): The FastAPI route to be wrapped.

    Returns:
        Callable: The wrapped route.
    """

    @wraps(route)
    def wrapped_route(*args, **kwargs):
        try:
            return route(*args, **kwargs)
        except ContentTypeInvalid as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Invalid Content-Type header provided with request."
                },
            )
        except ContentTypeMissing as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={"message": "Content-Type header not provided with request."},
            )
        except InputImageLoadError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": f"Could not load input image. Cause: {error.get_public_error_details()}"
                },
            )
        except ModelInputError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": f"Error with model input. Cause: {error}",
                    "help_url": error.help_url,
                },
            )
        except InvalidModelIDError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={"message": "Invalid Model ID sent in request."},
            )
        except InvalidMaskDecodeArgument as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Invalid mask decode argument sent. tradeoff_factor must be in [0.0, 1.0], "
                    "mask_decode_mode: must be one of ['accurate', 'fast', 'tradeoff']"
                },
            )
        except MissingApiKeyError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Required Roboflow API key is missing. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                    "to learn how to retrieve one."
                },
            )
        except (
            WorkflowSyntaxError,
            InvalidReferenceTargetError,
            ExecutionGraphStructureError,
            StepInputDimensionalityError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            content = WorkflowErrorResponse(
                message=str(error.public_message),
                error_type=error.__class__.__name__,
                context=str(error.context),
                inner_error_type=str(error.inner_error_type),
                inner_error_message=str(error.inner_error),
                blocks_errors=error.blocks_errors,
            )
            resp = JSONResponse(status_code=400, content=content.model_dump())
        except (
            WorkflowDefinitionError,
            ReferenceTypeError,
            RuntimeInputError,
            InvalidInputTypeError,
            OperationTypeNotRecognisedError,
            DynamicBlockError,
            WorkflowExecutionEngineVersionError,
            NotSupportedExecutionEngineError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "context": error.context,
                    "inner_error_type": error.inner_error_type,
                    "inner_error_message": str(error.inner_error),
                },
            )
        except (
            ProcessesManagerInvalidPayload,
            MalformedPayloadError,
            MessageToBigError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
        except (
            RoboflowAPINotAuthorizedError,
            ProcessesManagerAuthorisationError,
            UnauthorizedModelAccessError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=401,
                content={
                    "message": "Unauthorized access to roboflow API - check API key and make sure the key is valid for "
                    "workspace you use. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                    "to learn how to retrieve one."
                },
            )
        except RoboflowAPIForbiddenError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=403,
                content={
                    "message": "Unauthorized access to roboflow API - check API key and make sure the key is valid and "
                    "have required scopes. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                    "to learn how to retrieve one."
                },
            )
        except (RoboflowAPINotNotFoundError, ModelNotFoundError) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=404,
                content={
                    "message": "Requested Roboflow resource not found. Make sure that workspace, project or model "
                    "you referred in request exists."
                },
            )
        except ProcessesManagerNotFoundError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=404,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
        except ModelPackageNegotiationError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": f"Could not negotiate model package - {error}",
                    "help_url": error.help_url,
                },
            )
        except (
            InvalidEnvironmentVariableError,
            MissingServiceSecretError,
            ServiceConfigurationError,
            EnvironmentConfigurationError,
            InvalidEnvVariable,
            JetsonTypeResolutionError,
            MissingDependencyError,
            InvalidParameterError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500, content={"message": "Service misconfiguration."}
            )
        except (
            PreProcessingError,
            PostProcessingError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": "Model configuration related to pre- or post-processing is invalid."
                },
            )
        except ModelArtefactError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500, content={"message": "Model package is broken."}
            )
        except ModelLoadingError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": f"Model loading failed: {error}",
                    "help_url": error.help_url,
                },
            )
        except (UntrustedFileError, FileHashSumMissmatch) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": f"Issue with model package file: {error}",
                    "help_url": error.help_url,
                },
            )
        except ModelRetrievalError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": f"Could not retrieve model {error}",
                    "help_url": error.help_url,
                },
            )
        except OnnxProviderNotAvailable as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=501,
                content={
                    "message": "Could not find requested ONNX Runtime Provider. Check that you are using "
                    "the correct docker image on a supported device."
                },
            )
        except (
            MalformedRoboflowAPIResponseError,
            RoboflowAPIUnsuccessfulRequestError,
            WorkspaceLoadError,
            MalformedWorkflowResponseError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=502,
                content={"message": "Internal error. Request to Roboflow API failed."},
            )
        except InferenceModelNotFound as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=503,
                content={"message": "Model is temporarily not ready - retry request."},
                headers={"Retry-After": "1"},
            )
        except RoboflowAPIConnectionError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=503,
                content={
                    "message": "Internal error. Could not connect to Roboflow API."
                },
            )
        except ModelManagerLockAcquisitionError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=503,
                content={
                    "message": "Could not acquire model manager lock due to other request performing "
                    "blocking operation. Try again...."
                },
                headers={"Retry-After": "1"},
            )
        except RoboflowAPITimeoutError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=504,
                content={
                    "message": "Timeout when attempting to connect to Roboflow API."
                },
            )
        except ClientCausedStepExecutionError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            content = WorkflowErrorResponse(
                message=str(error.public_message),
                error_type=error.__class__.__name__,
                context=str(error.context),
                inner_error_type=str(error.inner_error_type),
                inner_error_message=str(error.inner_error),
                blocks_errors=[
                    WorkflowBlockError(
                        block_id=error.block_id,
                    ),
                ],
            )
            resp = JSONResponse(
                status_code=error.status_code,
                content=content.model_dump(),
            )
        except StepExecutionError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            content = WorkflowErrorResponse(
                message=str(error.public_message),
                error_type=error.__class__.__name__,
                context=str(error.context),
                inner_error_type=str(error.inner_error_type),
                inner_error_message=str(error.inner_error),
                blocks_errors=[
                    WorkflowBlockError(
                        block_id=error.block_id,
                        block_type=error.block_type,
                    ),
                ],
            )
            resp = JSONResponse(
                status_code=500,
                content=content.model_dump(),
            )
        except WorkflowError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "context": error.context,
                    "inner_error_type": error.inner_error_type,
                    "inner_error_message": str(error.inner_error),
                },
            )
        except (
            ProcessesManagerClientError,
            CommunicationProtocolError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
        except WebRTCConfigurationError as error:
            logger.error("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": str(error),
                    "error_type": "WebRTCConfigurationError",
                },
            )
        except CreditsExceededError as error:
            logger.error("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=402,
                content={
                    "message": "Not enough credits to perform this request.",
                    "error_type": "CreditsExceededError",
                },
            )
        except WorkspaceStreamQuotaError as error:
            logger.error("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=429,
                content={
                    "message": str(error),
                    "error_type": "WorkspaceStreamQuotaError",
                },
            )
        except Exception as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(status_code=500, content={"message": "Internal error."})
        return resp

    return wrapped_route


def with_route_exceptions_async(route):
    """
    A decorator that wraps a FastAPI route to handle specific exceptions. If an exception
    is caught, it returns a JSON response with the error message.

    Args:
        route (Callable): The FastAPI route to be wrapped.

    Returns:
        Callable: The wrapped route.
    """

    @wraps(route)
    async def wrapped_route(*args, **kwargs):
        try:
            return await route(*args, **kwargs)
        except ContentTypeInvalid as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Invalid Content-Type header provided with request."
                },
            )
        except ContentTypeMissing as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={"message": "Content-Type header not provided with request."},
            )
        except InputImageLoadError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": f"Could not load input image. Cause: {error.get_public_error_details()}"
                },
            )
        except ModelInputError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": f"Error with model input. Cause: {error}",
                    "help_url": error.help_url,
                },
            )
        except InvalidModelIDError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={"message": "Invalid Model ID sent in request."},
            )
        except InvalidMaskDecodeArgument as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Invalid mask decode argument sent. tradeoff_factor must be in [0.0, 1.0], "
                    "mask_decode_mode: must be one of ['accurate', 'fast', 'tradeoff']"
                },
            )
        except MissingApiKeyError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Required Roboflow API key is missing. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                    "to learn how to retrieve one."
                },
            )
        except (
            WorkflowSyntaxError,
            InvalidReferenceTargetError,
            ExecutionGraphStructureError,
            StepInputDimensionalityError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            content = WorkflowErrorResponse(
                message=str(error.public_message),
                error_type=error.__class__.__name__,
                context=str(error.context),
                inner_error_type=str(error.inner_error_type),
                inner_error_message=str(error.inner_error),
                blocks_errors=error.blocks_errors,
            )
            resp = JSONResponse(status_code=400, content=content.model_dump())
        except (
            WorkflowDefinitionError,
            ReferenceTypeError,
            RuntimeInputError,
            InvalidInputTypeError,
            OperationTypeNotRecognisedError,
            DynamicBlockError,
            WorkflowExecutionEngineVersionError,
            NotSupportedExecutionEngineError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "context": error.context,
                    "inner_error_type": error.inner_error_type,
                    "inner_error_message": str(error.inner_error),
                },
            )
        except (
            ProcessesManagerInvalidPayload,
            MalformedPayloadError,
            MessageToBigError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
        except (
            RoboflowAPINotAuthorizedError,
            ProcessesManagerAuthorisationError,
            UnauthorizedModelAccessError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=401,
                content={
                    "message": "Unauthorized access to roboflow API - check API key and make sure the key is valid for "
                    "workspace you use. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                    "to learn how to retrieve one."
                },
            )
        except RoboflowAPIForbiddenError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=403,
                content={
                    "message": "Unauthorized access to roboflow API - check API key and make sure the key is valid and "
                    "have required scopes. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                    "to learn how to retrieve one."
                },
            )
        except (RoboflowAPINotNotFoundError, ModelNotFoundError) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=404,
                content={
                    "message": "Requested Roboflow resource not found. Make sure that workspace, project or model "
                    "you referred in request exists."
                },
            )
        except ProcessesManagerNotFoundError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=404,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
        except ModelPackageNegotiationError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": f"Could not negotiate model package - {error}",
                    "help_url": error.help_url,
                },
            )
        except (
            InvalidEnvironmentVariableError,
            MissingServiceSecretError,
            ServiceConfigurationError,
            EnvironmentConfigurationError,
            InvalidEnvVariable,
            JetsonTypeResolutionError,
            MissingDependencyError,
            InvalidParameterError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500, content={"message": "Service misconfiguration."}
            )
        except (
            PreProcessingError,
            PostProcessingError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": "Model configuration related to pre- or post-processing is invalid."
                },
            )
        except ModelArtefactError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500, content={"message": "Model package is broken."}
            )
        except ModelLoadingError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": f"Model loading failed: {error}",
                    "help_url": error.help_url,
                },
            )
        except (UntrustedFileError, FileHashSumMissmatch) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": f"Issue with model package file: {error}",
                    "help_url": error.help_url,
                },
            )
        except ModelRetrievalError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": f"Could not retrieve model {error}",
                    "help_url": error.help_url,
                },
            )
        except OnnxProviderNotAvailable as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=501,
                content={
                    "message": "Could not find requested ONNX Runtime Provider. Check that you are using "
                    "the correct docker image on a supported device."
                },
            )
        except (
            MalformedRoboflowAPIResponseError,
            RoboflowAPIUnsuccessfulRequestError,
            WorkspaceLoadError,
            MalformedWorkflowResponseError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=502,
                content={"message": "Internal error. Request to Roboflow API failed."},
            )
        except InferenceModelNotFound as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=503,
                content={"message": "Model is temporarily not ready - retry request."},
                headers={"Retry-After": "1"},
            )
        except RoboflowAPIConnectionError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=503,
                content={
                    "message": "Internal error. Could not connect to Roboflow API."
                },
            )
        except ModelManagerLockAcquisitionError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=503,
                content={
                    "message": "Could not acquire model manager lock due to other request performing "
                    "blocking operation. Try again...."
                },
                headers={"Retry-After": "1"},
            )
        except RoboflowAPITimeoutError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=504,
                content={
                    "message": "Timeout when attempting to connect to Roboflow API."
                },
            )
        except ClientCausedStepExecutionError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            content = WorkflowErrorResponse(
                message=str(error.public_message),
                error_type=error.__class__.__name__,
                context=str(error.context),
                inner_error_type=str(error.inner_error_type),
                inner_error_message=str(error.inner_error),
                blocks_errors=[
                    WorkflowBlockError(
                        block_id=error.block_id,
                    ),
                ],
            )
            resp = JSONResponse(
                status_code=error.status_code,
                content=content.model_dump(),
            )
        except StepExecutionError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            content = WorkflowErrorResponse(
                message=str(error.public_message),
                error_type=error.__class__.__name__,
                context=str(error.context),
                inner_error_type=str(error.inner_error_type),
                inner_error_message=str(error.inner_error),
                blocks_errors=[
                    WorkflowBlockError(
                        block_id=error.block_id,
                        block_type=error.block_type,
                    ),
                ],
            )
            resp = JSONResponse(
                status_code=500,
                content=content.model_dump(),
            )
        except WorkflowError as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "context": error.context,
                    "inner_error_type": error.inner_error_type,
                    "inner_error_message": str(error.inner_error),
                },
            )
        except (
            ProcessesManagerClientError,
            CommunicationProtocolError,
        ) as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
        except WebRTCConfigurationError as error:
            logger.error("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": str(error),
                    "error_type": "WebRTCConfigurationError",
                },
            )
        except CreditsExceededError as error:
            logger.error("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=402,
                content={
                    "message": "Not enough credits to perform this request.",
                    "error_type": "CreditsExceededError",
                },
            )
        except WorkspaceStreamQuotaError as error:
            logger.error("%s: %s", type(error).__name__, error)
            resp = JSONResponse(
                status_code=429,
                content={
                    "message": str(error),
                    "error_type": "WorkspaceStreamQuotaError",
                },
            )
        except Exception as error:
            logger.exception("%s: %s", type(error).__name__, error)
            resp = JSONResponse(status_code=500, content={"message": "Internal error."})
        return resp

    return wrapped_route
