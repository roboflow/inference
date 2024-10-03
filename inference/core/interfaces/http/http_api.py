import base64
import os
import traceback
from functools import partial, wraps
from time import sleep
from typing import Any, Dict, List, Optional, Union

import asgi_correlation_id
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi_cprofile.profiler import CProfileMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.convertors import StringConvertor, register_url_convertor
from starlette.middleware.base import BaseHTTPMiddleware

from inference.core import logger
from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.entities.requests.clip import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
)
from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.gaze import GazeDetectionInferenceRequest
from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    InferenceRequest,
    InferenceRequestImage,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    LMMInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from inference.core.entities.requests.sam import (
    SamEmbeddingRequest,
    SamSegmentationRequest,
)
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2SegmentationRequest,
)
from inference.core.entities.requests.server_state import (
    AddModelRequest,
    ClearModelRequest,
)
from inference.core.entities.requests.trocr import TrOCRInferenceRequest
from inference.core.entities.requests.workflows import (
    DescribeBlocksRequest,
    DescribeInterfaceRequest,
    PredefinedWorkflowDescribeInterfaceRequest,
    PredefinedWorkflowInferenceRequest,
    WorkflowInferenceRequest,
    WorkflowSpecificationDescribeInterfaceRequest,
    WorkflowSpecificationInferenceRequest,
)
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.entities.responses.clip import (
    ClipCompareResponse,
    ClipEmbeddingResponse,
)
from inference.core.entities.responses.cogvlm import CogVLMResponse
from inference.core.entities.responses.gaze import GazeDetectionInferenceResponse
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    LMMInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    StubResponse,
)
from inference.core.entities.responses.notebooks import NotebookStartResponse
from inference.core.entities.responses.ocr import OCRInferenceResponse
from inference.core.entities.responses.sam import (
    SamEmbeddingResponse,
    SamSegmentationResponse,
)
from inference.core.entities.responses.sam2 import (
    Sam2EmbeddingResponse,
    Sam2SegmentationResponse,
)
from inference.core.entities.responses.server_state import (
    ModelsDescriptions,
    ServerVersionInfo,
)
from inference.core.entities.responses.workflows import (
    DescribeInterfaceResponse,
    ExecutionEngineVersions,
    WorkflowInferenceResponse,
    WorkflowsBlocksDescription,
    WorkflowsBlocksSchemaDescription,
    WorkflowValidationStatus,
)
from inference.core.env import (
    ALLOW_ORIGINS,
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_COGVLM_ENABLED,
    CORE_MODEL_DOCTR_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_GROUNDINGDINO_ENABLED,
    CORE_MODEL_OWLV2_ENABLED,
    CORE_MODEL_SAM2_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODEL_TROCR_ENABLED,
    CORE_MODEL_YOLO_WORLD_ENABLED,
    CORE_MODELS_ENABLED,
    DEDICATED_DEPLOYMENT_WORKSPACE_URL,
    DISABLE_WORKFLOW_ENDPOINTS,
    ENABLE_PROMETHEUS,
    ENABLE_STREAM_API,
    ENABLE_WORKFLOWS_PROFILING,
    LAMBDA,
    LEGACY_ROUTE_ENABLED,
    LMM_ENABLED,
    METLO_KEY,
    METRICS_ENABLED,
    NOTEBOOK_ENABLED,
    NOTEBOOK_PASSWORD,
    NOTEBOOK_PORT,
    PROFILE,
    ROBOFLOW_SERVICE_SECRET,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
    WORKFLOWS_PROFILER_BUFFER_SIZE,
    WORKFLOWS_STEP_EXECUTION_MODE,
)
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
from inference.core.interfaces.base import BaseInterface
from inference.core.interfaces.http.handlers.workflows import (
    handle_describe_workflows_blocks_request,
    handle_describe_workflows_interface,
)
from inference.core.interfaces.http.orjson_utils import (
    orjson_response,
    serialise_workflow_result,
)
from inference.core.interfaces.stream_manager.api.entities import (
    CommandResponse,
    ConsumePipelineResponse,
    InferencePipelineStatusResponse,
    ListPipelinesResponse,
)
from inference.core.interfaces.stream_manager.api.errors import (
    ProcessesManagerAuthorisationError,
    ProcessesManagerClientError,
    ProcessesManagerInvalidPayload,
    ProcessesManagerNotFoundError,
)
from inference.core.interfaces.stream_manager.api.stream_manager_client import (
    StreamManagerClient,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    ConsumeResultsPayload,
    InitialisePipelinePayload,
)
from inference.core.interfaces.stream_manager.manager_app.errors import (
    CommunicationProtocolError,
    MalformedPayloadError,
    MessageToBigError,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import (
    get_roboflow_dataset_type,
    get_roboflow_workspace,
    get_workflow_specification,
)
from inference.core.utils.notebooks import start_notebook
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
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
from inference.core.workflows.execution_engine.core import (
    ExecutionEngine,
    get_available_versions,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from inference.core.workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    NullWorkflowsProfiler,
    WorkflowsProfiler,
)
from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
    get_workflow_schema_description,
    parse_workflow_definition,
)
from inference.models.aliases import resolve_roboflow_model_alias
from inference.usage_tracking.collector import usage_collector

if LAMBDA:
    from inference.core.usage import trackUsage
if METLO_KEY:
    from metlo.fastapi import ASGIMiddleware

import time

from inference.core.version import __version__


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
    async def wrapped_route(*args, **kwargs):
        try:
            return await route(*args, **kwargs)
        except ContentTypeInvalid:
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Invalid Content-Type header provided with request."
                },
            )
            traceback.print_exc()
        except ContentTypeMissing:
            resp = JSONResponse(
                status_code=400,
                content={"message": "Content-Type header not provided with request."},
            )
            traceback.print_exc()
        except InputImageLoadError as e:
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": f"Could not load input image. Cause: {e.get_public_error_details()}"
                },
            )
            traceback.print_exc()
        except InvalidModelIDError:
            resp = JSONResponse(
                status_code=400,
                content={"message": "Invalid Model ID sent in request."},
            )
            traceback.print_exc()
        except InvalidMaskDecodeArgument:
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Invalid mask decode argument sent. tradeoff_factor must be in [0.0, 1.0], "
                    "mask_decode_mode: must be one of ['accurate', 'fast', 'tradeoff']"
                },
            )
            traceback.print_exc()
        except MissingApiKeyError:
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": "Required Roboflow API key is missing. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                    "to learn how to retrieve one."
                },
            )
            traceback.print_exc()
        except (
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
        ) as error:
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
            resp = JSONResponse(
                status_code=400,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
        except (RoboflowAPINotAuthorizedError, ProcessesManagerAuthorisationError):
            resp = JSONResponse(
                status_code=401,
                content={
                    "message": "Unauthorized access to roboflow API - check API key and make sure the key is valid for "
                    "workspace you use. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
                    "to learn how to retrieve one."
                },
            )
            traceback.print_exc()
        except (RoboflowAPINotNotFoundError, InferenceModelNotFound):
            resp = JSONResponse(
                status_code=404,
                content={
                    "message": "Requested Roboflow resource not found. Make sure that workspace, project or model "
                    "you referred in request exists."
                },
            )
            traceback.print_exc()
        except ProcessesManagerNotFoundError as error:
            resp = JSONResponse(
                status_code=404,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
            traceback.print_exc()
        except (
            InvalidEnvironmentVariableError,
            MissingServiceSecretError,
            ServiceConfigurationError,
        ):
            resp = JSONResponse(
                status_code=500, content={"message": "Service misconfiguration."}
            )
            traceback.print_exc()
        except (
            PreProcessingError,
            PostProcessingError,
        ):
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": "Model configuration related to pre- or post-processing is invalid."
                },
            )
            traceback.print_exc()
        except ModelArtefactError:
            resp = JSONResponse(
                status_code=500, content={"message": "Model package is broken."}
            )
            traceback.print_exc()
        except OnnxProviderNotAvailable:
            resp = JSONResponse(
                status_code=501,
                content={
                    "message": "Could not find requested ONNX Runtime Provider. Check that you are using "
                    "the correct docker image on a supported device."
                },
            )
            traceback.print_exc()
        except (
            MalformedRoboflowAPIResponseError,
            RoboflowAPIUnsuccessfulRequestError,
            WorkspaceLoadError,
            MalformedWorkflowResponseError,
        ):
            resp = JSONResponse(
                status_code=502,
                content={"message": "Internal error. Request to Roboflow API failed."},
            )
            traceback.print_exc()
        except RoboflowAPIConnectionError:
            resp = JSONResponse(
                status_code=503,
                content={
                    "message": "Internal error. Could not connect to Roboflow API."
                },
            )
            traceback.print_exc()
        except WorkflowError as error:
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
            traceback.print_exc()
        except (
            ProcessesManagerClientError,
            CommunicationProtocolError,
        ) as error:
            resp = JSONResponse(
                status_code=500,
                content={
                    "message": error.public_message,
                    "error_type": error.__class__.__name__,
                    "inner_error_type": error.inner_error_type,
                },
            )
            traceback.print_exc()
        except Exception:
            resp = JSONResponse(status_code=500, content={"message": "Internal error."})
            traceback.print_exc()
        return resp

    return wrapped_route


class LambdaMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        logger.info("Lambda is terminating, handle unsent usage payloads.")
        await usage_collector.async_push_usage_payloads()
        return response


class HttpInterface(BaseInterface):
    """Roboflow defined HTTP interface for a general-purpose inference server.

    This class sets up the FastAPI application and adds necessary middleware,
    as well as initializes the model manager and model registry for the inference server.

    Attributes:
        app (FastAPI): The FastAPI application instance.
        model_manager (ModelManager): The manager for handling different models.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        root_path: Optional[str] = None,
    ):
        """
        Initializes the HttpInterface with given model manager and model registry.

        Args:
            model_manager (ModelManager): The manager for handling different models.
            root_path (Optional[str]): The root path for the FastAPI application.

        Description:
            Deploy Roboflow trained models to nearly any compute environment!
        """
        description = "Roboflow inference server"
        app = FastAPI(
            title="Roboflow Inference Server",
            description=description,
            version=__version__,
            terms_of_service="https://roboflow.com/terms",
            contact={
                "name": "Roboflow Inc.",
                "url": "https://roboflow.com/contact",
                "email": "help@roboflow.com",
            },
            license_info={
                "name": "Apache 2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
            },
            root_path=root_path,
        )

        if ENABLE_PROMETHEUS:
            Instrumentator().expose(app, endpoint="/metrics")

        if METLO_KEY:
            app.add_middleware(
                ASGIMiddleware, host="https://app.metlo.com", api_key=METLO_KEY
            )
        if LAMBDA:
            app.add_middleware(LambdaMiddleware)

        if len(ALLOW_ORIGINS) > 0:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=ALLOW_ORIGINS,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Optionally add middleware for profiling the FastAPI server and underlying inference API code
        if PROFILE:
            app.add_middleware(
                CProfileMiddleware,
                enable=True,
                server_app=app,
                filename="/profile/output.pstats",
                strip_dirs=False,
                sort_by="cumulative",
            )
        app.add_middleware(asgi_correlation_id.CorrelationIdMiddleware)

        if METRICS_ENABLED:

            @app.middleware("http")
            async def count_errors(request: Request, call_next):
                """Middleware to count errors.

                Args:
                    request (Request): The incoming request.
                    call_next (Callable): The next middleware or endpoint to call.

                Returns:
                    Response: The response from the next middleware or endpoint.
                """
                response = await call_next(request)
                if self.model_manager.pingback and response.status_code >= 400:
                    self.model_manager.num_errors += 1
                return response

        if DEDICATED_DEPLOYMENT_WORKSPACE_URL:
            cached_api_keys = dict()
            cached_projects = dict()

            @app.middleware("http")
            async def check_authorization(request: Request, call_next):
                # exclusions
                skip_check = (
                    request.method not in ["GET", "POST"]
                    or request.url.path
                    in [
                        "/",
                        "/info",
                        "/workflows/blocks/describe",
                        "/workflows/definition/schema",
                    ]
                    or request.url.path.startswith("/static/")
                    or request.url.path.startswith("/_next/")
                )
                if skip_check:
                    return await call_next(request)

                def _unauthorized_response(msg):
                    return JSONResponse(
                        status_code=401,
                        content={
                            "status": 401,
                            "message": msg,
                        },
                    )

                # check api_key
                req_params = request.query_params
                json_params = dict()
                if (
                    request.headers.get("content-type", None) == "application/json"
                    and int(request.headers.get("content-length", 0)) > 0
                ):
                    json_params = await request.json()
                api_key = req_params.get("api_key", None) or json_params.get(
                    "api_key", None
                )

                if cached_api_keys.get(api_key, 0) < time.time():
                    try:
                        workspace_url = (
                            get_roboflow_workspace(api_key)
                            if api_key is not None
                            else None
                        )

                        if workspace_url != DEDICATED_DEPLOYMENT_WORKSPACE_URL:
                            return _unauthorized_response("Unauthorized api_key")

                        cached_api_keys[api_key] = (
                            time.time() + 3600
                        )  # expired after 1 hour
                    except RoboflowAPINotAuthorizedError as e:
                        return _unauthorized_response("Unauthorized api_key")

                # check project_url
                model_id = json_params.get("model_id", "")
                project_url = (
                    req_params.get("project", None)
                    or json_params.get("project", None)
                    or model_id.split("/")[0]
                )
                # only check when project_url is not None
                if (
                    project_url is not None
                    and cached_projects.get(project_url, 0) < time.time()
                ):
                    try:
                        _ = get_roboflow_dataset_type(
                            api_key, DEDICATED_DEPLOYMENT_WORKSPACE_URL, project_url
                        )

                        cached_projects[project_url] = (
                            time.time() + 3600
                        )  # expired after 1 hour
                    except RoboflowAPINotNotFoundError as e:
                        return _unauthorized_response("Unauthorized project")

                return await call_next(request)

        self.app = app
        self.model_manager = model_manager
        self.stream_manager_client: Optional[StreamManagerClient] = None

        if ENABLE_STREAM_API:
            operations_timeout = os.getenv("STREAM_MANAGER_OPERATIONS_TIMEOUT")
            if operations_timeout is not None:
                operations_timeout = float(operations_timeout)
            self.stream_manager_client = StreamManagerClient.init(
                host=os.getenv("STREAM_MANAGER_HOST", "127.0.0.1"),
                port=int(os.getenv("STREAM_MANAGER_PORT", "7070")),
                operations_timeout=operations_timeout,
            )

        async def process_inference_request(
            inference_request: InferenceRequest, **kwargs
        ) -> InferenceResponse:
            """Processes an inference request by calling the appropriate model.

            Args:
                inference_request (InferenceRequest): The request containing model ID and other inference details.

            Returns:
                InferenceResponse: The response containing the inference results.
            """
            de_aliased_model_id = resolve_roboflow_model_alias(
                model_id=inference_request.model_id
            )
            self.model_manager.add_model(de_aliased_model_id, inference_request.api_key)
            resp = await self.model_manager.infer_from_request(
                de_aliased_model_id, inference_request, **kwargs
            )
            return orjson_response(resp)

        def process_workflow_inference_request(
            workflow_request: WorkflowInferenceRequest,
            workflow_specification: dict,
            background_tasks: Optional[BackgroundTasks],
            profiler: WorkflowsProfiler,
        ) -> WorkflowInferenceResponse:
            workflow_init_parameters = {
                "workflows_core.model_manager": model_manager,
                "workflows_core.api_key": workflow_request.api_key,
                "workflows_core.background_tasks": background_tasks,
            }
            execution_engine = ExecutionEngine.init(
                workflow_definition=workflow_specification,
                init_parameters=workflow_init_parameters,
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
                prevent_local_images_loading=True,
                profiler=profiler,
            )
            result = execution_engine.run(runtime_parameters=workflow_request.inputs)
            with profiler.profile_execution_phase(
                name="workflow_results_serialisation",
                categories=["inference_package_operation"],
            ):
                outputs = serialise_workflow_result(
                    result=result,
                    excluded_fields=workflow_request.excluded_fields,
                )
            profiler_trace = profiler.export_trace()
            response = WorkflowInferenceResponse(
                outputs=outputs,
                profiler_trace=profiler_trace,
            )
            return orjson_response(response=response)

        def load_core_model(
            inference_request: InferenceRequest,
            api_key: Optional[str] = None,
            core_model: str = None,
        ) -> None:
            """Loads a core model (e.g., "clip" or "sam") into the model manager.

            Args:
                inference_request (InferenceRequest): The request containing version and other details.
                api_key (Optional[str]): The API key for the request.
                core_model (str): The core model type, e.g., "clip" or "sam".

            Returns:
                str: The core model ID.
            """
            if api_key:
                inference_request.api_key = api_key
            version_id_field = f"{core_model}_version_id"
            core_model_id = (
                f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
            )
            self.model_manager.add_model(core_model_id, inference_request.api_key)
            return core_model_id

        load_clip_model = partial(load_core_model, core_model="clip")
        """Loads the CLIP model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The CLIP model ID.
        """

        load_sam_model = partial(load_core_model, core_model="sam")
        """Loads the SAM model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The SAM model ID.
        """
        load_sam2_model = partial(load_core_model, core_model="sam2")
        """Loads the SAM2 model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The SAM2 model ID.
        """

        load_gaze_model = partial(load_core_model, core_model="gaze")
        """Loads the GAZE model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The GAZE model ID.
        """

        load_doctr_model = partial(load_core_model, core_model="doctr")
        """Loads the DocTR model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The DocTR model ID.
        """
        load_cogvlm_model = partial(load_core_model, core_model="cogvlm")
        load_paligemma_model = partial(load_core_model, core_model="paligemma")

        load_grounding_dino_model = partial(
            load_core_model, core_model="grounding_dino"
        )
        """Loads the Grounding DINO model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The Grounding DINO model ID.
        """

        load_yolo_world_model = partial(load_core_model, core_model="yolo_world")
        load_owlv2_model = partial(load_core_model, core_model="owlv2")
        """Loads the YOLO World model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The YOLO World model ID.
        """

        load_trocr_model = partial(load_core_model, core_model="trocr")
        """Loads the TrOCR model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The TrOCR model ID.
        """

        @app.get(
            "/info",
            response_model=ServerVersionInfo,
            summary="Info",
            description="Get the server name and version number",
        )
        async def root():
            """Endpoint to get the server name and version number.

            Returns:
                ServerVersionInfo: The server version information.
            """
            return ServerVersionInfo(
                name="Roboflow Inference Server",
                version=__version__,
                uuid=GLOBAL_INFERENCE_SERVER_ID,
            )

        # The current AWS Lambda authorizer only supports path parameters, therefore we can only use the legacy infer route. This case statement excludes routes which won't work for the current Lambda authorizer.
        if not LAMBDA:

            @app.get(
                "/model/registry",
                response_model=ModelsDescriptions,
                summary="Get model keys",
                description="Get the ID of each loaded model",
            )
            async def registry():
                """Get the ID of each loaded model in the registry.

                Returns:
                    ModelsDescriptions: The object containing models descriptions
                """
                logger.debug(f"Reached /model/registry")
                models_descriptions = self.model_manager.describe_models()
                return ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

            @app.post(
                "/model/add",
                response_model=ModelsDescriptions,
                summary="Load a model",
                description="Load the model with the given model ID",
            )
            @with_route_exceptions
            async def model_add(request: AddModelRequest):
                """Load the model with the given model ID into the model manager.

                Args:
                    request (AddModelRequest): The request containing the model ID and optional API key.

                Returns:
                    ModelsDescriptions: The object containing models descriptions
                """
                logger.debug(f"Reached /model/add")
                de_aliased_model_id = resolve_roboflow_model_alias(
                    model_id=request.model_id
                )
                self.model_manager.add_model(de_aliased_model_id, request.api_key)
                models_descriptions = self.model_manager.describe_models()
                return ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

            @app.post(
                "/model/remove",
                response_model=ModelsDescriptions,
                summary="Remove a model",
                description="Remove the model with the given model ID",
            )
            @with_route_exceptions
            async def model_remove(request: ClearModelRequest):
                """Remove the model with the given model ID from the model manager.

                Args:
                    request (ClearModelRequest): The request containing the model ID to be removed.

                Returns:
                    ModelsDescriptions: The object containing models descriptions
                """
                logger.debug(f"Reached /model/remove")
                de_aliased_model_id = resolve_roboflow_model_alias(
                    model_id=request.model_id
                )
                self.model_manager.remove(de_aliased_model_id)
                models_descriptions = self.model_manager.describe_models()
                return ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

            @app.post(
                "/model/clear",
                response_model=ModelsDescriptions,
                summary="Remove all models",
                description="Remove all loaded models",
            )
            @with_route_exceptions
            async def model_clear():
                """Remove all loaded models from the model manager.

                Returns:
                    ModelsDescriptions: The object containing models descriptions
                """
                logger.debug(f"Reached /model/clear")
                self.model_manager.clear()
                models_descriptions = self.model_manager.describe_models()
                return ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

            @app.post(
                "/infer/object_detection",
                response_model=Union[
                    ObjectDetectionInferenceResponse,
                    List[ObjectDetectionInferenceResponse],
                    StubResponse,
                ],
                summary="Object detection infer",
                description="Run inference with the specified object detection model",
                response_model_exclude_none=True,
            )
            @with_route_exceptions
            async def infer_object_detection(
                inference_request: ObjectDetectionInferenceRequest,
                background_tasks: BackgroundTasks,
            ):
                """Run inference with the specified object detection model.

                Args:
                    inference_request (ObjectDetectionInferenceRequest): The request containing the necessary details for object detection.
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks

                Returns:
                    Union[ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]]: The response containing the inference results.
                """
                logger.debug(f"Reached /infer/object_detection")
                return await process_inference_request(
                    inference_request,
                    active_learning_eligible=True,
                    background_tasks=background_tasks,
                )

            @app.post(
                "/infer/instance_segmentation",
                response_model=Union[
                    InstanceSegmentationInferenceResponse, StubResponse
                ],
                summary="Instance segmentation infer",
                description="Run inference with the specified instance segmentation model",
            )
            @with_route_exceptions
            async def infer_instance_segmentation(
                inference_request: InstanceSegmentationInferenceRequest,
                background_tasks: BackgroundTasks,
            ):
                """Run inference with the specified instance segmentation model.

                Args:
                    inference_request (InstanceSegmentationInferenceRequest): The request containing the necessary details for instance segmentation.
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks

                Returns:
                    InstanceSegmentationInferenceResponse: The response containing the inference results.
                """
                logger.debug(f"Reached /infer/instance_segmentation")
                return await process_inference_request(
                    inference_request,
                    active_learning_eligible=True,
                    background_tasks=background_tasks,
                )

            @app.post(
                "/infer/classification",
                response_model=Union[
                    ClassificationInferenceResponse,
                    MultiLabelClassificationInferenceResponse,
                    StubResponse,
                ],
                summary="Classification infer",
                description="Run inference with the specified classification model",
            )
            @with_route_exceptions
            async def infer_classification(
                inference_request: ClassificationInferenceRequest,
                background_tasks: BackgroundTasks,
            ):
                """Run inference with the specified classification model.

                Args:
                    inference_request (ClassificationInferenceRequest): The request containing the necessary details for classification.
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks

                Returns:
                    Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]: The response containing the inference results.
                """
                logger.debug(f"Reached /infer/classification")
                return await process_inference_request(
                    inference_request,
                    active_learning_eligible=True,
                    background_tasks=background_tasks,
                )

            @app.post(
                "/infer/keypoints_detection",
                response_model=Union[KeypointsDetectionInferenceResponse, StubResponse],
                summary="Keypoints detection infer",
                description="Run inference with the specified keypoints detection model",
            )
            @with_route_exceptions
            async def infer_keypoints(
                inference_request: KeypointsDetectionInferenceRequest,
            ):
                """Run inference with the specified keypoints detection model.

                Args:
                    inference_request (KeypointsDetectionInferenceRequest): The request containing the necessary details for keypoints detection.

                Returns:
                    Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]: The response containing the inference results.
                """
                logger.debug(f"Reached /infer/keypoints_detection")
                return await process_inference_request(inference_request)

            if LMM_ENABLED:

                @app.post(
                    "/infer/lmm",
                    response_model=Union[
                        LMMInferenceResponse,
                        List[LMMInferenceResponse],
                        StubResponse,
                    ],
                    summary="Large multi-modal model infer",
                    description="Run inference with the specified large multi-modal model",
                    response_model_exclude_none=True,
                )
                @with_route_exceptions
                async def infer_lmm(
                    inference_request: LMMInferenceRequest,
                ):
                    """Run inference with the specified object detection model.

                    Args:
                        inference_request (ObjectDetectionInferenceRequest): The request containing the necessary details for object detection.
                        background_tasks: (BackgroundTasks) pool of fastapi background tasks

                    Returns:
                        Union[ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]]: The response containing the inference results.
                    """
                    logger.debug(f"Reached /infer/lmm")
                    return await process_inference_request(inference_request)

        if not DISABLE_WORKFLOW_ENDPOINTS:

            @app.post(
                "/{workspace_name}/workflows/{workflow_id}/describe_interface",
                response_model=DescribeInterfaceResponse,
                summary="Endpoint to describe interface of predefined workflow",
                description="Checks Roboflow API for workflow definition, once acquired - describes workflow inputs and outputs",
            )
            @with_route_exceptions
            async def describe_predefined_workflow_interface(
                workspace_name: str,
                workflow_id: str,
                workflow_request: PredefinedWorkflowDescribeInterfaceRequest,
            ) -> DescribeInterfaceResponse:
                workflow_specification = get_workflow_specification(
                    api_key=workflow_request.api_key,
                    workspace_id=workspace_name,
                    workflow_id=workflow_id,
                    use_cache=workflow_request.use_cache,
                )
                return handle_describe_workflows_interface(
                    definition=workflow_specification,
                )

            @app.post(
                "/workflows/describe_interface",
                response_model=DescribeInterfaceResponse,
                summary="Endpoint to describe interface of workflow given in request",
                description="Parses workflow definition and retrieves describes inputs and outputs",
            )
            @with_route_exceptions
            async def describe_workflow_interface(
                workflow_request: WorkflowSpecificationDescribeInterfaceRequest,
            ) -> DescribeInterfaceResponse:
                return handle_describe_workflows_interface(
                    definition=workflow_request.specification,
                )

            @app.post(
                "/{workspace_name}/workflows/{workflow_id}",
                response_model=WorkflowInferenceResponse,
                summary="Endpoint to run predefined workflow",
                description="Checks Roboflow API for workflow definition, once acquired - parses and executes injecting runtime parameters from request body",
            )
            @app.post(
                "/infer/workflows/{workspace_name}/{workflow_id}",
                response_model=WorkflowInferenceResponse,
                summary="[LEGACY] Endpoint to run predefined workflow",
                description="Checks Roboflow API for workflow definition, once acquired - parses and executes injecting runtime parameters from request body. This endpoint is deprecated and will be removed end of Q2 2024",
                deprecated=True,
            )
            @with_route_exceptions
            async def infer_from_predefined_workflow(
                workspace_name: str,
                workflow_id: str,
                workflow_request: PredefinedWorkflowInferenceRequest,
                background_tasks: BackgroundTasks,
            ) -> WorkflowInferenceResponse:
                # TODO: get rid of async: https://github.com/roboflow/inference/issues/569
                if ENABLE_WORKFLOWS_PROFILING and workflow_request.enable_profiling:
                    profiler = BaseWorkflowsProfiler.init(
                        max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE,
                    )
                else:
                    profiler = NullWorkflowsProfiler.init()
                with profiler.profile_execution_phase(
                    name="workflow_definition_fetching",
                    categories=["inference_package_operation"],
                ):
                    workflow_specification = get_workflow_specification(
                        api_key=workflow_request.api_key,
                        workspace_id=workspace_name,
                        workflow_id=workflow_id,
                        use_cache=workflow_request.use_cache,
                    )
                return process_workflow_inference_request(
                    workflow_request=workflow_request,
                    workflow_specification=workflow_specification,
                    background_tasks=background_tasks if not LAMBDA else None,
                    profiler=profiler,
                )

            @app.post(
                "/workflows/run",
                response_model=WorkflowInferenceResponse,
                summary="Endpoint to run workflow specification provided in payload",
                description="Parses and executes workflow specification, injecting runtime parameters from request body.",
            )
            @app.post(
                "/infer/workflows",
                response_model=WorkflowInferenceResponse,
                summary="[LEGACY] Endpoint to run workflow specification provided in payload",
                description="Parses and executes workflow specification, injecting runtime parameters from request body. This endpoint is deprecated and will be removed end of Q2 2024.",
                deprecated=True,
            )
            @with_route_exceptions
            async def infer_from_workflow(
                workflow_request: WorkflowSpecificationInferenceRequest,
                background_tasks: BackgroundTasks,
            ) -> WorkflowInferenceResponse:
                # TODO: get rid of async: https://github.com/roboflow/inference/issues/569
                if ENABLE_WORKFLOWS_PROFILING and workflow_request.enable_profiling:
                    profiler = BaseWorkflowsProfiler.init(
                        max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE,
                    )
                else:
                    profiler = NullWorkflowsProfiler.init()
                return process_workflow_inference_request(
                    workflow_request=workflow_request,
                    workflow_specification=workflow_request.specification,
                    background_tasks=background_tasks if not LAMBDA else None,
                    profiler=profiler,
                )

            @app.get(
                "/workflows/execution_engine/versions",
                response_model=ExecutionEngineVersions,
                summary="Returns available Execution Engine versions sorted from oldest to newest",
                description="Returns available Execution Engine versions sorted from oldest to newest",
            )
            @with_route_exceptions
            async def get_execution_engine_versions() -> ExecutionEngineVersions:
                # TODO: get rid of async: https://github.com/roboflow/inference/issues/569
                versions = get_available_versions()
                return ExecutionEngineVersions(versions=versions)

            @app.get(
                "/workflows/blocks/describe",
                response_model=WorkflowsBlocksDescription,
                summary="[LEGACY] Endpoint to get definition of workflows blocks that are accessible",
                description="Endpoint provides detailed information about workflows building blocks that are "
                "accessible in the inference server. This information could be used to programmatically "
                "build / display workflows.",
                deprecated=True,
            )
            @with_route_exceptions
            async def describe_workflows_blocks() -> WorkflowsBlocksDescription:
                return handle_describe_workflows_blocks_request()

            @app.post(
                "/workflows/blocks/describe",
                response_model=WorkflowsBlocksDescription,
                summary="[EXPERIMENTAL] Endpoint to get definition of workflows blocks that are accessible",
                description="Endpoint provides detailed information about workflows building blocks that are "
                "accessible in the inference server. This information could be used to programmatically "
                "build / display workflows. Additionally - in request body one can specify list of "
                "dynamic blocks definitions which will be transformed into blocks and used to generate "
                "schemas and definitions of connections",
            )
            @with_route_exceptions
            async def describe_workflows_blocks(
                request: Optional[DescribeBlocksRequest] = None,
            ) -> WorkflowsBlocksDescription:
                # TODO: get rid of async: https://github.com/roboflow/inference/issues/569
                dynamic_blocks_definitions = None
                requested_execution_engine_version = None
                if request is not None:
                    dynamic_blocks_definitions = request.dynamic_blocks_definitions
                    requested_execution_engine_version = (
                        request.execution_engine_version
                    )
                return handle_describe_workflows_blocks_request(
                    dynamic_blocks_definitions=dynamic_blocks_definitions,
                    requested_execution_engine_version=requested_execution_engine_version,
                )

            @app.get(
                "/workflows/definition/schema",
                response_model=WorkflowsBlocksSchemaDescription,
                summary="Endpoint to fetch the workflows block schema",
                description="Endpoint to fetch the schema of all available blocks. This information can be "
                "used to validate workflow definitions and suggest syntax in the JSON editor.",
            )
            @with_route_exceptions
            async def get_workflow_schema() -> WorkflowsBlocksSchemaDescription:
                return get_workflow_schema_description()

            @app.post(
                "/workflows/blocks/dynamic_outputs",
                response_model=List[OutputDefinition],
                summary="[EXPERIMENTAL] Endpoint to get definition of dynamic output for workflow step",
                description="Endpoint to be used when step outputs can be discovered only after "
                "filling manifest with data.",
            )
            @with_route_exceptions
            async def get_dynamic_block_outputs(
                step_manifest: Dict[str, Any]
            ) -> List[OutputDefinition]:
                # TODO: get rid of async: https://github.com/roboflow/inference/issues/569
                # Potentially TODO: dynamic blocks do not support dynamic outputs, but if it changes
                # we need to provide dynamic blocks manifests here
                dummy_workflow_definition = {
                    "version": "1.0",
                    "inputs": [],
                    "steps": [step_manifest],
                    "outputs": [],
                }
                available_blocks = load_workflow_blocks()
                parsed_definition = parse_workflow_definition(
                    raw_workflow_definition=dummy_workflow_definition,
                    available_blocks=available_blocks,
                )
                parsed_manifest = parsed_definition.steps[0]
                return parsed_manifest.get_actual_outputs()

            @app.post(
                "/workflows/validate",
                response_model=WorkflowValidationStatus,
                summary="[EXPERIMENTAL] Endpoint to validate",
                description="Endpoint provides a way to check validity of JSON workflow definition.",
            )
            @with_route_exceptions
            async def validate_workflow(
                specification: dict,
            ) -> WorkflowValidationStatus:
                # TODO: get rid of async: https://github.com/roboflow/inference/issues/569
                step_execution_mode = StepExecutionMode(WORKFLOWS_STEP_EXECUTION_MODE)
                workflow_init_parameters = {
                    "workflows_core.model_manager": model_manager,
                    "workflows_core.api_key": None,
                    "workflows_core.background_tasks": None,
                    "workflows_core.step_execution_mode": step_execution_mode,
                }
                _ = ExecutionEngine.init(
                    workflow_definition=specification,
                    init_parameters=workflow_init_parameters,
                    max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
                    prevent_local_images_loading=True,
                )
                return WorkflowValidationStatus(status="ok")

        if ENABLE_STREAM_API:

            @app.get(
                "/inference_pipelines/list",
                response_model=ListPipelinesResponse,
                summary="[EXPERIMENTAL] List active InferencePipelines",
                description="[EXPERIMENTAL] Listing all active InferencePipelines processing videos",
            )
            @with_route_exceptions
            async def list_pipelines(_: Request) -> ListPipelinesResponse:
                return await self.stream_manager_client.list_pipelines()

            @app.get(
                "/inference_pipelines/{pipeline_id}/status",
                response_model=InferencePipelineStatusResponse,
                summary="[EXPERIMENTAL] Get status of InferencePipeline",
                description="[EXPERIMENTAL] Get status of InferencePipeline",
            )
            @with_route_exceptions
            async def get_status(pipeline_id: str) -> InferencePipelineStatusResponse:
                return await self.stream_manager_client.get_status(
                    pipeline_id=pipeline_id
                )

            @app.post(
                "/inference_pipelines/initialise",
                response_model=CommandResponse,
                summary="[EXPERIMENTAL] Starts new InferencePipeline",
                description="[EXPERIMENTAL] Starts new InferencePipeline",
            )
            @with_route_exceptions
            async def initialise(request: InitialisePipelinePayload) -> CommandResponse:
                return await self.stream_manager_client.initialise_pipeline(
                    initialisation_request=request
                )

            @app.post(
                "/inference_pipelines/{pipeline_id}/pause",
                response_model=CommandResponse,
                summary="[EXPERIMENTAL] Pauses the InferencePipeline",
                description="[EXPERIMENTAL] Pauses the InferencePipeline",
            )
            @with_route_exceptions
            async def pause(pipeline_id: str) -> CommandResponse:
                return await self.stream_manager_client.pause_pipeline(
                    pipeline_id=pipeline_id
                )

            @app.post(
                "/inference_pipelines/{pipeline_id}/resume",
                response_model=CommandResponse,
                summary="[EXPERIMENTAL] Resumes the InferencePipeline",
                description="[EXPERIMENTAL] Resumes the InferencePipeline",
            )
            @with_route_exceptions
            async def resume(pipeline_id: str) -> CommandResponse:
                return await self.stream_manager_client.resume_pipeline(
                    pipeline_id=pipeline_id
                )

            @app.post(
                "/inference_pipelines/{pipeline_id}/terminate",
                response_model=CommandResponse,
                summary="[EXPERIMENTAL] Terminates the InferencePipeline",
                description="[EXPERIMENTAL] Terminates the InferencePipeline",
            )
            @with_route_exceptions
            async def terminate(pipeline_id: str) -> CommandResponse:
                return await self.stream_manager_client.terminate_pipeline(
                    pipeline_id=pipeline_id
                )

            @app.get(
                "/inference_pipelines/{pipeline_id}/consume",
                response_model=ConsumePipelineResponse,
                summary="[EXPERIMENTAL] Consumes InferencePipeline result",
                description="[EXPERIMENTAL] Consumes InferencePipeline result",
            )
            @with_route_exceptions
            async def consume(
                pipeline_id: str,
                request: Optional[ConsumeResultsPayload] = None,
            ) -> ConsumePipelineResponse:
                if request is None:
                    request = ConsumeResultsPayload()
                return await self.stream_manager_client.consume_pipeline_result(
                    pipeline_id=pipeline_id,
                    excluded_fields=request.excluded_fields,
                )

        if CORE_MODELS_ENABLED:
            if CORE_MODEL_CLIP_ENABLED:

                @app.post(
                    "/clip/embed_image",
                    response_model=ClipEmbeddingResponse,
                    summary="CLIP Image Embeddings",
                    description="Run the Open AI CLIP model to embed image data.",
                )
                @with_route_exceptions
                async def clip_embed_image(
                    inference_request: ClipImageEmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Embeds image data using the OpenAI CLIP model.

                    Args:
                        inference_request (ClipImageEmbeddingRequest): The request containing the image to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        ClipEmbeddingResponse: The response containing the embedded image.
                    """
                    logger.debug(f"Reached /clip/embed_image")
                    clip_model_id = load_clip_model(inference_request, api_key=api_key)
                    response = await self.model_manager.infer_from_request(
                        clip_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(clip_model_id, actor)
                    return response

                @app.post(
                    "/clip/embed_text",
                    response_model=ClipEmbeddingResponse,
                    summary="CLIP Text Embeddings",
                    description="Run the Open AI CLIP model to embed text data.",
                )
                @with_route_exceptions
                async def clip_embed_text(
                    inference_request: ClipTextEmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Embeds text data using the OpenAI CLIP model.

                    Args:
                        inference_request (ClipTextEmbeddingRequest): The request containing the text to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        ClipEmbeddingResponse: The response containing the embedded text.
                    """
                    logger.debug(f"Reached /clip/embed_text")
                    clip_model_id = load_clip_model(inference_request, api_key=api_key)
                    response = await self.model_manager.infer_from_request(
                        clip_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(clip_model_id, actor)
                    return response

                @app.post(
                    "/clip/compare",
                    response_model=ClipCompareResponse,
                    summary="CLIP Compare",
                    description="Run the Open AI CLIP model to compute similarity scores.",
                )
                @with_route_exceptions
                async def clip_compare(
                    inference_request: ClipCompareRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Computes similarity scores using the OpenAI CLIP model.

                    Args:
                        inference_request (ClipCompareRequest): The request containing the data to be compared.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        ClipCompareResponse: The response containing the similarity scores.
                    """
                    logger.debug(f"Reached /clip/compare")
                    clip_model_id = load_clip_model(inference_request, api_key=api_key)
                    response = await self.model_manager.infer_from_request(
                        clip_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(clip_model_id, actor, n=2)
                    return response

            if CORE_MODEL_GROUNDINGDINO_ENABLED:

                @app.post(
                    "/grounding_dino/infer",
                    response_model=ObjectDetectionInferenceResponse,
                    summary="Grounding DINO inference.",
                    description="Run the Grounding DINO zero-shot object detection model.",
                )
                @with_route_exceptions
                async def grounding_dino_infer(
                    inference_request: GroundingDINOInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Embeds image data using the Grounding DINO model.

                    Args:
                        inference_request GroundingDINOInferenceRequest): The request containing the image on which to run object detection.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        ObjectDetectionInferenceResponse: The object detection response.
                    """
                    logger.debug(f"Reached /grounding_dino/infer")
                    grounding_dino_model_id = load_grounding_dino_model(
                        inference_request, api_key=api_key
                    )
                    response = await self.model_manager.infer_from_request(
                        grounding_dino_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(grounding_dino_model_id, actor)
                    return response

            if CORE_MODEL_YOLO_WORLD_ENABLED:

                @app.post(
                    "/yolo_world/infer",
                    response_model=ObjectDetectionInferenceResponse,
                    summary="YOLO-World inference.",
                    description="Run the YOLO-World zero-shot object detection model.",
                    response_model_exclude_none=True,
                )
                @with_route_exceptions
                async def yolo_world_infer(
                    inference_request: YOLOWorldInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Runs the YOLO-World zero-shot object detection model.

                    Args:
                        inference_request (YOLOWorldInferenceRequest): The request containing the image on which to run object detection.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        ObjectDetectionInferenceResponse: The object detection response.
                    """
                    logger.debug(f"Reached /yolo_world/infer. Loading model")
                    yolo_world_model_id = load_yolo_world_model(
                        inference_request, api_key=api_key
                    )
                    logger.debug("YOLOWorld model loaded. Staring the inference.")
                    response = await self.model_manager.infer_from_request(
                        yolo_world_model_id, inference_request
                    )
                    logger.debug("YOLOWorld prediction available.")
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(yolo_world_model_id, actor)
                        logger.debug("Usage of YOLOWorld denoted.")
                    return response

            if CORE_MODEL_DOCTR_ENABLED:

                @app.post(
                    "/doctr/ocr",
                    response_model=OCRInferenceResponse,
                    summary="DocTR OCR response",
                    description="Run the DocTR OCR model to retrieve text in an image.",
                )
                @with_route_exceptions
                async def doctr_retrieve_text(
                    inference_request: DoctrOCRInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Embeds image data using the DocTR model.

                    Args:
                        inference_request (M.DoctrOCRInferenceRequest): The request containing the image from which to retrieve text.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.OCRInferenceResponse: The response containing the embedded image.
                    """
                    logger.debug(f"Reached /doctr/ocr")
                    doctr_model_id = load_doctr_model(
                        inference_request, api_key=api_key
                    )
                    response = await self.model_manager.infer_from_request(
                        doctr_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(doctr_model_id, actor)
                    return response

            if CORE_MODEL_SAM_ENABLED:

                @app.post(
                    "/sam/embed_image",
                    response_model=SamEmbeddingResponse,
                    summary="SAM Image Embeddings",
                    description="Run the Meta AI Segmant Anything Model to embed image data.",
                )
                @with_route_exceptions
                async def sam_embed_image(
                    inference_request: SamEmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Embeds image data using the Meta AI Segmant Anything Model (SAM).

                    Args:
                        inference_request (SamEmbeddingRequest): The request containing the image to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.SamEmbeddingResponse or Response: The response containing the embedded image.
                    """
                    logger.debug(f"Reached /sam/embed_image")
                    sam_model_id = load_sam_model(inference_request, api_key=api_key)
                    model_response = await self.model_manager.infer_from_request(
                        sam_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(sam_model_id, actor)
                    if inference_request.format == "binary":
                        return Response(
                            content=model_response.embeddings,
                            headers={"Content-Type": "application/octet-stream"},
                        )
                    return model_response

                @app.post(
                    "/sam/segment_image",
                    response_model=SamSegmentationResponse,
                    summary="SAM Image Segmentation",
                    description="Run the Meta AI Segmant Anything Model to generate segmenations for image data.",
                )
                @with_route_exceptions
                async def sam_segment_image(
                    inference_request: SamSegmentationRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Generates segmentations for image data using the Meta AI Segmant Anything Model (SAM).

                    Args:
                        inference_request (SamSegmentationRequest): The request containing the image to be segmented.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.SamSegmentationResponse or Response: The response containing the segmented image.
                    """
                    logger.debug(f"Reached /sam/segment_image")
                    sam_model_id = load_sam_model(inference_request, api_key=api_key)
                    model_response = await self.model_manager.infer_from_request(
                        sam_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(sam_model_id, actor)
                    if inference_request.format == "binary":
                        return Response(
                            content=model_response,
                            headers={"Content-Type": "application/octet-stream"},
                        )
                    return model_response

            if CORE_MODEL_SAM2_ENABLED:

                @app.post(
                    "/sam2/embed_image",
                    response_model=Sam2EmbeddingResponse,
                    summary="SAM2 Image Embeddings",
                    description="Run the Meta AI Segment Anything 2 Model to embed image data.",
                )
                @with_route_exceptions
                async def sam2_embed_image(
                    inference_request: Sam2EmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Embeds image data using the Meta AI Segment Anything Model (SAM).

                    Args:
                        inference_request (SamEmbeddingRequest): The request containing the image to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.Sam2EmbeddingResponse or Response: The response affirming the image has been embedded
                    """
                    logger.debug(f"Reached /sam2/embed_image")
                    sam2_model_id = load_sam2_model(inference_request, api_key=api_key)
                    model_response = await self.model_manager.infer_from_request(
                        sam2_model_id, inference_request
                    )
                    return model_response

                @app.post(
                    "/sam2/segment_image",
                    response_model=Sam2SegmentationResponse,
                    summary="SAM2 Image Segmentation",
                    description="Run the Meta AI Segment Anything 2 Model to generate segmenations for image data.",
                )
                @with_route_exceptions
                async def sam2_segment_image(
                    inference_request: Sam2SegmentationRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Generates segmentations for image data using the Meta AI Segment Anything Model (SAM).

                    Args:
                        inference_request (Sam2SegmentationRequest): The request containing the image to be segmented.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.SamSegmentationResponse or Response: The response containing the segmented image.
                    """
                    logger.debug(f"Reached /sam2/segment_image")
                    sam2_model_id = load_sam2_model(inference_request, api_key=api_key)
                    model_response = await self.model_manager.infer_from_request(
                        sam2_model_id, inference_request
                    )
                    if inference_request.format == "binary":
                        return Response(
                            content=model_response,
                            headers={"Content-Type": "application/octet-stream"},
                        )
                    return model_response

            if CORE_MODEL_OWLV2_ENABLED:

                @app.post(
                    "/owlv2/infer",
                    response_model=ObjectDetectionInferenceResponse,
                    summary="Owlv2 image prompting",
                    description="Run the google owlv2 model to few-shot object detect",
                )
                @with_route_exceptions
                async def owlv2_infer(
                    inference_request: OwlV2InferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Embeds image data using the Meta AI Segmant Anything Model (SAM).

                    Args:
                        inference_request (SamEmbeddingRequest): The request containing the image to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.Sam2EmbeddingResponse or Response: The response affirming the image has been embedded
                    """
                    logger.debug(f"Reached /owlv2/infer")
                    owl2_model_id = load_owlv2_model(inference_request, api_key=api_key)
                    model_response = await self.model_manager.infer_from_request(
                        owl2_model_id, inference_request
                    )
                    return model_response

            if CORE_MODEL_GAZE_ENABLED:

                @app.post(
                    "/gaze/gaze_detection",
                    response_model=List[GazeDetectionInferenceResponse],
                    summary="Gaze Detection",
                    description="Run the gaze detection model to detect gaze.",
                )
                @with_route_exceptions
                async def gaze_detection(
                    inference_request: GazeDetectionInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Detect gaze using the gaze detection model.

                    Args:
                        inference_request (M.GazeDetectionRequest): The request containing the image to be detected.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.GazeDetectionResponse: The response containing all the detected faces and the corresponding gazes.
                    """
                    logger.debug(f"Reached /gaze/gaze_detection")
                    gaze_model_id = load_gaze_model(inference_request, api_key=api_key)
                    response = await self.model_manager.infer_from_request(
                        gaze_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(gaze_model_id, actor)
                    return response

            if CORE_MODEL_COGVLM_ENABLED:

                @app.post(
                    "/llm/cogvlm",
                    response_model=CogVLMResponse,
                    summary="CogVLM",
                    description="Run the CogVLM model to chat or describe an image.",
                )
                @with_route_exceptions
                async def cog_vlm(
                    inference_request: CogVLMInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Chat with CogVLM or ask it about an image. Multi-image requests not currently supported.

                    Args:
                        inference_request (M.CogVLMInferenceRequest): The request containing the prompt and image to be described.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.CogVLMResponse: The model's text response
                    """
                    logger.debug(f"Reached /llm/cogvlm")
                    cog_model_id = load_cogvlm_model(inference_request, api_key=api_key)
                    response = await self.model_manager.infer_from_request(
                        cog_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(cog_model_id, actor)
                    return response

            if CORE_MODEL_TROCR_ENABLED:

                @app.post(
                    "/ocr/trocr",
                    response_model=OCRInferenceResponse,
                    summary="TrOCR OCR response",
                    description="Run the TrOCR model to retrieve text in an image.",
                )
                @with_route_exceptions
                async def trocr_retrieve_text(
                    inference_request: TrOCRInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                ):
                    """
                    Retrieves text from image data using the TrOCR model.

                    Args:
                        inference_request (TrOCRInferenceRequest): The request containing the image from which to retrieve text.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        OCRInferenceResponse: The response containing the retrieved text.
                    """
                    logger.debug(f"Reached /trocr/ocr")
                    trocr_model_id = load_trocr_model(
                        inference_request, api_key=api_key
                    )
                    response = await self.model_manager.infer_from_request(
                        trocr_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(trocr_model_id, actor)
                    return response

        if not LAMBDA:

            @app.get(
                "/notebook/start",
                summary="Jupyter Lab Server Start",
                description="Starts a jupyter lab server for running development code",
            )
            @with_route_exceptions
            async def notebook_start(browserless: bool = False):
                """Starts a jupyter lab server for running development code.

                Args:
                    inference_request (NotebookStartRequest): The request containing the necessary details for starting a jupyter lab server.
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks

                Returns:
                    NotebookStartResponse: The response containing the URL of the jupyter lab server.
                """
                logger.debug(f"Reached /notebook/start")
                if NOTEBOOK_ENABLED:
                    start_notebook()
                    if browserless:
                        return {
                            "success": True,
                            "message": f"Jupyter Lab server started at http://localhost:{NOTEBOOK_PORT}?token={NOTEBOOK_PASSWORD}",
                        }
                    else:
                        sleep(2)
                        return RedirectResponse(
                            f"http://localhost:{NOTEBOOK_PORT}/lab/tree/quickstart.ipynb?token={NOTEBOOK_PASSWORD}"
                        )
                else:
                    if browserless:
                        return {
                            "success": False,
                            "message": "Notebook server is not enabled. Enable notebooks via the NOTEBOOK_ENABLED environment variable.",
                        }
                    else:
                        return RedirectResponse(f"/notebook-instructions.html")

        if LEGACY_ROUTE_ENABLED:

            class IntStringConvertor(StringConvertor):
                """
                Match digits but keep them as string.
                """

                regex = "\d+"

            register_url_convertor("int_string", IntStringConvertor())

            # Legacy object detection inference path for backwards compatability
            @app.get(
                "/{dataset_id}/{version_id:int_string}",
                # Order matters in this response model Union. It will use the first matching model. For example, Object Detection Inference Response is a subset of Instance segmentation inference response, so instance segmentation must come first in order for the matching logic to work.
                response_model=Union[
                    InstanceSegmentationInferenceResponse,
                    KeypointsDetectionInferenceResponse,
                    ObjectDetectionInferenceResponse,
                    ClassificationInferenceResponse,
                    MultiLabelClassificationInferenceResponse,
                    StubResponse,
                    Any,
                ],
                response_model_exclude_none=True,
            )
            @app.post(
                "/{dataset_id}/{version_id:int_string}",
                # Order matters in this response model Union. It will use the first matching model. For example, Object Detection Inference Response is a subset of Instance segmentation inference response, so instance segmentation must come first in order for the matching logic to work.
                response_model=Union[
                    InstanceSegmentationInferenceResponse,
                    KeypointsDetectionInferenceResponse,
                    ObjectDetectionInferenceResponse,
                    ClassificationInferenceResponse,
                    MultiLabelClassificationInferenceResponse,
                    StubResponse,
                    Any,
                ],
                response_model_exclude_none=True,
            )
            @with_route_exceptions
            async def legacy_infer_from_request(
                background_tasks: BackgroundTasks,
                request: Request,
                dataset_id: str = Path(
                    description="ID of a Roboflow dataset corresponding to the model to use for inference"
                ),
                version_id: str = Path(
                    description="ID of a Roboflow dataset version corresponding to the model to use for inference"
                ),
                api_key: Optional[str] = Query(
                    None,
                    description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                ),
                confidence: float = Query(
                    0.4,
                    description="The confidence threshold used to filter out predictions",
                ),
                keypoint_confidence: float = Query(
                    0.0,
                    description="The confidence threshold used to filter out keypoints that are not visible based on model confidence",
                ),
                format: str = Query(
                    "json",
                    description="One of 'json' or 'image'. If 'json' prediction data is return as a JSON string. If 'image' prediction data is visualized and overlayed on the original input image.",
                ),
                image: Optional[str] = Query(
                    None,
                    description="The publically accessible URL of an image to use for inference.",
                ),
                image_type: Optional[str] = Query(
                    "base64",
                    description="One of base64 or numpy. Note, numpy input is not supported for Roboflow Hosted Inference.",
                ),
                labels: Optional[bool] = Query(
                    False,
                    description="If true, labels will be include in any inference visualization.",
                ),
                mask_decode_mode: Optional[str] = Query(
                    "accurate",
                    description="One of 'accurate' or 'fast'. If 'accurate' the mask will be decoded using the original image size. If 'fast' the mask will be decoded using the original mask size. 'accurate' is slower but more accurate.",
                ),
                tradeoff_factor: Optional[float] = Query(
                    0.0,
                    description="The amount to tradeoff between 0='fast' and 1='accurate'",
                ),
                max_detections: int = Query(
                    300,
                    description="The maximum number of detections to return. This is used to limit the number of predictions returned by the model. The model may return more predictions than this number, but only the top `max_detections` predictions will be returned.",
                ),
                overlap: float = Query(
                    0.3,
                    description="The IoU threhsold that must be met for a box pair to be considered duplicate during NMS",
                ),
                stroke: int = Query(
                    1, description="The stroke width used when visualizing predictions"
                ),
                countinference: Optional[bool] = Query(
                    True,
                    description="If false, does not track inference against usage.",
                    include_in_schema=False,
                ),
                service_secret: Optional[str] = Query(
                    None,
                    description="Shared secret used to authenticate requests to the inference server from internal services (e.g. to allow disabling inference usage tracking via the `countinference` query parameter)",
                    include_in_schema=False,
                ),
                disable_preproc_auto_orient: Optional[bool] = Query(
                    False, description="If true, disables automatic image orientation"
                ),
                disable_preproc_contrast: Optional[bool] = Query(
                    False, description="If true, disables automatic contrast adjustment"
                ),
                disable_preproc_grayscale: Optional[bool] = Query(
                    False,
                    description="If true, disables automatic grayscale conversion",
                ),
                disable_preproc_static_crop: Optional[bool] = Query(
                    False, description="If true, disables automatic static crop"
                ),
                disable_active_learning: Optional[bool] = Query(
                    default=False,
                    description="If true, the predictions will be prevented from registration by Active Learning (if the functionality is enabled)",
                ),
                active_learning_target_dataset: Optional[str] = Query(
                    default=None,
                    description="Parameter to be used when Active Learning data registration should happen against different dataset than the one pointed by model_id",
                ),
                source: Optional[str] = Query(
                    "external",
                    description="The source of the inference request",
                ),
                source_info: Optional[str] = Query(
                    "external",
                    description="The detailed source information of the inference request",
                ),
            ):
                """
                Legacy inference endpoint for object detection, instance segmentation, and classification.

                Args:
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks
                    dataset_id (str): ID of a Roboflow dataset corresponding to the model to use for inference.
                    version_id (str): ID of a Roboflow dataset version corresponding to the model to use for inference.
                    api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                    # Other parameters described in the function signature...

                Returns:
                    Union[InstanceSegmentationInferenceResponse, KeypointsDetectionInferenceRequest, ObjectDetectionInferenceResponse, ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse, Any]: The response containing the inference results.
                """
                logger.debug(
                    f"Reached legacy route /:dataset_id/:version_id with {dataset_id}/{version_id}"
                )
                model_id = f"{dataset_id}/{version_id}"

                if confidence >= 1:
                    confidence /= 100
                elif confidence < 0.01:
                    confidence = 0.01

                if overlap >= 1:
                    overlap /= 100

                if image is not None:
                    request_image = InferenceRequestImage(type="url", value=image)
                else:
                    if "Content-Type" not in request.headers:
                        raise ContentTypeMissing(
                            f"Request must include a Content-Type header"
                        )
                    if "multipart/form-data" in request.headers["Content-Type"]:
                        form_data = await request.form()
                        base64_image_str = await form_data["file"].read()
                        base64_image_str = base64.b64encode(base64_image_str)
                        request_image = InferenceRequestImage(
                            type="base64", value=base64_image_str.decode("ascii")
                        )
                    elif (
                        "application/x-www-form-urlencoded"
                        in request.headers["Content-Type"]
                        or "application/json" in request.headers["Content-Type"]
                    ):
                        data = await request.body()
                        request_image = InferenceRequestImage(
                            type=image_type, value=data
                        )
                    else:
                        raise ContentTypeInvalid(
                            f"Invalid Content-Type: {request.headers['Content-Type']}"
                        )

                if LAMBDA:
                    request_model_id = (
                        request.scope["aws.event"]["requestContext"]["authorizer"][
                            "lambda"
                        ]["model"]["endpoint"]
                        .replace("--", "/")
                        .replace("rf-", "")
                        .replace("nu-", "")
                    )
                    actor = request.scope["aws.event"]["requestContext"]["authorizer"][
                        "lambda"
                    ]["actor"]
                    if countinference:
                        trackUsage(request_model_id, actor)
                    else:
                        if service_secret != ROBOFLOW_SERVICE_SECRET:
                            raise MissingServiceSecretError(
                                "Service secret is required to disable inference usage tracking"
                            )
                else:
                    request_model_id = model_id
                logger.debug(
                    f"State of model registry: {self.model_manager.describe_models()}"
                )
                self.model_manager.add_model(
                    request_model_id, api_key, model_id_alias=model_id
                )

                task_type = self.model_manager.get_task_type(model_id, api_key=api_key)
                inference_request_type = ObjectDetectionInferenceRequest
                args = dict()
                if task_type == "instance-segmentation":
                    inference_request_type = InstanceSegmentationInferenceRequest
                    args = {
                        "mask_decode_mode": mask_decode_mode,
                        "tradeoff_factor": tradeoff_factor,
                    }
                elif task_type == "classification":
                    inference_request_type = ClassificationInferenceRequest
                elif task_type == "keypoint-detection":
                    inference_request_type = KeypointsDetectionInferenceRequest
                    args = {"keypoint_confidence": keypoint_confidence}
                inference_request = inference_request_type(
                    api_key=api_key,
                    model_id=model_id,
                    image=request_image,
                    confidence=confidence,
                    iou_threshold=overlap,
                    max_detections=max_detections,
                    visualization_labels=labels,
                    visualization_stroke_width=stroke,
                    visualize_predictions=(
                        format == "image" or format == "image_and_json"
                    ),
                    disable_preproc_auto_orient=disable_preproc_auto_orient,
                    disable_preproc_contrast=disable_preproc_contrast,
                    disable_preproc_grayscale=disable_preproc_grayscale,
                    disable_preproc_static_crop=disable_preproc_static_crop,
                    disable_active_learning=disable_active_learning,
                    active_learning_target_dataset=active_learning_target_dataset,
                    source=source,
                    source_info=source_info,
                    **args,
                )

                inference_response = await self.model_manager.infer_from_request(
                    inference_request.model_id,
                    inference_request,
                    active_learning_eligible=True,
                    background_tasks=background_tasks,
                )
                logger.debug("Response ready.")
                if format == "image":
                    return Response(
                        content=inference_response.visualization,
                        media_type="image/jpeg",
                    )
                else:
                    return orjson_response(inference_response)

        if not LAMBDA:
            # Legacy clear cache endpoint for backwards compatability
            @app.get("/clear_cache", response_model=str)
            async def legacy_clear_cache():
                """
                Clears the model cache.

                This endpoint provides a way to clear the cache of loaded models.

                Returns:
                    str: A string indicating that the cache has been cleared.
                """
                logger.debug(f"Reached /clear_cache")
                await model_clear()
                return "Cache Cleared"

            # Legacy add model endpoint for backwards compatability
            @app.get("/start/{dataset_id}/{version_id}")
            async def model_add(dataset_id: str, version_id: str, api_key: str = None):
                """
                Starts a model inference session.

                This endpoint initializes and starts an inference session for the specified model version.

                Args:
                    dataset_id (str): ID of a Roboflow dataset corresponding to the model.
                    version_id (str): ID of a Roboflow dataset version corresponding to the model.
                    api_key (str, optional): Roboflow API Key for artifact retrieval.

                Returns:
                    JSONResponse: A response object containing the status and a success message.
                """
                logger.debug(
                    f"Reached /start/{dataset_id}/{version_id} with {dataset_id}/{version_id}"
                )
                model_id = f"{dataset_id}/{version_id}"
                self.model_manager.add_model(model_id, api_key)

                return JSONResponse(
                    {
                        "status": 200,
                        "message": "inference session started from local memory.",
                    }
                )

        app.mount(
            "/",
            StaticFiles(directory="./inference/landing/out", html=True),
            name="static",
        )

    def run(self):
        uvicorn.run(self.app, host="127.0.0.1", port=8080)
