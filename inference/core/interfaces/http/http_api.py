import base64
import concurrent
import os
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from threading import Lock, Thread
from time import sleep
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import asgi_correlation_id
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Query,
    Request,
)
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi_cprofile.profiler import CProfileMiddleware
from starlette.datastructures import UploadFile
from starlette.middleware.base import BaseHTTPMiddleware

from inference.core import logger
from inference.core.constants import PROCESSING_TIME_HEADER
from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.entities.requests.clip import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
)
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.gaze import GazeDetectionInferenceRequest
from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    DepthEstimationRequest,
    InferenceRequest,
    InferenceRequestImage,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    LMMInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderCompareRequest,
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.core.entities.requests.sam import (
    SamEmbeddingRequest,
    SamSegmentationRequest,
)
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2SegmentationRequest,
)
from inference.core.entities.requests.sam3 import (
    Sam3EmbeddingRequest,
    Sam3SegmentationRequest,
)
from inference.core.entities.requests.server_state import (
    AddModelRequest,
    ClearModelRequest,
)
from inference.core.entities.requests.trocr import TrOCRInferenceRequest
from inference.core.entities.requests.workflows import (
    DescribeBlocksRequest,
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
from inference.core.entities.responses.gaze import GazeDetectionInferenceResponse
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    DepthEstimationResponse,
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
from inference.core.entities.responses.perception_encoder import (
    PerceptionEncoderCompareResponse,
    PerceptionEncoderEmbeddingResponse,
)
from inference.core.entities.responses.sam import (
    SamEmbeddingResponse,
    SamSegmentationResponse,
)
from inference.core.entities.responses.sam2 import (
    Sam2EmbeddingResponse,
    Sam2SegmentationResponse,
)
from inference.core.entities.responses.sam3 import (
    Sam3EmbeddingResponse,
    Sam3SegmentationResponse,
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
    API_KEY,
    API_LOGGING_ENABLED,
    BUILDER_ORIGIN,
    CONFIDENCE_LOWER_BOUND_OOM_PREVENTION,
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_DOCTR_ENABLED,
    CORE_MODEL_EASYOCR_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_GROUNDINGDINO_ENABLED,
    CORE_MODEL_OWLV2_ENABLED,
    CORE_MODEL_PE_ENABLED,
    CORE_MODEL_SAM2_ENABLED,
    CORE_MODEL_SAM3_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODEL_TROCR_ENABLED,
    CORE_MODEL_YOLO_WORLD_ENABLED,
    CORE_MODELS_ENABLED,
    CORRELATION_ID_HEADER,
    DEDICATED_DEPLOYMENT_WORKSPACE_URL,
    DEPTH_ESTIMATION_ENABLED,
    DISABLE_WORKFLOW_ENDPOINTS,
    DOCKER_SOCKET_PATH,
    ENABLE_BUILDER,
    ENABLE_DASHBOARD,
    ENABLE_PROMETHEUS,
    ENABLE_STREAM_API,
    ENABLE_WORKFLOWS_PROFILING,
    GCP_SERVERLESS,
    GET_MODEL_REGISTRY_ENABLED,
    LAMBDA,
    LEGACY_ROUTE_ENABLED,
    LMM_ENABLED,
    METRICS_ENABLED,
    MOONDREAM2_ENABLED,
    NOTEBOOK_ENABLED,
    NOTEBOOK_PASSWORD,
    NOTEBOOK_PORT,
    PRELOAD_MODELS,
    PROFILE,
    ROBOFLOW_SERVICE_SECRET,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
    WORKFLOWS_PROFILER_BUFFER_SIZE,
    WORKFLOWS_STEP_EXECUTION_MODE,
)
from inference.core.exceptions import (
    ContentTypeInvalid,
    ContentTypeMissing,
    InputImageLoadError,
    MissingServiceSecretError,
    RoboflowAPINotAuthorizedError,
    WorkspaceLoadError,
)
from inference.core.interfaces.base import BaseInterface
from inference.core.interfaces.http.dependencies import (
    parse_body_content_for_legacy_request_handler,
)
from inference.core.interfaces.http.error_handlers import (
    with_route_exceptions,
    with_route_exceptions_async,
)
from inference.core.interfaces.http.handlers.workflows import (
    filter_out_unwanted_workflow_outputs,
    handle_describe_workflows_blocks_request,
    handle_describe_workflows_interface,
)
from inference.core.interfaces.http.middlewares.cors import PathAwareCORSMiddleware
from inference.core.interfaces.http.middlewares.gzip import gzip_response_if_requested
from inference.core.interfaces.http.orjson_utils import (
    orjson_response,
    orjson_response_keeping_parent_id,
)
from inference.core.interfaces.stream_manager.api.entities import (
    CommandResponse,
    ConsumePipelineResponse,
    InferencePipelineStatusResponse,
    InitializeWebRTCPipelineResponse,
    ListPipelinesResponse,
)
from inference.core.interfaces.stream_manager.api.stream_manager_client import (
    StreamManagerClient,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    ConsumeResultsPayload,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.metrics import get_container_stats
from inference.core.managers.prometheus import InferenceInstrumentator
from inference.core.roboflow_api import (
    get_roboflow_workspace,
    get_roboflow_workspace_async,
    get_workflow_specification,
)
from inference.core.utils.container import is_docker_socket_mounted
from inference.core.utils.notebooks import start_notebook
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
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

import time

from inference.core.roboflow_api import ModelEndpointType
from inference.core.version import __version__

try:
    from inference_sdk.config import EXECUTION_ID_HEADER, execution_id
except ImportError:
    execution_id = None
    EXECUTION_ID_HEADER = None


class LambdaMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        logger.info("Lambda is terminating, handle unsent usage payloads.")
        await usage_collector.async_push_usage_payloads()
        return response


class GCPServerlessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if execution_id is not None:
            execution_id_value = request.headers.get(EXECUTION_ID_HEADER)
            if not execution_id_value:
                execution_id_value = f"{time.time_ns()}_{uuid4().hex[:4]}"
            execution_id.set(execution_id_value)
        t1 = time.time()
        response = await call_next(request)
        t2 = time.time()
        response.headers[PROCESSING_TIME_HEADER] = str(t2 - t1)
        if execution_id is not None:
            response.headers[EXECUTION_ID_HEADER] = execution_id_value
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
        # Ensure in-memory logging is initialized as early as possible for all runtimes
        try:
            from inference.core.logging.memory_handler import setup_memory_logging

            setup_memory_logging()
        except Exception:
            pass

        app.mount(
            "/static",
            StaticFiles(directory="./inference/landing/out/static", html=True),
            name="static",
        )
        app.mount(
            "/_next/static",
            StaticFiles(directory="./inference/landing/out/_next/static", html=True),
            name="_next_static",
        )

        @app.on_event("shutdown")
        async def on_shutdown():
            logger.info("Shutting down %s", description)
            await usage_collector.async_push_usage_payloads()

        InferenceInstrumentator(app, model_manager=model_manager, endpoint="/metrics")
        if LAMBDA:
            app.add_middleware(LambdaMiddleware)
        if GCP_SERVERLESS:
            app.add_middleware(GCPServerlessMiddleware)

        if len(ALLOW_ORIGINS) > 0:
            # Add CORS Middleware (but not for /build**, which is controlled separately)
            app.add_middleware(
                PathAwareCORSMiddleware,
                match_paths=r"^(?!/build).*",
                allow_origins=ALLOW_ORIGINS,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
                expose_headers=[PROCESSING_TIME_HEADER],
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
        if API_LOGGING_ENABLED:
            app.add_middleware(
                asgi_correlation_id.CorrelationIdMiddleware,
                header_name=CORRELATION_ID_HEADER,
                update_request_header=True,
                generator=lambda: uuid4().hex,
                validator=lambda a: True,
                transformer=lambda a: a,
            )
        else:
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

        if not (LAMBDA or GCP_SERVERLESS):

            @app.get("/device/stats")
            def device_stats():
                not_configured_error_message = {
                    "error": "Device statistics endpoint is not enabled.",
                    "hint": "Mount the Docker socket and point its location when running the docker "
                    "container to collect device stats "
                    "(i.e. `docker run ... -v /var/run/docker.sock:/var/run/docker.sock "
                    "-e DOCKER_SOCKET_PATH=/var/run/docker.sock ...`).",
                }
                if not DOCKER_SOCKET_PATH:
                    return JSONResponse(
                        status_code=404,
                        content=not_configured_error_message,
                    )
                if not is_docker_socket_mounted(docker_socket_path=DOCKER_SOCKET_PATH):
                    return JSONResponse(
                        status_code=500,
                        content=not_configured_error_message,
                    )
                container_stats = get_container_stats(
                    docker_socket_path=DOCKER_SOCKET_PATH
                )
                return JSONResponse(status_code=200, content=container_stats)

        cached_api_keys = dict()

        if GCP_SERVERLESS:

            @app.middleware("http")
            async def check_authorization_serverless(request: Request, call_next):
                # exclusions
                skip_check = (
                    request.method not in ["GET", "POST"]
                    or request.url.path
                    in [
                        "/",
                        "/docs",
                        "/info",
                        "/openapi.json",  # needed for /docs and /redoc
                        "/model/registry",  # dont auth this route, usually not used on serverlerless, but queue based serverless uses it internally (not accessible from outside)
                    ]
                    or request.url.path.startswith("/static/")
                    or request.url.path.startswith("/_next/")
                )

                # for these routes we only want to auth if dynamic python modules are provided
                if request.url.path in [
                    "/workflows/blocks/describe",
                    "/workflows/definition/schema",
                ]:
                    if request.method == "GET":
                        skip_check = True

                    elif (
                        request.headers.get("content-type", None) == "application/json"
                        and int(request.headers.get("content-length", 0)) > 0
                    ):
                        json_params = await request.json()
                        dynamic_blocks_definitions = json_params.get(
                            "dynamic_blocks_definitions", None
                        )
                        if not dynamic_blocks_definitions:
                            skip_check = True

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

                req_params = request.query_params
                json_params = dict()
                api_key = req_params.get("api_key", None)
                if (
                    api_key is None
                    and request.headers.get("content-type", None) == "application/json"
                    and int(request.headers.get("content-length", 0)) > 0
                ):
                    # have to try catch here, because some legacy endpoints that abuse Content-Type header but dont actually receive json
                    try:
                        json_params = await request.json()
                    except Exception:
                        pass
                api_key = json_params.get("api_key", api_key)

                if api_key is None:
                    return _unauthorized_response("Unauthorized api_key")

                if cached_api_keys.get(api_key, 0) < time.time():
                    try:
                        await get_roboflow_workspace_async(api_key=api_key)
                        cached_api_keys[api_key] = (
                            time.time() + 3600
                        )  # expired after 1 hour
                    except (RoboflowAPINotAuthorizedError, WorkspaceLoadError):
                        return _unauthorized_response("Unauthorized api_key")

                return await call_next(request)

        if DEDICATED_DEPLOYMENT_WORKSPACE_URL:

            @app.middleware("http")
            async def check_authorization(request: Request, call_next):
                # exclusions
                skip_check = (
                    request.method not in ["GET", "POST"]
                    or request.url.path
                    in [
                        "/",
                        "/docs",
                        "/redoc",
                        "/info",
                        "/openapi.json",  # needed for /docs and /redoc
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
                api_key = req_params.get("api_key", None)
                if (
                    api_key is None
                    and request.headers.get("content-type", None) == "application/json"
                    and int(request.headers.get("content-length", 0)) > 0
                ):
                    # have to try catch here, because some legacy endpoints that abuse Content-Type header but dont actually receive json
                    try:
                        json_params = await request.json()
                    except Exception:
                        pass
                api_key = json_params.get("api_key", api_key)

                if api_key is None:
                    return _unauthorized_response("Unauthorized api_key")

                if cached_api_keys.get(api_key, 0) < time.time():
                    try:
                        # TODO: make this request async!
                        if api_key is None:
                            workspace_url = None
                        else:
                            workspace_url = await get_roboflow_workspace_async(
                                api_key=api_key
                            )

                        if workspace_url != DEDICATED_DEPLOYMENT_WORKSPACE_URL:
                            return _unauthorized_response("Unauthorized api_key")

                        cached_api_keys[api_key] = (
                            time.time() + 3600
                        )  # expired after 1 hour
                    except (RoboflowAPINotAuthorizedError, WorkspaceLoadError):
                        return _unauthorized_response("Unauthorized api_key")

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

        def process_inference_request(
            inference_request: InferenceRequest,
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
            **kwargs,
        ) -> InferenceResponse:
            """Processes an inference request by calling the appropriate model.

            Args:
                inference_request (InferenceRequest): The request containing model ID and other inference details.
                countinference (Optional[bool]): Whether to count inference for usage.
                service_secret (Optional[str]): The service secret.

            Returns:
                InferenceResponse: The response containing the inference results.
            """
            de_aliased_model_id = resolve_roboflow_model_alias(
                model_id=inference_request.model_id
            )
            self.model_manager.add_model(
                de_aliased_model_id,
                inference_request.api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            resp = self.model_manager.infer_from_request_sync(
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
                workflow_id=workflow_request.workflow_id,
            )
            is_preview = False
            if hasattr(workflow_request, "is_preview"):
                is_preview = workflow_request.is_preview
            workflow_results = execution_engine.run(
                runtime_parameters=workflow_request.inputs,
                serialize_results=True,
                _is_preview=is_preview,
            )
            with profiler.profile_execution_phase(
                name="workflow_results_filtering",
                categories=["inference_package_operation"],
            ):
                outputs = filter_out_unwanted_workflow_outputs(
                    workflow_results=workflow_results,
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
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ) -> None:
            """Loads a core model (e.g., "clip" or "sam") into the model manager.

            Args:
                inference_request (InferenceRequest): The request containing version and other details.
                api_key (Optional[str]): The API key for the request.
                core_model (str): The core model type, e.g., "clip" or "sam".
                countinference (Optional[bool]): Whether to count inference or not.
                service_secret (Optional[str]): The service secret for the request.

            Returns:
                str: The core model ID.
            """
            if api_key:
                inference_request.api_key = api_key
            version_id_field = f"{core_model}_version_id"
            core_model_id = (
                f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
            )
            self.model_manager.add_model(
                core_model_id,
                inference_request.api_key,
                endpoint_type=ModelEndpointType.CORE_MODEL,
                countinference=countinference,
                service_secret=service_secret,
            )
            return core_model_id

        load_clip_model = partial(load_core_model, core_model="clip")
        """Loads the CLIP model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The CLIP model ID.
        """

        load_pe_model = partial(load_core_model, core_model="perception_encoder")
        """Loads the Perception Encoder model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The Perception Encoder model ID.
        """

        load_sam_model = partial(load_core_model, core_model="sam")
        """Loads the SAM model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The SAM model ID.
        """
        load_sam2_model = partial(load_core_model, core_model="sam2")
        """Loads the SAM2 model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The SAM2 model ID.
        """

        load_gaze_model = partial(load_core_model, core_model="gaze")
        """Loads the GAZE model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The GAZE model ID.
        """

        load_doctr_model = partial(load_core_model, core_model="doctr")
        """Loads the DocTR model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The DocTR model ID.
        """

        load_easy_ocr_model = partial(load_core_model, core_model="easy_ocr")
        """Loads the EasyOCR model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The EasyOCR model ID.
        """

        load_paligemma_model = partial(load_core_model, core_model="paligemma")

        load_grounding_dino_model = partial(
            load_core_model, core_model="grounding_dino"
        )
        """Loads the Grounding DINO model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The Grounding DINO model ID.
        """

        load_yolo_world_model = partial(load_core_model, core_model="yolo_world")
        load_owlv2_model = partial(load_core_model, core_model="owlv2")
        """Loads the YOLO World model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The YOLO World model ID.
        """

        load_trocr_model = partial(load_core_model, core_model="trocr")
        """Loads the TrOCR model into the model manager.

        Args:
        Same as `load_core_model`.

        Returns:
        The TrOCR model ID.
        """

        @app.get(
            "/info",
            response_model=ServerVersionInfo,
            summary="Info",
            description="Get the server name and version number",
        )
        def root():
            """Endpoint to get the server name and version number.

            Returns:
                ServerVersionInfo: The server version information.
            """
            return ServerVersionInfo(
                name="Roboflow Inference Server",
                version=__version__,
                uuid=GLOBAL_INFERENCE_SERVER_ID,
            )

        @app.get(
            "/logs",
            summary="Get Recent Logs",
            description="Get recent application logs for debugging",
        )
        def get_logs(
            limit: Optional[int] = Query(
                100, description="Maximum number of log entries to return"
            ),
            level: Optional[str] = Query(
                None,
                description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
            ),
            since: Optional[str] = Query(
                None, description="Return logs since this ISO timestamp"
            ),
        ):
            """Get recent application logs from memory.

            Only available when ENABLE_IN_MEMORY_LOGS environment variable is set to 'true'.

            Args:
                limit: Maximum number of log entries (default 100)
                level: Filter by log level
                since: ISO timestamp to filter logs since

            Returns:
                List of log entries with timestamp, level, logger, and message
            """
            # Check if in-memory logging is enabled
            from inference.core.logging.memory_handler import (
                get_recent_logs,
                is_memory_logging_enabled,
            )

            if not is_memory_logging_enabled():
                raise HTTPException(
                    status_code=404, detail="Logs endpoint not available"
                )

            try:
                logs = get_recent_logs(limit=limit or 100, level=level, since=since)
                return {"logs": logs, "total_count": len(logs)}
            except (ImportError, ModuleNotFoundError):
                raise HTTPException(
                    status_code=500, detail="Logging system not properly initialized"
                )

        if not LAMBDA and GET_MODEL_REGISTRY_ENABLED:

            @app.get(
                "/model/registry",
                response_model=ModelsDescriptions,
                summary="Get model keys",
                description="Get the ID of each loaded model",
            )
            def registry():
                """Get the ID of each loaded model in the registry.

                Returns:
                    ModelsDescriptions: The object containing models descriptions
                """
                logger.debug(f"Reached /model/registry")
                models_descriptions = self.model_manager.describe_models()
                return ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

        # The current AWS Lambda authorizer only supports path parameters, therefore we can only use the legacy infer route. This case statement excludes routes which won't work for the current Lambda authorizer.
        if not (LAMBDA or GCP_SERVERLESS):

            @app.post(
                "/model/add",
                response_model=ModelsDescriptions,
                summary="Load a model",
                description="Load the model with the given model ID",
            )
            @with_route_exceptions
            def model_add(
                request: AddModelRequest,
                countinference: Optional[bool] = None,
                service_secret: Optional[str] = None,
            ):
                """Load the model with the given model ID into the model manager.

                Args:
                    request (AddModelRequest): The request containing the model ID and optional API key.
                    countinference (Optional[bool]): Whether to count inference or not.
                    service_secret (Optional[str]): The service secret for the request.

                Returns:
                    ModelsDescriptions: The object containing models descriptions
                """
                logger.debug(f"Reached /model/add")
                de_aliased_model_id = resolve_roboflow_model_alias(
                    model_id=request.model_id
                )
                logger.info(f"Loading model: {de_aliased_model_id}")
                self.model_manager.add_model(
                    de_aliased_model_id,
                    request.api_key,
                    countinference=countinference,
                    service_secret=service_secret,
                )
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
            def model_remove(request: ClearModelRequest):
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
            def model_clear():
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

        # these NEW endpoints need authentication protection
        if not LAMBDA and not GCP_SERVERLESS:

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
            @usage_collector("request")
            def infer_object_detection(
                inference_request: ObjectDetectionInferenceRequest,
                background_tasks: BackgroundTasks,
                countinference: Optional[bool] = None,
                service_secret: Optional[str] = None,
            ):
                """Run inference with the specified object detection model.

                Args:
                    inference_request (ObjectDetectionInferenceRequest): The request containing the necessary details for object detection.
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks

                Returns:
                    Union[ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]]: The response containing the inference results.
                """
                logger.debug(f"Reached /infer/object_detection")
                return process_inference_request(
                    inference_request,
                    active_learning_eligible=True,
                    background_tasks=background_tasks,
                    countinference=countinference,
                    service_secret=service_secret,
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
            @usage_collector("request")
            def infer_instance_segmentation(
                inference_request: InstanceSegmentationInferenceRequest,
                background_tasks: BackgroundTasks,
                countinference: Optional[bool] = None,
                service_secret: Optional[str] = None,
            ):
                """Run inference with the specified instance segmentation model.

                Args:
                    inference_request (InstanceSegmentationInferenceRequest): The request containing the necessary details for instance segmentation.
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks

                Returns:
                    InstanceSegmentationInferenceResponse: The response containing the inference results.
                """
                logger.debug(f"Reached /infer/instance_segmentation")
                return process_inference_request(
                    inference_request,
                    active_learning_eligible=True,
                    background_tasks=background_tasks,
                    countinference=countinference,
                    service_secret=service_secret,
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
            @usage_collector("request")
            def infer_classification(
                inference_request: ClassificationInferenceRequest,
                background_tasks: BackgroundTasks,
                countinference: Optional[bool] = None,
                service_secret: Optional[str] = None,
            ):
                """Run inference with the specified classification model.

                Args:
                    inference_request (ClassificationInferenceRequest): The request containing the necessary details for classification.
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks

                Returns:
                    Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]: The response containing the inference results.
                """
                logger.debug(f"Reached /infer/classification")
                return process_inference_request(
                    inference_request,
                    active_learning_eligible=True,
                    background_tasks=background_tasks,
                    countinference=countinference,
                    service_secret=service_secret,
                )

            @app.post(
                "/infer/keypoints_detection",
                response_model=Union[KeypointsDetectionInferenceResponse, StubResponse],
                summary="Keypoints detection infer",
                description="Run inference with the specified keypoints detection model",
            )
            @with_route_exceptions
            @usage_collector("request")
            def infer_keypoints(
                inference_request: KeypointsDetectionInferenceRequest,
                countinference: Optional[bool] = None,
                service_secret: Optional[str] = None,
            ):
                """Run inference with the specified keypoints detection model.

                Args:
                    inference_request (KeypointsDetectionInferenceRequest): The request containing the necessary details for keypoints detection.

                Returns:
                    Union[ClassificationInferenceResponse, MultiLabelClassificationInferenceResponse]: The response containing the inference results.
                """
                logger.debug(f"Reached /infer/keypoints_detection")
                return process_inference_request(
                    inference_request,
                    countinference=countinference,
                    service_secret=service_secret,
                )

            if LMM_ENABLED or MOONDREAM2_ENABLED:

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
                @usage_collector("request")
                def infer_lmm(
                    inference_request: LMMInferenceRequest,
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    """Run inference with the specified object detection model.

                    Args:
                        inference_request (ObjectDetectionInferenceRequest): The request containing the necessary details for object detection.
                        background_tasks: (BackgroundTasks) pool of fastapi background tasks

                    Returns:
                        Union[ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]]: The response containing the inference results.
                    """
                    logger.debug(f"Reached /infer/lmm")
                    return process_inference_request(
                        inference_request,
                        countinference=countinference,
                        service_secret=service_secret,
                    )

        if not DISABLE_WORKFLOW_ENDPOINTS:

            @app.post(
                "/{workspace_name}/workflows/{workflow_id}/describe_interface",
                response_model=DescribeInterfaceResponse,
                summary="Endpoint to describe interface of predefined workflow",
                description="Checks Roboflow API for workflow definition, once acquired - describes workflow inputs and outputs",
            )
            @with_route_exceptions
            def describe_predefined_workflow_interface(
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
            def describe_workflow_interface(
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
            @usage_collector("request")
            def infer_from_predefined_workflow(
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
                if not workflow_request.workflow_id:
                    workflow_request.workflow_id = workflow_id
                if not workflow_specification.get("id"):
                    logger.warning(
                        "Internal workflow ID missing in specification for '%s'",
                        workflow_id,
                    )
                return process_workflow_inference_request(
                    workflow_request=workflow_request,
                    workflow_specification=workflow_specification,
                    background_tasks=(
                        background_tasks if not (LAMBDA or GCP_SERVERLESS) else None
                    ),
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
            @usage_collector("request")
            def infer_from_workflow(
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
                    background_tasks=(
                        background_tasks if not (LAMBDA or GCP_SERVERLESS) else None
                    ),
                    profiler=profiler,
                )

            @app.get(
                "/workflows/execution_engine/versions",
                response_model=ExecutionEngineVersions,
                summary="Returns available Execution Engine versions sorted from oldest to newest",
                description="Returns available Execution Engine versions sorted from oldest to newest",
            )
            @with_route_exceptions
            def get_execution_engine_versions() -> ExecutionEngineVersions:
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
            def describe_workflows_blocks(
                request: Request,
            ) -> Union[WorkflowsBlocksDescription, Response]:
                result = handle_describe_workflows_blocks_request()
                return gzip_response_if_requested(request=request, response=result)

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
            def describe_workflows_blocks(
                request: Request,
                request_payload: Optional[DescribeBlocksRequest] = None,
            ) -> Union[WorkflowsBlocksDescription, Response]:
                # TODO: get rid of async: https://github.com/roboflow/inference/issues/569
                dynamic_blocks_definitions = None
                requested_execution_engine_version = None
                api_key = None
                if request_payload is not None:
                    dynamic_blocks_definitions = (
                        request_payload.dynamic_blocks_definitions
                    )
                    requested_execution_engine_version = (
                        request_payload.execution_engine_version
                    )
                    api_key = request_payload.api_key or request.query_params.get(
                        "api_key", None
                    )
                result = handle_describe_workflows_blocks_request(
                    dynamic_blocks_definitions=dynamic_blocks_definitions,
                    requested_execution_engine_version=requested_execution_engine_version,
                    api_key=api_key,
                )
                return gzip_response_if_requested(request=request, response=result)

            @app.get(
                "/workflows/definition/schema",
                response_model=WorkflowsBlocksSchemaDescription,
                summary="Endpoint to fetch the workflows block schema",
                description="Endpoint to fetch the schema of all available blocks. This information can be "
                "used to validate workflow definitions and suggest syntax in the JSON editor.",
            )
            @with_route_exceptions
            def get_workflow_schema(
                request: Request,
            ) -> WorkflowsBlocksSchemaDescription:
                result = get_workflow_schema_description()
                return gzip_response_if_requested(request, response=result)

            @app.post(
                "/workflows/blocks/dynamic_outputs",
                response_model=List[OutputDefinition],
                summary="[EXPERIMENTAL] Endpoint to get definition of dynamic output for workflow step",
                description="Endpoint to be used when step outputs can be discovered only after "
                "filling manifest with data.",
            )
            @with_route_exceptions
            def get_dynamic_block_outputs(
                step_manifest: Dict[str, Any],
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
            def validate_workflow(
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
            @with_route_exceptions_async
            async def list_pipelines(_: Request) -> ListPipelinesResponse:
                return await self.stream_manager_client.list_pipelines()

            @app.get(
                "/inference_pipelines/{pipeline_id}/status",
                response_model=InferencePipelineStatusResponse,
                summary="[EXPERIMENTAL] Get status of InferencePipeline",
                description="[EXPERIMENTAL] Get status of InferencePipeline",
            )
            @with_route_exceptions_async
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
            @with_route_exceptions_async
            async def initialise(request: InitialisePipelinePayload) -> CommandResponse:
                return await self.stream_manager_client.initialise_pipeline(
                    initialisation_request=request
                )

            @app.post(
                "/inference_pipelines/initialise_webrtc",
                response_model=InitializeWebRTCPipelineResponse,
                summary="[EXPERIMENTAL] Establishes WebRTC peer connection and starts new InferencePipeline consuming video track",
                description="[EXPERIMENTAL] Establishes WebRTC peer connection and starts new InferencePipeline consuming video track",
            )
            @with_route_exceptions_async
            async def initialise_webrtc_inference_pipeline(
                request: InitialiseWebRTCPipelinePayload,
            ) -> CommandResponse:
                logger.debug("Received initialise webrtc inference pipeline request")
                resp = await self.stream_manager_client.initialise_webrtc_pipeline(
                    initialisation_request=request
                )
                logger.debug("Returning initialise webrtc inference pipeline response")
                return resp

            @app.post(
                "/inference_pipelines/{pipeline_id}/pause",
                response_model=CommandResponse,
                summary="[EXPERIMENTAL] Pauses the InferencePipeline",
                description="[EXPERIMENTAL] Pauses the InferencePipeline",
            )
            @with_route_exceptions_async
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
            @with_route_exceptions_async
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
            @with_route_exceptions_async
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
            @with_route_exceptions_async
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

        # Enable preloading models at startup
        if (
            (PRELOAD_MODELS or DEDICATED_DEPLOYMENT_WORKSPACE_URL)
            and API_KEY
            and not (LAMBDA or GCP_SERVERLESS)
        ):

            class ModelInitState:
                """Class to track model initialization state."""

                def __init__(self):
                    self.is_ready = False
                    self.lock = Lock()  # For thread-safe updates
                    self.initialization_errors = []  # Track errors per model

            model_init_state = ModelInitState()

            def initialize_models(state: ModelInitState):
                """Perform asynchronous initialization tasks to load models."""
                # Limit the number of concurrent tasks to prevent resource exhaustion

                def load_model(model_id):
                    logger.debug(f"load_model({model_id}) - starting", flush=True)
                    try:
                        # TODO: how to add timeout here? Probably best to timeout model loading?
                        model_add(
                            AddModelRequest(
                                model_id=model_id,
                                model_type=None,
                                api_key=API_KEY,
                            )
                        )
                        logger.info(f"Model {model_id} loaded successfully.")
                    except Exception as e:
                        error_msg = f"Error loading model {model_id}: {e}"
                        logger.error(error_msg)
                        with state.lock:
                            state.initialization_errors.append((model_id, str(e)))
                    logger.debug(f"load_model({model_id}) - finished", flush=True)

                if PRELOAD_MODELS:
                    # Create tasks for each model to be loaded
                    model_loading_executor = ThreadPoolExecutor(max_workers=2)
                    loaded_futures: List[Tuple[str, Future]] = []
                    for model_id in PRELOAD_MODELS:
                        future = model_loading_executor.submit(
                            load_model, model_id=model_id
                        )
                        loaded_futures.append((model_id, future))

                    for model_id, future in loaded_futures:
                        try:
                            future.result(timeout=300)
                        except (
                            TimeoutError,
                            CancelledError,
                            concurrent.futures.TimeoutError,
                        ):
                            state.initialization_errors.append(
                                (
                                    model_id,
                                    "Could not finalise model loading before timeout",
                                )
                            )
                            future.cancel()

                # Update the readiness state in a thread-safe manner
                with state.lock:
                    state.is_ready = True

            @app.on_event("startup")
            def startup_model_init():
                """Initialize the models on startup."""
                startup_thread = Thread(
                    target=initialize_models, args=(model_init_state,), daemon=True
                )
                startup_thread.start()
                logger.info("Model initialization started in the background.")

            @app.get("/readiness", status_code=200)
            def readiness(
                state: ModelInitState = Depends(lambda: model_init_state),
            ):
                """Readiness endpoint for Kubernetes readiness probe."""
                with state.lock:
                    if state.is_ready:
                        return {"status": "ready"}
                    else:
                        return JSONResponse(
                            content={"status": "not ready"}, status_code=503
                        )

            @app.get("/healthz", status_code=200)
            def healthz():
                """Health endpoint for Kubernetes liveness probe."""
                return {"status": "healthy"}

        if CORE_MODELS_ENABLED:
            if CORE_MODEL_CLIP_ENABLED:

                @app.post(
                    "/clip/embed_image",
                    response_model=ClipEmbeddingResponse,
                    summary="CLIP Image Embeddings",
                    description="Run the Open AI CLIP model to embed image data.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def clip_embed_image(
                    inference_request: ClipImageEmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    clip_model_id = load_clip_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
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
                @usage_collector("request")
                def clip_embed_text(
                    inference_request: ClipTextEmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    clip_model_id = load_clip_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
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
                @usage_collector("request")
                def clip_compare(
                    inference_request: ClipCompareRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    clip_model_id = load_clip_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        clip_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(clip_model_id, actor, n=2)
                    return response

            if CORE_MODEL_PE_ENABLED:

                @app.post(
                    "/perception_encoder/embed_image",
                    response_model=PerceptionEncoderEmbeddingResponse,
                    summary="PE Image Embeddings",
                    description="Run the Meta Perception Encoder model to embed image data.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def pe_embed_image(
                    inference_request: PerceptionEncoderImageEmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    """
                    Embeds image data using the Perception Encoder PE model.

                    Args:
                        inference_request (PerceptionEncoderImageEmbeddingRequest): The request containing the image to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        PerceptionEncoderEmbeddingResponse: The response containing the embedded image.
                    """
                    logger.debug(f"Reached /perception_encoder/embed_image")
                    pe_model_id = load_pe_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        pe_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(pe_model_id, actor)
                    return response

                @app.post(
                    "/perception_encoder/embed_text",
                    response_model=PerceptionEncoderEmbeddingResponse,
                    summary="Perception Encoder Text Embeddings",
                    description="Run the Meta Perception Encoder model to embed text data.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def pe_embed_text(
                    inference_request: PerceptionEncoderTextEmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    """
                    Embeds text data using the Meta Perception Encoder model.

                    Args:
                        inference_request (PerceptionEncoderTextEmbeddingRequest): The request containing the text to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        PerceptionEncoderEmbeddingResponse: The response containing the embedded text.
                    """
                    logger.debug(f"Reached /perception_encoder/embed_text")
                    pe_model_id = load_pe_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        pe_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(pe_model_id, actor)
                    return response

                @app.post(
                    "/perception_encoder/compare",
                    response_model=PerceptionEncoderCompareResponse,
                    summary="Perception Encoder Compare",
                    description="Run the Meta Perception Encoder model to compute similarity scores.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def pe_compare(
                    inference_request: PerceptionEncoderCompareRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    """
                    Computes similarity scores using the Meta Perception Encoder model.

                    Args:
                        inference_request (PerceptionEncoderCompareRequest): The request containing the data to be compared.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        PerceptionEncoderCompareResponse: The response containing the similarity scores.
                    """
                    logger.debug(f"Reached /perception_encoder/compare")
                    pe_model_id = load_pe_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        pe_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(pe_model_id, actor, n=2)
                    return response

            if CORE_MODEL_GROUNDINGDINO_ENABLED:

                @app.post(
                    "/grounding_dino/infer",
                    response_model=ObjectDetectionInferenceResponse,
                    summary="Grounding DINO inference.",
                    description="Run the Grounding DINO zero-shot object detection model.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def grounding_dino_infer(
                    inference_request: GroundingDINOInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
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
                @usage_collector("request")
                def yolo_world_infer(
                    inference_request: YOLOWorldInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    logger.debug("YOLOWorld model loaded. Staring the inference.")
                    response = self.model_manager.infer_from_request_sync(
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
                    response_model=Union[
                        OCRInferenceResponse, List[OCRInferenceResponse]
                    ],
                    summary="DocTR OCR response",
                    description="Run the DocTR OCR model to retrieve text in an image.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def doctr_retrieve_text(
                    inference_request: DoctrOCRInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    """
                    Embeds image data using the DocTR model.

                    Args:
                        inference_request (M.DoctrOCRInferenceRequest): The request containing the image from which to retrieve text.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        OCRInferenceResponse: The response containing the embedded image.
                    """
                    logger.debug(f"Reached /doctr/ocr")
                    doctr_model_id = load_doctr_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        doctr_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(doctr_model_id, actor)
                    return orjson_response_keeping_parent_id(response)

            if CORE_MODEL_EASYOCR_ENABLED:

                @app.post(
                    "/easy_ocr/ocr",
                    response_model=Union[
                        OCRInferenceResponse, List[OCRInferenceResponse]
                    ],
                    summary="EasyOCR OCR response",
                    description="Run the EasyOCR model to retrieve text in an image.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def easy_ocr_retrieve_text(
                    inference_request: EasyOCRInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    """
                    Embeds image data using the EasyOCR model.

                    Args:
                        inference_request (EasyOCRInferenceRequest): The request containing the image from which to retrieve text.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        OCRInferenceResponse: The response containing the embedded image.
                    """
                    logger.debug(f"Reached /easy_ocr/ocr")
                    easy_ocr_model_id = load_easy_ocr_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        easy_ocr_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(easy_ocr_model_id, actor)
                    return orjson_response_keeping_parent_id(response)

            if CORE_MODEL_SAM_ENABLED:

                @app.post(
                    "/sam/embed_image",
                    response_model=SamEmbeddingResponse,
                    summary="SAM Image Embeddings",
                    description="Run the Meta AI Segmant Anything Model to embed image data.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def sam_embed_image(
                    inference_request: SamEmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    sam_model_id = load_sam_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    model_response = self.model_manager.infer_from_request_sync(
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
                @usage_collector("request")
                def sam_segment_image(
                    inference_request: SamSegmentationRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    sam_model_id = load_sam_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    model_response = self.model_manager.infer_from_request_sync(
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
                @usage_collector("request")
                def sam2_embed_image(
                    inference_request: Sam2EmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    sam2_model_id = load_sam2_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    model_response = self.model_manager.infer_from_request_sync(
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
                @usage_collector("request")
                def sam2_segment_image(
                    inference_request: Sam2SegmentationRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    sam2_model_id = load_sam2_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    model_response = self.model_manager.infer_from_request_sync(
                        sam2_model_id, inference_request
                    )
                    if inference_request.format == "binary":
                        return Response(
                            content=model_response,
                            headers={"Content-Type": "application/octet-stream"},
                        )
                    return model_response

            if CORE_MODEL_SAM3_ENABLED:

                @app.post(
                    "/sam3/embed_image",
                    response_model=Sam3EmbeddingResponse,
                    summary="Seg preview Image Embeddings",
                    description="Run the  Model to embed image data.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def sam3_embed_image(
                    inference_request: Sam2EmbeddingRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    logger.debug(f"Reached /sam3/embed_image")

                    from inference.models.sam3.interactive_image_segmentation import (
                        Sam3ForInteractiveImageSegmentation,
                    )

                    self.model_manager.add_model(
                        "sam3/sam3_video_model_only",
                        api_key=api_key,
                        endpoint_type=ModelEndpointType.CORE_MODEL,
                        countinference=countinference,
                        service_secret=service_secret,
                        model_class_override=Sam3ForInteractiveImageSegmentation,
                    )

                    model_response = self.model_manager.infer_from_request_sync(
                        "sam3/sam3_video_model_only", inference_request
                    )
                    return model_response

                @app.post(
                    "/sam3/visual_segment",
                    response_model=Sam2SegmentationResponse,
                    summary="Seg preview Image Segmentation",
                    description="Run the segmentations for image data.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def sam3_visual_segment(
                    inference_request: Sam2SegmentationRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):

                    from inference.models.sam3.interactive_image_segmentation import (
                        Sam3ForInteractiveImageSegmentation,
                    )

                    self.model_manager.add_model(
                        "sam3/sam3_video_model_only",
                        api_key=api_key,
                        endpoint_type=ModelEndpointType.CORE_MODEL,
                        countinference=countinference,
                        service_secret=service_secret,
                        model_class_override=Sam3ForInteractiveImageSegmentation,
                    )

                    model_response = self.model_manager.infer_from_request_sync(
                        "sam3/sam3_video_model_only", inference_request
                    )
                    return model_response

                @app.post(
                    "/seg-preview/segment_image",
                    response_model=Sam3SegmentationResponse,
                    summary="Seg preview Image Segmentation",
                    description="Run the segmentations for image data.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def sam3_segment_image(
                    inference_request: Sam3SegmentationRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    logger.debug(f"Reached /sam3/segment_image")

                    if inference_request.model_id.startswith("sam3/"):
                        self.model_manager.add_model(
                            inference_request.model_id,
                            api_key=api_key,
                            endpoint_type=ModelEndpointType.CORE_MODEL,
                            countinference=countinference,
                            service_secret=service_secret,
                        )
                    else:
                        self.model_manager.add_model(
                            inference_request.model_id,
                            api_key=api_key,
                            endpoint_type=ModelEndpointType.ORT,
                            countinference=countinference,
                            service_secret=service_secret,
                        )

                    model_response = self.model_manager.infer_from_request_sync(
                        inference_request.model_id, inference_request
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
                @usage_collector("request")
                def owlv2_infer(
                    inference_request: OwlV2InferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    owl2_model_id = load_owlv2_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    model_response = self.model_manager.infer_from_request_sync(
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
                @usage_collector("request")
                def gaze_detection(
                    inference_request: GazeDetectionInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                    gaze_model_id = load_gaze_model(
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        gaze_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(gaze_model_id, actor)
                    return response

            if DEPTH_ESTIMATION_ENABLED:

                @app.post(
                    "/infer/depth-estimation",
                    response_model=DepthEstimationResponse,
                    summary="Depth Estimation",
                    description="Run the depth estimation model to generate a depth map.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def depth_estimation(
                    inference_request: DepthEstimationRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
                ):
                    """
                    Generate a depth map using the depth estimation model.

                    Args:
                        inference_request (DepthEstimationRequest): The request containing the image to estimate depth for.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        DepthEstimationResponse: The response containing the normalized depth map and optional visualization.
                    """
                    logger.debug(f"Reached /infer/depth-estimation")
                    depth_model_id = inference_request.model_id
                    self.model_manager.add_model(
                        depth_model_id,
                        inference_request.api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        depth_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(depth_model_id, actor)

                    # Extract data from nested response structure
                    depth_data = response.response
                    depth_response = DepthEstimationResponse(
                        normalized_depth=depth_data["normalized_depth"].tolist(),
                        image=depth_data["image"].numpy_image.tobytes().hex(),
                    )
                    return depth_response

            if CORE_MODEL_TROCR_ENABLED:

                @app.post(
                    "/ocr/trocr",
                    response_model=OCRInferenceResponse,
                    summary="TrOCR OCR response",
                    description="Run the TrOCR model to retrieve text in an image.",
                )
                @with_route_exceptions
                @usage_collector("request")
                def trocr_retrieve_text(
                    inference_request: TrOCRInferenceRequest,
                    request: Request,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    countinference: Optional[bool] = None,
                    service_secret: Optional[str] = None,
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
                        inference_request,
                        api_key=api_key,
                        countinference=countinference,
                        service_secret=service_secret,
                    )
                    response = self.model_manager.infer_from_request_sync(
                        trocr_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(trocr_model_id, actor)
                    return orjson_response_keeping_parent_id(response)

        if not (LAMBDA or GCP_SERVERLESS):

            @app.get(
                "/notebook/start",
                summary="Jupyter Lab Server Start",
                description="Starts a jupyter lab server for running development code",
            )
            @with_route_exceptions
            def notebook_start(browserless: bool = False):
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

        if ENABLE_BUILDER:
            from inference.core.interfaces.http.builder.routes import (
                router as builder_router,
            )

            # Allow CORS on only the API, but not the builder UI/iframe (where the CSRF is passed)
            app.add_middleware(
                PathAwareCORSMiddleware,
                match_paths=r"^/build/api.*",
                allow_origins=[BUILDER_ORIGIN],
                allow_methods=["*"],
                allow_headers=["*"],
                allow_credentials=True,
            )

            # Attach all routes from builder to the /build prefix
            app.include_router(builder_router, prefix="/build", tags=["builder"])

        if LEGACY_ROUTE_ENABLED:
            # Legacy object detection inference path for backwards compatibility
            @app.get(
                "/{dataset_id}/{version_id:str}",
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
                "/{dataset_id}/{version_id:str}",
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
            @usage_collector("request")
            def legacy_infer_from_request(
                background_tasks: BackgroundTasks,
                request: Request,
                request_body: Annotated[
                    Optional[Union[bytes, UploadFile]],
                    Depends(parse_body_content_for_legacy_request_handler),
                ],
                dataset_id: str = Path(
                    description="ID of a Roboflow dataset corresponding to the model to use for inference OR workspace ID"
                ),
                version_id: str = Path(
                    description="ID of a Roboflow dataset version corresponding to the model to use for inference OR model ID"
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
                disable_model_monitoring: Optional[bool] = Query(
                    False,
                    description="If true, disables model monitoring for this request",
                    include_in_schema=False,
                ),
            ):
                """
                Legacy inference endpoint for object detection, instance segmentation, and classification.

                Args:
                    background_tasks: (BackgroundTasks) pool of fastapi background tasks
                    dataset_id (str): ID of a Roboflow dataset corresponding to the model to use for inference OR workspace ID
                    version_id (str): ID of a Roboflow dataset version corresponding to the model to use for inference OR model ID
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
                elif confidence < CONFIDENCE_LOWER_BOUND_OOM_PREVENTION:
                    # allowing lower confidence results in RAM usage explosion
                    confidence = CONFIDENCE_LOWER_BOUND_OOM_PREVENTION

                if overlap >= 1:
                    overlap /= 100
                if image is not None:
                    request_image = InferenceRequestImage(type="url", value=image)
                else:
                    if "Content-Type" not in request.headers:
                        raise ContentTypeMissing(
                            f"Request must include a Content-Type header"
                        )
                    if isinstance(request_body, UploadFile):
                        base64_image_str = request_body.file.read()
                        base64_image_str = base64.b64encode(base64_image_str)
                        request_image = InferenceRequestImage(
                            type="base64", value=base64_image_str.decode("ascii")
                        )
                    elif isinstance(request_body, bytes):
                        request_image = InferenceRequestImage(
                            type=image_type, value=request_body
                        )
                    elif request_body is None:
                        raise InputImageLoadError(
                            message="Image not found in request body.",
                            public_message="Image not found in request body.",
                        )
                    else:
                        raise ContentTypeInvalid(
                            f"Invalid Content-Type: {request.headers['Content-Type']}"
                        )

                if not countinference and service_secret != ROBOFLOW_SERVICE_SECRET:
                    raise MissingServiceSecretError(
                        "Service secret is required to disable inference usage tracking"
                    )
                if LAMBDA:
                    logger.debug("request.scope: %s", request.scope)
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
                        logger.info("Not counting inference for usage")
                else:
                    request_model_id = model_id
                logger.debug(
                    f"State of model registry: {self.model_manager.describe_models()}"
                )
                self.model_manager.add_model(
                    request_model_id,
                    api_key,
                    model_id_alias=model_id,
                    countinference=countinference,
                    service_secret=service_secret,
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
                    usage_billable=countinference,
                    disable_model_monitoring=disable_model_monitoring,
                    **args,
                )
                inference_response = self.model_manager.infer_from_request_sync(
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

        if not (LAMBDA or GCP_SERVERLESS):
            # Legacy clear cache endpoint for backwards compatibility
            @app.get("/clear_cache", response_model=str)
            def legacy_clear_cache():
                """
                Clears the model cache.

                This endpoint provides a way to clear the cache of loaded models.

                Returns:
                    str: A string indicating that the cache has been cleared.
                """
                logger.debug(f"Reached /clear_cache")
                model_clear()
                return "Cache Cleared"

            # Legacy add model endpoint for backwards compatibility
            @app.get("/start/{dataset_id}/{version_id}")
            def model_add_legacy(
                dataset_id: str,
                version_id: str,
                api_key: str = None,
                countinference: Optional[bool] = None,
                service_secret: Optional[str] = None,
            ):
                """
                Starts a model inference session.

                This endpoint initializes and starts an inference session for the specified model version.

                Args:
                    dataset_id (str): ID of a Roboflow dataset corresponding to the model.
                    version_id (str): ID of a Roboflow dataset version corresponding to the model.
                    api_key (str, optional): Roboflow API Key for artifact retrieval.
                    countinference (Optional[bool]): Whether to count inference or not.
                    service_secret (Optional[str]): The service secret for the request.

                Returns:
                    JSONResponse: A response object containing the status and a success message.
                """
                logger.debug(
                    f"Reached /start/{dataset_id}/{version_id} with {dataset_id}/{version_id}"
                )
                model_id = f"{dataset_id}/{version_id}"
                self.model_manager.add_model(
                    model_id,
                    api_key,
                    countinference=countinference,
                    service_secret=service_secret,
                )

                return JSONResponse(
                    {
                        "status": 200,
                        "message": "inference session started from local memory.",
                    }
                )

        if not ENABLE_DASHBOARD:

            @app.get("/dashboard.html")
            @app.head("/dashboard.html")
            async def dashboard_guard():
                return Response(status_code=404)

        @app.exception_handler(InputImageLoadError)
        async def unicorn_exception_handler(request: Request, exc: InputImageLoadError):
            return JSONResponse(
                status_code=400,
                content={
                    "message": f"Could not load input image. Cause: {exc.get_public_error_details()}"
                },
            )

        app.mount(
            "/",
            StaticFiles(directory="./inference/landing/out", html=True),
            name="root",
        )

    def run(self):
        uvicorn.run(self.app, host="127.0.0.1", port=8080)


def load_gaze_model(
    inference_request: GazeDetectionInferenceRequest, api_key: Optional[str] = None
) -> str:
    """Loads the gaze detection model.

    Args:
        inference_request (GazeDetectionInferenceRequest): The inference request.
        api_key (Optional[str], default None): The Roboflow API key.

    Returns:
        str: The model ID.
    """
    return inference_request.model_id
