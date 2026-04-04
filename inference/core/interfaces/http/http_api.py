import base64
import concurrent
import os
import re
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from functools import partial
from threading import Lock, Thread
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import asgi_correlation_id
import requests
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
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi_cprofile.profiler import CProfileMiddleware
from pydantic import ValidationError
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
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
    SemanticSegmentationInferenceRequest,
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
from inference.core.entities.requests.sam3 import Sam3SegmentationRequest
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
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
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    SemanticSegmentationInferenceResponse,
    StubResponse,
)
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
    API_BASE_URL,
    API_LOGGING_ENABLED,
    BUILDER_ORIGIN,
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
    ENABLE_STREAM_API,
    ENABLE_WORKFLOWS_PROFILING,
    GCP_SERVERLESS,
    GET_MODEL_REGISTRY_ENABLED,
    HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED,
    HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_WORKERS,
    LAMBDA,
    LMM_ENABLED,
    METRICS_ENABLED,
    PINNED_MODELS,
    PRELOAD_API_KEY,
    PRELOAD_MODELS,
    PROFILE,
    ROBOFLOW_INTERNAL_SERVICE_NAME,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    SAM3_EXEC_MODE,
    SAM3_FINE_TUNED_MODELS_ENABLED,
    USE_INFERENCE_MODELS,
    WEBRTC_WORKER_ENABLED,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
    WORKFLOWS_PROFILER_BUFFER_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING,
    WORKFLOWS_STEP_EXECUTION_MODE,
)
from inference.core.exceptions import (
    InputImageLoadError,
    MissingApiKeyError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    WebRTCConfigurationError,
    WorkspaceLoadError,
)
from inference.core.interfaces.base import BaseInterface
from inference.core.interfaces.http.error_handlers import (
    with_route_exceptions,
)
from inference.core.interfaces.http.routes.info import create_info_router
from inference.core.interfaces.http.routes.inference import create_inference_router
from inference.core.interfaces.http.routes.models import create_models_router
from inference.core.interfaces.http.routes.core_models import create_core_models_router
from inference.core.interfaces.http.routes.stream import create_stream_router
from inference.core.interfaces.http.routes.webrtc import create_webrtc_worker_router
from inference.core.interfaces.http.routes.notebook import create_notebook_router
from inference.core.interfaces.http.routes.workflows import create_workflows_router
from inference.core.interfaces.http.routes.legacy import create_legacy_router
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
    ListPipelinesResponse,
)
from inference.core.interfaces.stream_manager.api.stream_manager_client import (
    StreamManagerClient,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    ConsumeResultsPayload,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
    OperationStatus,
)
from inference.core.managers.base import ModelManager
from inference.core.managers.metrics import get_container_stats
from inference.core.managers.prometheus import InferenceInstrumentator
from inference.core.roboflow_api import (
    build_roboflow_api_headers,
    get_roboflow_workspace,
    get_roboflow_workspace_async,
    get_workflow_specification,
)
from inference.core.utils.container import is_docker_socket_mounted
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import WorkflowError, WorkflowSyntaxError
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
    from inference_sdk.config import (
        EXECUTION_ID_HEADER,
        INTERNAL_REMOTE_EXEC_REQ_HEADER,
        INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER,
        RemoteProcessingTimeCollector,
        apply_duration_minimum,
        execution_id,
        remote_processing_times,
    )
except ImportError:
    execution_id = None
    remote_processing_times = None
    RemoteProcessingTimeCollector = None
    EXECUTION_ID_HEADER = None
    INTERNAL_REMOTE_EXEC_REQ_HEADER = None
    INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER = None
    apply_duration_minimum = None


def get_content_type(request: Request) -> str:
    content_type = request.headers.get("content-type", "")
    return content_type.split(";")[0].strip()


class LambdaMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        logger.info("Lambda is terminating, handle unsent usage payloads.")
        await usage_collector.async_push_usage_payloads()
        return response


REMOTE_PROCESSING_TIME_HEADER = "X-Remote-Processing-Time"
REMOTE_PROCESSING_TIMES_HEADER = "X-Remote-Processing-Times"


class GCPServerlessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if execution_id is not None:
            execution_id_value = request.headers.get(EXECUTION_ID_HEADER)
            if not execution_id_value:
                execution_id_value = f"{time.time_ns()}_{uuid4().hex[:4]}"
            execution_id.set(execution_id_value)
        is_verified_internal = False
        if apply_duration_minimum is not None:
            is_verified_internal = bool(
                ROBOFLOW_INTERNAL_SERVICE_SECRET
                and INTERNAL_REMOTE_EXEC_REQ_HEADER
                and request.headers.get(INTERNAL_REMOTE_EXEC_REQ_HEADER)
                == ROBOFLOW_INTERNAL_SERVICE_SECRET
            )
            apply_duration_minimum.set(not is_verified_internal)
        collector = None
        if (
            WORKFLOWS_REMOTE_EXECUTION_TIME_FORWARDING
            and remote_processing_times is not None
            and RemoteProcessingTimeCollector is not None
        ):
            collector = RemoteProcessingTimeCollector()
            remote_processing_times.set(collector)
        t1 = time.time()
        response = await call_next(request)
        t2 = time.time()
        response.headers[PROCESSING_TIME_HEADER] = str(t2 - t1)
        if collector is not None and collector.has_data():
            total, detail = collector.summarize()
            response.headers[REMOTE_PROCESSING_TIME_HEADER] = str(total)
            if detail is not None:
                response.headers[REMOTE_PROCESSING_TIMES_HEADER] = detail
        if execution_id is not None:
            response.headers[EXECUTION_ID_HEADER] = execution_id_value
        if INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER is not None:
            response.headers[INTERNAL_REMOTE_EXEC_REQ_VERIFIED_HEADER] = str(
                is_verified_internal
            ).lower()
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
                expose_headers=[
                    PROCESSING_TIME_HEADER,
                    REMOTE_PROCESSING_TIME_HEADER,
                    REMOTE_PROCESSING_TIMES_HEADER,
                ],
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
                        "/healthz",  # health check endpoint for liveness probe
                        "/readiness",
                        "/metrics",
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
                        get_content_type(request) == "application/json"
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
                    and get_content_type(request) == "application/json"
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
                        "/healthz",  # health check endpoint for liveness probe
                        "/readiness",
                        "/metrics",
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
                    and get_content_type(request) == "application/json"
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

        @app.middleware("http")
        async def add_inference_engine_headers(request: Request, call_next):
            response = await call_next(request)
            inference_engine = (
                "inference-models" if USE_INFERENCE_MODELS else "old-inference"
            )
            response.headers["x-inference-engine"] = inference_engine
            return response

        self.app = app
        self.model_manager = model_manager
        self.stream_manager_client: Optional[StreamManagerClient] = None
        self.shared_thread_pool_executor: Optional[ThreadPoolExecutor] = None
        if HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_ENABLED:
            self.shared_thread_pool_executor = ThreadPoolExecutor(
                max_workers=HTTP_API_SHARED_WORKFLOWS_THREAD_POOL_WORKERS
            )

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

        app.include_router(create_info_router())

        if (not LAMBDA and GET_MODEL_REGISTRY_ENABLED) or not (LAMBDA or GCP_SERVERLESS):

            app.include_router(
                create_models_router(model_manager=self.model_manager)
                )

        # these NEW endpoints need authentication protection
        if not LAMBDA and not GCP_SERVERLESS:

            app.include_router(create_inference_router(model_manager=self.model_manager))

        if not DISABLE_WORKFLOW_ENDPOINTS:
            app.include_router(
                create_workflows_router(
                    model_manager=model_manager,
                    shared_thread_pool_executor=self.shared_thread_pool_executor,
                )
            )

        if WEBRTC_WORKER_ENABLED:

            app.include_router(create_webrtc_worker_router())

        if ENABLE_STREAM_API:
            app.include_router(
                create_stream_router(stream_manager_client=self.stream_manager_client)
                )

        # Enable preloading models at startup
        if (
            (PRELOAD_MODELS or PINNED_MODELS or DEDICATED_DEPLOYMENT_WORKSPACE_URL)
            and PRELOAD_API_KEY
            and (PINNED_MODELS or not (LAMBDA or GCP_SERVERLESS))
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

                def load_model(model_id):
                    t_start = time.perf_counter()
                    de_aliased = resolve_roboflow_model_alias(model_id=model_id)
                    logger.info(
                        f"Preload: starting model load for '{model_id}' (resolved: '{de_aliased}')"
                    )
                    try:
                        self.model_manager.add_model(
                            de_aliased,
                            PRELOAD_API_KEY,
                        )
                        load_time = time.perf_counter() - t_start
                        logger.info(
                            f"Preload: model '{model_id}' loaded successfully in {load_time:.1f}s"
                        )
                    except Exception as e:
                        load_time = time.perf_counter() - t_start
                        error_msg = f"Preload: error loading model '{model_id}' after {load_time:.1f}s: {e}"
                        logger.error(error_msg)
                        with state.lock:
                            state.initialization_errors.append((model_id, str(e)))
                        return

                    # Pin if this model is in PINNED_MODELS
                    if (
                        PINNED_MODELS
                        and model_id in PINNED_MODELS
                        and hasattr(self.model_manager, "pin_model")
                    ):
                        self.model_manager.pin_model(de_aliased)

                all_models = list(
                    dict.fromkeys((PRELOAD_MODELS or []) + (PINNED_MODELS or []))
                )
                if all_models:
                    # Create tasks for each model to be loaded
                    model_loading_executor = ThreadPoolExecutor(max_workers=2)
                    loaded_futures: List[Tuple[str, Future]] = []
                    for model_id in all_models:
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
            app.include_router(
                create_core_models_router(model_manager=self.model_manager)
            )

        if not (LAMBDA or GCP_SERVERLESS):

            app.include_router(create_notebook_router())

        if ENABLE_BUILDER:
            from inference.core.interfaces.http.builder.routes import (
                router as builder_router,
            )

            # Allow CORS on builder API and workflow endpoints needed by the builder UI
            # Enables Private Network Access for Chrome 142+ (local development)
            app.add_middleware(
                PathAwareCORSMiddleware,
                match_paths=r"^/(build/api|workflows/).*",
                allow_origins=[BUILDER_ORIGIN],
                allow_methods=["*"],
                allow_headers=["*"],
                allow_credentials=True,
                allow_private_network=True,
            )

            # Attach all routes from builder to the /build prefix
            app.include_router(builder_router, prefix="/build", tags=["builder"])

        # Legacy router: infer route when LEGACY_ROUTE_ENABLED; clear_cache/start when not (LAMBDA or GCP_SERVERLESS)
        app.include_router(
            create_legacy_router(model_manager=self.model_manager)
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
