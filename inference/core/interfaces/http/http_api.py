import base64
import traceback
from functools import partial, wraps
from time import sleep
from typing import Any, List, Optional, Union

import asgi_correlation_id
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi_cprofile.profiler import CProfileMiddleware

from inference.core import logger
from inference.core.cache import cache
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
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.requests.sam import (
    SamEmbeddingRequest,
    SamSegmentationRequest,
)
from inference.core.entities.requests.server_state import (
    AddModelRequest,
    ClearModelRequest,
)
from inference.core.entities.requests.workflows import (
    WorkflowInferenceRequest,
    WorkflowSpecificationInferenceRequest,
)
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.entities.responses.clip import (
    ClipCompareResponse,
    ClipEmbeddingResponse,
)
from inference.core.entities.responses.cogvlm import CogVLMResponse
from inference.core.entities.responses.doctr import DoctrOCRInferenceResponse
from inference.core.entities.responses.gaze import GazeDetectionInferenceResponse
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    StubResponse,
)
from inference.core.entities.responses.notebooks import NotebookStartResponse
from inference.core.entities.responses.sam import (
    SamEmbeddingResponse,
    SamSegmentationResponse,
)
from inference.core.entities.responses.server_state import (
    ModelsDescriptions,
    ServerVersionInfo,
)
from inference.core.entities.responses.workflows import WorkflowInferenceResponse
from inference.core.env import (
    ALLOW_ORIGINS,
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_COGVLM_ENABLED,
    CORE_MODEL_DOCTR_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_GROUNDINGDINO_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODEL_YOLO_WORLD_ENABLED,
    CORE_MODELS_ENABLED,
    DISABLE_WORKFLOW_ENDPOINTS,
    LAMBDA,
    LEGACY_ROUTE_ENABLED,
    METLO_KEY,
    METRICS_ENABLED,
    NOTEBOOK_ENABLED,
    NOTEBOOK_PASSWORD,
    NOTEBOOK_PORT,
    PROFILE,
    ROBOFLOW_SERVICE_SECRET,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
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
from inference.core.interfaces.http.orjson_utils import (
    orjson_response,
    serialise_workflow_result,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_workflow_specification
from inference.core.utils.notebooks import start_notebook
from inference.enterprise.workflows.complier.core import compile_and_execute_async
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.steps_executors.active_learning_middlewares import (
    WorkflowsActiveLearningMiddleware,
)
from inference.enterprise.workflows.errors import (
    ExecutionEngineError,
    RuntimePayloadError,
    WorkflowsCompilerError,
)
from inference.models.aliases import resolve_roboflow_model_alias

if LAMBDA:
    from inference.core.usage import trackUsage
if METLO_KEY:
    from metlo.fastapi import ASGIMiddleware

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
        except RuntimePayloadError as e:
            resp = JSONResponse(
                status_code=400, content={"message": e.get_public_message()}
            )
            traceback.print_exc()
        except RoboflowAPINotAuthorizedError:
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
        except (
            WorkflowsCompilerError,
            ExecutionEngineError,
        ) as e:
            resp = JSONResponse(
                status_code=500, content={"message": e.get_public_message()}
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
        except Exception:
            resp = JSONResponse(status_code=500, content={"message": "Internal error."})
            traceback.print_exc()
        return resp

    return wrapped_route


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
        if METLO_KEY:
            app.add_middleware(
                ASGIMiddleware, host="https://app.metlo.com", api_key=METLO_KEY
            )

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
                if response.status_code >= 400:
                    self.model_manager.num_errors += 1
                return response

        self.app = app
        self.model_manager = model_manager
        self.workflows_active_learning_middleware = WorkflowsActiveLearningMiddleware(
            cache=cache,
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

        async def process_workflow_inference_request(
            workflow_request: WorkflowInferenceRequest,
            workflow_specification: dict,
            background_tasks: Optional[BackgroundTasks],
        ) -> WorkflowInferenceResponse:
            step_execution_mode = StepExecutionMode(WORKFLOWS_STEP_EXECUTION_MODE)
            result = await compile_and_execute_async(
                workflow_specification=workflow_specification,
                runtime_parameters=workflow_request.inputs,
                model_manager=model_manager,
                api_key=workflow_request.api_key,
                max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
                step_execution_mode=step_execution_mode,
                active_learning_middleware=self.workflows_active_learning_middleware,
                background_tasks=background_tasks,
            )
            outputs = serialise_workflow_result(
                result=result,
                excluded_fields=workflow_request.excluded_fields,
            )
            response = WorkflowInferenceResponse(outputs=outputs)
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
        """Loads the YOLO World model into the model manager.

        Args:
        inference_request: The request containing version and other details.
        api_key: The API key for the request.

        Returns:
        The YOLO World model ID.
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

        if not DISABLE_WORKFLOW_ENDPOINTS:

            @app.post(
                "/infer/workflows/{workspace_name}/{workflow_name}",
                response_model=WorkflowInferenceResponse,
                summary="Endpoint to trigger inference from predefined workflow",
                description="Checks Roboflow API for workflow definition, once acquired - parses and executes injecting runtime parameters from request body",
            )
            @with_route_exceptions
            async def infer_from_predefined_workflow(
                workspace_name: str,
                workflow_name: str,
                workflow_request: WorkflowInferenceRequest,
                background_tasks: BackgroundTasks,
            ) -> WorkflowInferenceResponse:
                workflow_specification = get_workflow_specification(
                    api_key=workflow_request.api_key,
                    workspace_id=workspace_name,
                    workflow_name=workflow_name,
                )
                return await process_workflow_inference_request(
                    workflow_request=workflow_request,
                    workflow_specification=workflow_specification,
                    background_tasks=background_tasks if not LAMBDA else None,
                )

            @app.post(
                "/infer/workflows",
                response_model=WorkflowInferenceResponse,
                summary="Endpoint to trigger inference from workflow specification provided in payload",
                description="Parses and executes workflow specification, injecting runtime parameters from request body",
            )
            @with_route_exceptions
            async def infer_from_workflow(
                workflow_request: WorkflowSpecificationInferenceRequest,
                background_tasks: BackgroundTasks,
            ) -> WorkflowInferenceResponse:
                workflow_specification = {
                    "specification": workflow_request.specification
                }
                return await process_workflow_inference_request(
                    workflow_request=workflow_request,
                    workflow_specification=workflow_specification,
                    background_tasks=background_tasks if not LAMBDA else None,
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
                    response_model=DoctrOCRInferenceResponse,
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
                        M.DoctrOCRInferenceResponse: The response containing the embedded image.
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

        if LEGACY_ROUTE_ENABLED:
            # Legacy object detection inference path for backwards compatability
            @app.post(
                "/{dataset_id}/{version_id}",
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
                    visualize_predictions=True if format == "image" else False,
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

        app.mount(
            "/",
            StaticFiles(directory="./inference/landing/out", html=True),
            name="static",
        )

    def run(self):
        uvicorn.run(self.app, host="127.0.0.1", port=8080)
