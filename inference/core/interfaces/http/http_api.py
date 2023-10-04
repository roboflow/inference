import traceback
from functools import partial, wraps
from typing import Any, List, Optional, Union

import uvicorn
from fastapi import Body, FastAPI, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi_cprofile.profiler import CProfileMiddleware

from inference.core import data_models as M
from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.env import (
    ALLOW_ORIGINS,
    CLIP_MODEL_ID,
    CORE_MODEL_CLIP_ENABLED,
    CORE_MODEL_GAZE_ENABLED,
    CORE_MODEL_SAM_ENABLED,
    CORE_MODELS_ENABLED,
    LAMBDA,
    LEGACY_ROUTE_ENABLED,
    METLO_KEY,
    METRICS_ENABLED,
    PROFILE,
    ROBOFLOW_SERVICE_SECRET,
)
from inference.core.exceptions import (
    ContentTypeInvalid,
    ContentTypeMissing,
    DatasetLoadError,
    EngineIgnitionFailure,
    InferenceModelNotFound,
    InvalidEnvironmentVariableError,
    MissingApiKeyError,
    MissingServiceSecretError,
    ModelArtifactsRetrievalError,
    ModelCompilationFailure,
    OnnxProviderNotAvailable,
    TensorrtRoboflowAPIError,
    WorkspaceLoadError,
)
from inference.core.interfaces.base import BaseInterface
from inference.core.managers.base import ModelManager
from inference.core.registries.base import ModelRegistry

if LAMBDA:
    from inference.core.usage import trackUsage
if METLO_KEY:
    from metlo.fastapi import ASGIMiddleware

from inference.core.registries.roboflow import RoboflowModelRegistry
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
        except ContentTypeInvalid as e:
            resp = JSONResponse(status_code=400, content={"message": str(e)})
            traceback.print_exc()
        except ContentTypeMissing as e:
            resp = JSONResponse(status_code=400, content={"message": str(e)})
            traceback.print_exc()
        except DatasetLoadError as e:
            resp = JSONResponse(status_code=500, content={"message": str(e)})
            traceback.print_exc()
        except EngineIgnitionFailure as e:
            resp = JSONResponse(status_code=500, content={"message": str(e)})
            traceback.print_exc()
        except InferenceModelNotFound as e:
            resp = JSONResponse(status_code=404, content={"message": str(e)})
            traceback.print_exc()
        except InvalidEnvironmentVariableError as e:
            resp = JSONResponse(status_code=500, content={"message": str(e)})
            traceback.print_exc()
        except MissingApiKeyError as e:
            resp = JSONResponse(status_code=400, content={"message": str(e)})
            traceback.print_exc()
        except ModelArtifactsRetrievalError as e:
            resp = JSONResponse(status_code=500, content={"message": str(e)})
            traceback.print_exc()
        except ModelCompilationFailure as e:
            resp = JSONResponse(status_code=500, content={"message": str(e)})
            traceback.print_exc()
        except OnnxProviderNotAvailable as e:
            resp = JSONResponse(status_code=501, content={"message": str(e)})
            traceback.print_exc()
        except TensorrtRoboflowAPIError as e:
            resp = JSONResponse(status_code=500, content={"message": str(e)})
            traceback.print_exc()
        except WorkspaceLoadError as e:
            resp = JSONResponse(status_code=500, content={"message": str(e)})
            traceback.print_exc()
        except Exception as e:
            resp = JSONResponse(status_code=500, content={"message": str(e)})
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
        model_registry (RoboflowModelRegistry): The registry containing the Roboflow models.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        model_registry: RoboflowModelRegistry,
        root_path: Optional[str] = None,
    ):
        """
        Initializes the HttpInterface with given model manager and model registry.

        Args:
            model_manager (ModelManager): The manager for handling different models.
            model_registry (RoboflowModelRegistry): The registry containing the Roboflow models.
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
                    model_manager.model_manager.num_errors += 1
                return response

        self.app = app
        self.model_manager = model_manager
        self.model_registry = model_registry

        def process_inference_request(
            inference_request: M.InferenceRequest,
        ) -> M.InferenceResponse:
            """Processes an inference request by calling the appropriate model.

            Args:
                inference_request (M.InferenceRequest): The request containing model ID and other inference details.

            Returns:
                M.InferenceResponse: The response containing the inference results.
            """
            if inference_request.model_id not in self.model_manager:
                model = self.model_registry.get_model(
                    inference_request.model_id, inference_request.api_key
                )(
                    model_id=inference_request.model_id,
                    api_key=inference_request.api_key,
                )
                self.model_manager.add_model(inference_request.model_id, model)
            return self.model_manager.infer_from_request(
                inference_request.model_id, inference_request
            )

        def load_core_model(
            inference_request: M.InferenceRequest,
            api_key: Optional[str] = None,
            core_model: str = None,
        ) -> None:
            """Loads a core model (e.g., "clip" or "sam") into the model manager.

            Args:
                inference_request (M.InferenceRequest): The request containing version and other details.
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
            if core_model_id not in self.model_manager:
                model = self.model_registry.get_model(
                    core_model_id, inference_request.api_key
                )(
                    model_id=core_model_id,
                    api_key=inference_request.api_key,
                )
                self.model_manager.add_model(core_model_id, model)
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

        @app.get(
            "/",
            response_model=M.ServerVersionInfo,
            summary="Root",
            description="Get the server name and version number",
        )
        async def root():
            """Endpoint to get the server name and version number.

            Returns:
                M.ServerVersionInfo: The server version information.
            """
            return M.ServerVersionInfo(
                name="Roboflow Inference Server",
                version=__version__,
                uuid=GLOBAL_INFERENCE_SERVER_ID,
            )

        @app.get(
            "/info",
            response_model=M.ServerVersionInfo,
            summary="Info",
            description="Get the server name and version number",
        )
        async def root():
            """Endpoint to get the server name and version number.

            Returns:
                M.ServerVersionInfo: The server version information.
            """
            return M.ServerVersionInfo(
                name="Roboflow Inference Server",
                version=__version__,
                uuid=GLOBAL_INFERENCE_SERVER_ID,
            )

        # The current AWS Lambda authorizer only supports path parameters, therefore we can only use the legacy infer route. This case statement excludes routes which won't work for the current Lambda authorizer.
        if not LAMBDA:

            @app.get(
                "/model/registry",
                response_model=M.ModelsDescriptions,
                summary="Get model keys",
                description="Get the ID of each loaded model",
            )
            async def registry():
                """Get the ID of each loaded model in the registry.

                Returns:
                    M.ModelsDescriptions: The object containing models descriptions
                """
                models_descriptions = self.model_manager.describe_models()
                return M.ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

            @app.post(
                "/model/add",
                response_model=M.ModelsDescriptions,
                summary="Load a model",
                description="Load the model with the given model ID",
            )
            @with_route_exceptions
            async def model_add(request: M.AddModelRequest):
                """Load the model with the given model ID into the model manager.

                Args:
                    request (M.AddModelRequest): The request containing the model ID and optional API key.

                Returns:
                    M.ModelsDescriptions: The object containing models descriptions
                """

                if request.model_id not in self.model_manager:
                    model_class = self.model_registry.get_model(
                        request.model_id, request.api_key
                    )
                    model = model_class(
                        model_id=request.model_id, api_key=request.api_key
                    )
                    self.model_manager.add_model(request.model_id, model)
                models_descriptions = self.model_manager.describe_models()
                return M.ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

            @app.post(
                "/model/remove",
                response_model=M.ModelsDescriptions,
                summary="Remove a model",
                description="Remove the model with the given model ID",
            )
            @with_route_exceptions
            async def model_remove(request: M.ClearModelRequest):
                """Remove the model with the given model ID from the model manager.

                Args:
                    request (M.ClearModelRequest): The request containing the model ID to be removed.

                Returns:
                    M.ModelsDescriptions: The object containing models descriptions
                """

                self.model_manager.remove(request.model_id)
                models_descriptions = self.model_manager.describe_models()
                return M.ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

            @app.post(
                "/model/clear",
                response_model=M.ModelsDescriptions,
                summary="Remove all models",
                description="Remove all loaded models",
            )
            @with_route_exceptions
            async def model_clear():
                """Remove all loaded models from the model manager.

                Returns:
                    M.ModelsDescriptions: The object containing models descriptions
                """

                self.model_manager.clear()
                models_descriptions = self.model_manager.describe_models()
                return M.ModelsDescriptions.from_models_descriptions(
                    models_descriptions=models_descriptions
                )

            @app.post(
                "/infer/object_detection",
                response_model=Union[
                    M.ObjectDetectionInferenceResponse,
                    List[M.ObjectDetectionInferenceResponse],
                ],
                summary="Object detection infer",
                description="Run inference with the specified object detection model",
                response_model_exclude_none=True,
            )
            @with_route_exceptions
            async def infer_object_detection(
                inference_request: M.ObjectDetectionInferenceRequest,
            ):
                """Run inference with the specified object detection model.

                Args:
                    inference_request (M.ObjectDetectionInferenceRequest): The request containing the necessary details for object detection.

                Returns:
                    Union[M.ObjectDetectionInferenceResponse, List[M.ObjectDetectionInferenceResponse]]: The response containing the inference results.
                """

                return process_inference_request(inference_request)

            @app.post(
                "/infer/instance_segmentation",
                response_model=M.InstanceSegmentationInferenceResponse,
                summary="Instance segmentation infer",
                description="Run inference with the specified instance segmentation model",
            )
            @with_route_exceptions
            async def infer_instance_segmentation(
                inference_request: M.InstanceSegmentationInferenceRequest,
            ):
                """Run inference with the specified instance segmentation model.

                Args:
                    inference_request (M.InstanceSegmentationInferenceRequest): The request containing the necessary details for instance segmentation.

                Returns:
                    M.InstanceSegmentationInferenceResponse: The response containing the inference results.
                """

                return process_inference_request(inference_request)

            @app.post(
                "/infer/classification",
                response_model=Union[
                    M.ClassificationInferenceResponse,
                    M.MultiLabelClassificationInferenceResponse,
                ],
                summary="Classification infer",
                description="Run inference with the specified classification model",
            )
            @with_route_exceptions
            async def infer_classification(
                inference_request: M.ClassificationInferenceRequest,
            ):
                """Run inference with the specified classification model.

                Args:
                    inference_request (M.ClassificationInferenceRequest): The request containing the necessary details for classification.

                Returns:
                    Union[M.ClassificationInferenceResponse, M.MultiLabelClassificationInferenceResponse]: The response containing the inference results.
                """

                return process_inference_request(inference_request)

        if CORE_MODELS_ENABLED:
            if CORE_MODEL_CLIP_ENABLED:

                @app.post(
                    "/clip/embed_image",
                    response_model=M.ClipEmbeddingResponse,
                    summary="CLIP Image Embeddings",
                    description="Run the Open AI CLIP model to embed image data.",
                )
                @with_route_exceptions
                async def clip_embed_image(
                    inference_request: M.ClipImageEmbeddingRequest,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    request: Request = Body(),
                ):
                    """
                    Embeds image data using the OpenAI CLIP model.

                    Args:
                        inference_request (M.ClipImageEmbeddingRequest): The request containing the image to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.ClipEmbeddingResponse: The response containing the embedded image.
                    """
                    clip_model_id = load_clip_model(inference_request, api_key=api_key)
                    response = self.model_manager.infer_from_request(
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
                    response_model=M.ClipEmbeddingResponse,
                    summary="CLIP Text Embeddings",
                    description="Run the Open AI CLIP model to embed text data.",
                )
                @with_route_exceptions
                async def clip_embed_text(
                    inference_request: M.ClipTextEmbeddingRequest,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    request: Request = Body(),
                ):
                    """
                    Embeds text data using the OpenAI CLIP model.

                    Args:
                        inference_request (M.ClipTextEmbeddingRequest): The request containing the text to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.ClipEmbeddingResponse: The response containing the embedded text.
                    """
                    clip_model_id = load_clip_model(inference_request, api_key=api_key)
                    response = self.model_manager.infer_from_request(
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
                    response_model=M.ClipCompareResponse,
                    summary="CLIP Compare",
                    description="Run the Open AI CLIP model to compute similarity scores.",
                )
                @with_route_exceptions
                async def clip_compare(
                    inference_request: M.ClipCompareRequest,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    request: Request = Body(),
                ):
                    """
                    Computes similarity scores using the OpenAI CLIP model.

                    Args:
                        inference_request (M.ClipCompareRequest): The request containing the data to be compared.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.ClipCompareResponse: The response containing the similarity scores.
                    """
                    clip_model_id = load_clip_model(inference_request, api_key=api_key)
                    response = self.model_manager.infer_from_request(
                        clip_model_id, inference_request
                    )
                    if LAMBDA:
                        actor = request.scope["aws.event"]["requestContext"][
                            "authorizer"
                        ]["lambda"]["actor"]
                        trackUsage(clip_model_id, actor, n=2)
                    return response

            if CORE_MODEL_SAM_ENABLED:

                @app.post(
                    "/sam/embed_image",
                    response_model=M.SamEmbeddingResponse,
                    summary="SAM Image Embeddings",
                    description="Run the Meta AI Segmant Anything Model to embed image data.",
                )
                @with_route_exceptions
                async def sam_embed_image(
                    inference_request: M.SamEmbeddingRequest,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    request: Request = Body(),
                ):
                    """
                    Embeds image data using the Meta AI Segmant Anything Model (SAM).

                    Args:
                        inference_request (M.SamEmbeddingRequest): The request containing the image to be embedded.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.SamEmbeddingResponse or Response: The response containing the embedded image.
                    """
                    sam_model_id = load_sam_model(inference_request, api_key=api_key)
                    model_response = self.model_manager.infer_from_request(
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
                    response_model=M.SamSegmentationResponse,
                    summary="SAM Image Segmentation",
                    description="Run the Meta AI Segmant Anything Model to generate segmenations for image data.",
                )
                @with_route_exceptions
                async def sam_segment_image(
                    inference_request: M.SamSegmentationRequest,
                    api_key: Optional[str] = Query(
                        None,
                        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                    ),
                    request: Request = Body(),
                ):
                    """
                    Generates segmentations for image data using the Meta AI Segmant Anything Model (SAM).

                    Args:
                        inference_request (M.SamSegmentationRequest): The request containing the image to be segmented.
                        api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                        request (Request, default Body()): The HTTP request.

                    Returns:
                        M.SamSegmentationResponse or Response: The response containing the segmented image.
                    """
                    sam_model_id = load_sam_model(inference_request, api_key=api_key)
                    model_response = self.model_manager.infer_from_request(
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
                        response_model=List[M.GazeDetectionInferenceResponse],
                        summary="Gaze Detection",
                        description="Run the gaze detection model to detect gaze.",
                    )
                    @with_route_exceptions
                    async def gaze_detection(
                        inference_request: M.GazeDetectionInferenceRequest,
                        api_key: Optional[str] = Query(
                            None,
                            description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
                        ),
                        request: Request = Body(),
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
                        gaze_model_id = load_gaze_model(
                            inference_request, api_key=api_key
                        )
                        response = self.model_manager.infer_from_request(
                            gaze_model_id, inference_request
                        )
                        if LAMBDA:
                            actor = request.scope["aws.event"]["requestContext"][
                                "authorizer"
                            ]["lambda"]["actor"]
                            trackUsage(gaze_model_id, actor)
                        return response

        if LEGACY_ROUTE_ENABLED:
            # Legacy object detection inference path for backwards compatability
            @app.post(
                "/{dataset_id}/{version_id}",
                # Order matters in this response model Union. It will use the first matching model. For example, Object Detection Inference Response is a subset of Instance segmentation inference response, so instance segmentation must come first in order for the matching logic to work.
                response_model=Union[
                    M.InstanceSegmentationInferenceResponse,
                    M.ObjectDetectionInferenceResponse,
                    M.ClassificationInferenceResponse,
                    M.MultiLabelClassificationInferenceResponse,
                    Any,
                ],
                response_model_exclude_none=True,
            )
            @with_route_exceptions
            async def legacy_infer_from_request(
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
                request: Request = Body(),
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
            ):
                """
                Legacy inference endpoint for object detection, instance segmentation, and classification.

                Args:
                    dataset_id (str): ID of a Roboflow dataset corresponding to the model to use for inference.
                    version_id (str): ID of a Roboflow dataset version corresponding to the model to use for inference.
                    api_key (Optional[str], default None): Roboflow API Key passed to the model during initialization for artifact retrieval.
                    # Other parameters described in the function signature...

                Returns:
                    Union[M.InstanceSegmentationInferenceResponse, M.ObjectDetectionInferenceResponse, M.ClassificationInferenceResponse, M.MultiLabelClassificationInferenceResponse, Any]: The response containing the inference results.
                """
                model_id = f"{dataset_id}/{version_id}"

                if confidence >= 1:
                    confidence /= 100
                elif confidence < 0.01:
                    confidence = 0.01

                if overlap >= 1:
                    overlap /= 100

                if image is not None:
                    request_image = M.InferenceRequestImage(type="url", value=image)
                else:
                    if "Content-Type" not in request.headers:
                        raise ContentTypeMissing(
                            f"Request must include a Content-Type header"
                        )
                    if "multipart/form-data" in request.headers["Content-Type"]:
                        form_data = await request.form()
                        base64_image_str = form_data["file"].file
                        request_image = M.InferenceRequestImage(
                            type="multipart", value=base64_image_str
                        )
                    elif (
                        "application/x-www-form-urlencoded"
                        in request.headers["Content-Type"]
                        or "application/json" in request.headers["Content-Type"]
                    ):
                        data = await request.body()
                        request_image = M.InferenceRequestImage(
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

                if request_model_id not in self.model_manager:
                    model = self.model_registry.get_model(model_id, api_key)(
                        model_id=request_model_id, api_key=api_key
                    )
                    self.model_manager.add_model(request_model_id, model)

                task_type = self.model_manager.get_task_type(request_model_id)
                inference_request_type = M.ObjectDetectionInferenceRequest
                args = dict()
                if task_type == "instance-segmentation":
                    inference_request_type = M.InstanceSegmentationInferenceRequest
                    args = {
                        "mask_decode_mode": mask_decode_mode,
                        "tradeoff_factor": tradeoff_factor,
                    }
                elif task_type == "classification":
                    inference_request_type = M.ClassificationInferenceRequest

                inference_request = inference_request_type(
                    model_id=request_model_id,
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
                    **args,
                )

                inference_response = self.model_manager.infer_from_request(
                    inference_request.model_id, inference_request
                )

                if format == "image":
                    return Response(
                        content=inference_response.visualization,
                        media_type="image/jpeg",
                    )
                else:
                    return inference_response

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
                model_id = f"{dataset_id}/{version_id}"
                model = self.model_registry.get_model(model_id, api_key)(
                    model_id=model_id, api_key=api_key
                )
                self.model_manager.add_model(model_id, model)

                return JSONResponse(
                    {
                        "status": 200,
                        "message": "inference session started from local memory.",
                    }
                )

    def run(self):
        uvicorn.run(self.app, host="127.0.0.1", port=8080)
