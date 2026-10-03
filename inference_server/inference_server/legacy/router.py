import asyncio
import base64
import logging
import time
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
)
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from inference_sdk.http.utils.aliases import resolve_roboflow_model_alias
from inference_server import configuration, platform_http, server_identity, telemetry
from inference_server.dependencies import get_model_manager
from inference_server.legacy.action_recognition import (
    ACTION_RECOGNITION_TASK,
    classify_video,
    ensure_action_recognition_route,
)
from inference_server.legacy.active_learning_registration import register_inference
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    Route,
    request_alias_for,
    requested_model_id_for,
    resolved_model_for,
)
from inference_server.legacy.common import (
    as_image_list,
    image_load_error,
    load_request_images,
    orjson_response,
    resolve_api_key,
)
from inference_server.legacy.cuda_health import check_cuda_health
from inference_server.legacy.entities import (
    ActionRecognitionInferenceRequest,
    ActionRecognitionInferenceResponse,
    AddModelRequest,
    AnomalyDetectionResponse,
    ClassificationInferenceRequest,
    ClassificationInferenceResponse,
    ClearModelRequest,
    ClipCompareRequest,
    ClipCompareResponse,
    ClipEmbeddingResponse,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
    Confidence,
    DepthEstimationRequest,
    DepthEstimationResponse,
    DoctrOCRInferenceRequest,
    EasyOCRInferenceRequest,
    GroundingDINOInferenceRequest,
    InferenceRequestImage,
    InferenceRequestVideo,
    InstanceSegmentationInferenceRequest,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceRequest,
    KeypointsDetectionInferenceResponse,
    LMMInferenceRequest,
    LMMInferenceResponse,
    ModelDescriptionEntity,
    ModelsDescriptions,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceRequest,
    ObjectDetectionInferenceResponse,
    OCRInferenceResponse,
    OwlV2InferenceRequest,
    PerceptionEncoderCompareRequest,
    PerceptionEncoderCompareResponse,
    PerceptionEncoderEmbeddingResponse,
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
    PPOCRInferenceRequest,
    Sam2EmbeddingRequest,
    Sam2EmbeddingResponse,
    Sam2SegmentationRequest,
    Sam2SegmentationResponse,
    Sam3EmbeddingResponse,
    Sam3SegmentationRequest,
    Sam3SegmentationResponse,
    SamEmbeddingRequest,
    SamEmbeddingResponse,
    SamSegmentationRequest,
    SamSegmentationResponse,
    SemanticSegmentationInferenceRequest,
    SemanticSegmentationInferenceResponse,
    ServerVersionInfo,
    StubResponse,
    TrOCRInferenceRequest,
    YOLOWorldInferenceRequest,
)
from inference_server.legacy.errors import (
    LegacyHTTPError,
    MissingServiceSecretError,
    redact_text,
    with_legacy_errors,
)
from inference_server.legacy.telemetry_recording import record_telemetry
from inference_server.legacy.translation import (
    build_embedding_calls,
    build_few_shot_params,
    build_interactive_segmentation_params,
    build_open_vocabulary_params,
    build_task_params,
    build_vlm_params,
    encode_normalized_depth_to_png8,
    encode_normalized_depth_to_png16,
    ensure_ocr_request_supported,
    ensure_request_supported,
    few_shot_class_names,
    is_metric_depth_model_class,
    repack_depth_estimation,
    repack_embedding_response,
    repack_interactive_segmentation_response,
    repack_moondream_detection,
    repack_object_detection_response,
    repack_prediction,
    repack_structured_ocr_response,
    repack_text_ocr_response,
    repack_vlm_response,
    requested_open_vocabulary_classes,
    resolve_request_action,
)
from inference_server.legacy.visualization import (
    encode_image_to_jpeg_bytes,
    render_visualization,
)
from inference_server import pingback
from inference_server.hosted.common import service_secret_is_valid
from inference_server.prometheus import measure_inference
from inference_server.usage.request_hook import report_request_usage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["legacy"])
infer_router = APIRouter(tags=["legacy"])
control_plane_router = APIRouter(tags=["legacy"])
registry_router = APIRouter(tags=["legacy"])
catch_all_router = APIRouter(tags=["legacy"])
clip_router = APIRouter(tags=["legacy"])
perception_encoder_router = APIRouter(tags=["legacy"])
doctr_router = APIRouter(tags=["legacy"])
easy_ocr_router = APIRouter(tags=["legacy"])
trocr_router = APIRouter(tags=["legacy"])
pp_ocr_router = APIRouter(tags=["legacy"])
yolo_world_router = APIRouter(tags=["legacy"])
grounding_dino_router = APIRouter(tags=["legacy"])
owlv2_router = APIRouter(tags=["legacy"])
gaze_router = APIRouter(tags=["legacy"])
lmm_router = APIRouter(tags=["legacy"])
depth_router = APIRouter(tags=["legacy"])
sam_router = APIRouter(tags=["legacy"])
sam2_router = APIRouter(tags=["legacy"])
sam3_router = APIRouter(tags=["legacy"])
sam3_3d_router = APIRouter(tags=["legacy"])
action_recognition_router = APIRouter(tags=["legacy"])

_CORE_MODEL_ROUTER_GROUPS = (
    (("CORE_MODEL_CLIP_ENABLED",), clip_router),
    (("CORE_MODEL_PE_ENABLED",), perception_encoder_router),
    (("CORE_MODEL_DOCTR_ENABLED",), doctr_router),
    (("CORE_MODEL_EASYOCR_ENABLED",), easy_ocr_router),
    (("CORE_MODEL_TROCR_ENABLED",), trocr_router),
    (("CORE_MODEL_PPOCR_ENABLED",), pp_ocr_router),
    (("CORE_MODEL_YOLO_WORLD_ENABLED",), yolo_world_router),
    (("CORE_MODEL_GROUNDINGDINO_ENABLED",), grounding_dino_router),
    (("CORE_MODEL_OWLV2_ENABLED",), owlv2_router),
    (("CORE_MODEL_GAZE_ENABLED",), gaze_router),
    (("DEPTH_ESTIMATION_ENABLED",), depth_router),
    (("CORE_MODEL_SAM_ENABLED",), sam_router),
    (("CORE_MODEL_SAM2_ENABLED",), sam2_router),
    (("CORE_MODEL_SAM3_ENABLED",), sam3_router),
    (("SAM3_3D_OBJECTS_ENABLED",), sam3_3d_router),
    (("ACTION_RECOGNITION_ENABLED",), action_recognition_router),
)

_VISUALIZATION_FORMATS = ("image", "image_and_json")
_CONTENT_TYPE_MISSING_MESSAGE = "Content-Type header not provided with request."
_CONTENT_TYPE_INVALID_MESSAGE = "Invalid Content-Type header provided with request."
_MULTIPART_PART_MISSING_MESSAGE = (
    "Expected image to be send in part named 'file' of multipart/form-data request"
)
_YOLO_WORLD_UNSUPPORTED_MESSAGE = (
    "YOLO-World is not supported by this inference server configuration."
)

_TASK_UNAVAILABLE_MESSAGE = (
    "{route} is not available on inference_server: no model class for this task is "
    "registered with the model manager"
)
SAM3_CONCEPT_REMOTE_FAILURE_MESSAGE = "SAM3 remote request failed."
SAM3_VISUAL_REMOTE_FAILURE_MESSAGE = "SAM3 visual_segment remote request failed."
SAM3_EMBEDDING_REMOTE_UNSUPPORTED_MESSAGE = (
    "SAM3 embedding is not supported in remote execution mode."
)
_DEPTH_SINGLE_IMAGE_MESSAGE = "Depth estimation accepts a single image."
FINE_TUNED_SAM3_DEPLOYMENT_ERROR = (
    "Fine-tuned SAM 3 models are not supported on Serverless. "
    "Use the base SAM 3 model (sam3/sam3_final), a Dedicated Deployment, "
    "or self-hosted Inference."
)
SAM3_INTERACTIVE_MODEL_ID = "sam3/sam3_interactive"
_GAZE_DEPRECATION_BODY = {
    "message": (
        "Feature '/gaze/gaze_detection' has been removed from inference. Reason: "
        "MediaPipe dependency removed from inference; endpoint is a 410 stub.. "
        "Removed in end of Q2 2026. No drop-in replacement is provided; contact "
        "Roboflow if you require this capability."
    ),
    "error_type": "FeatureDeprecatedError",
    "feature": "/gaze/gaze_detection",
    "removal_release": "end of Q2 2026",
    "replacement": None,
    "reason": "MediaPipe dependency removed from inference; endpoint is a 410 stub.",
}


def get_bridge(request: Request) -> LegacyModelBridge:
    return request.app.state.legacy_bridge


def include_legacy_routers(app: FastAPI) -> None:
    hosted = configuration.LAMBDA or configuration.GCP_SERVERLESS
    app.include_router(router)
    if not hosted:
        app.include_router(infer_router)
    if not configuration.LAMBDA and (
        configuration.LMM_ENABLED or configuration.MOONDREAM2_ENABLED
    ):
        app.include_router(lmm_router)
    if configuration.CORE_MODELS_ENABLED:
        for flag_names, group_router in _CORE_MODEL_ROUTER_GROUPS:
            if any(getattr(configuration, name) for name in flag_names):
                app.include_router(group_router)
    if configuration.LEGACY_CONTROL_PLANE_ROUTES_ENABLED:
        if not hosted:
            app.include_router(control_plane_router)
        if not configuration.LAMBDA and configuration.GET_MODEL_REGISTRY_ENABLED:
            app.include_router(registry_router)


def include_legacy_catch_all(app: FastAPI) -> None:
    if configuration.LEGACY_CATCH_ALL_ROUTE_ENABLED:
        app.include_router(catch_all_router)


@router.get(
    "/info",
    response_model=ServerVersionInfo,
    summary="Info",
    description="Get the server name and version number",
)
async def info() -> ServerVersionInfo:
    server_id = await asyncio.to_thread(server_identity.get_inference_server_id)

    return ServerVersionInfo(
        name="Roboflow Inference Server",
        version=configuration.SERVER_VERSION,
        uuid=server_id,
    )


@router.get("/healthz", status_code=200)
async def healthz() -> Response:
    is_healthy, error = check_cuda_health()
    if is_healthy:
        return JSONResponse(content={"status": "healthy"})
    logger.error("CUDA health check failed: %s", error)
    return JSONResponse(
        content={"status": "unhealthy", "reason": "cuda_error"}, status_code=503
    )


@router.get("/readiness", status_code=200)
async def readiness(
    request: Request, model_manager: Any = Depends(get_model_manager)
) -> Response:
    try:
        await model_manager.stats()
    except Exception:
        return JSONResponse(content={"status": "not ready"}, status_code=503)
    if not request.app.state.preload_finished:
        return JSONResponse(content={"status": "not ready"}, status_code=503)
    return JSONResponse(content={"status": "ready"})


async def _models_descriptions(bridge: LegacyModelBridge) -> ModelsDescriptions:
    routes = await bridge.describe()
    descriptions = []
    for route in routes:
        if not route.requested_at:
            descriptions.append(
                _model_description(
                    route,
                    model_id=route.registry_id,
                    request_aliases=[],
                    request_paths=[],
                )
            )
            continue
        for model_id in sorted(route.requested_at):
            descriptions.append(
                _model_description(
                    route,
                    model_id=model_id,
                    request_aliases=sorted(
                        route.request_aliases_by_id.get(model_id, ())
                    ),
                    request_paths=sorted(route.request_paths_by_id.get(model_id, {})),
                )
            )

    models_descriptions = ModelsDescriptions.from_models_descriptions(
        descriptions,
        model_vram_bytes=[route.vram_bytes for route in routes],
    )

    return models_descriptions


def _model_description(
    route: Route,
    *,
    model_id: str,
    request_aliases: list[str],
    request_paths: list[str],
) -> ModelDescriptionEntity:
    return ModelDescriptionEntity(
        model_id=model_id,
        task_type=route.task_type,
        input_height=route.input_height,
        input_width=route.input_width,
        vram_bytes=route.vram_bytes,
        request_aliases=request_aliases,
        request_paths=request_paths,
    )


@registry_router.get(
    "/model/registry",
    response_model=ModelsDescriptions,
    summary="Get model keys",
    description="Get the ID of each loaded model",
)
@with_legacy_errors
async def registry(bridge: LegacyModelBridge = Depends(get_bridge)):
    return await _models_descriptions(bridge)


@control_plane_router.post(
    "/model/add",
    response_model=ModelsDescriptions,
    summary="Load a model",
    description="Load the model with the given model ID",
)
@with_legacy_errors
async def model_add(
    request: Request,
    add_model_request: AddModelRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
):
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), add_model_request.api_key
    )
    row_key = resolve_roboflow_model_alias(add_model_request.model_id)
    route = await bridge.load(
        add_model_request.model_id,
        api_key,
        row_key=row_key,
        path=request.scope["path"],
    )
    bridge.record_request(route, row_key, request.scope["path"])
    return await _models_descriptions(bridge)


@control_plane_router.post(
    "/model/remove",
    response_model=ModelsDescriptions,
    summary="Remove a model",
    description="Remove the model with the given model ID",
)
@with_legacy_errors
async def model_remove(
    clear_model_request: ClearModelRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
):
    await bridge.remove(clear_model_request.model_id)
    return await _models_descriptions(bridge)


@control_plane_router.post(
    "/model/clear",
    response_model=ModelsDescriptions,
    summary="Remove all models",
    description="Remove all loaded models",
)
@with_legacy_errors
async def model_clear(bridge: LegacyModelBridge = Depends(get_bridge)):
    await bridge.unload_all()
    return await _models_descriptions(bridge)


@control_plane_router.get("/clear_cache", response_model=str)
@with_legacy_errors
async def clear_cache(bridge: LegacyModelBridge = Depends(get_bridge)):
    await bridge.unload_all()
    return "Cache Cleared"


@control_plane_router.get("/start/{dataset_id}/{version_id}")
@with_legacy_errors
async def model_add_legacy(
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
    countinference: Optional[bool] = Query(True, include_in_schema=False),
    service_secret: Optional[str] = Query(None, include_in_schema=False),
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    model_id = f"{dataset_id}/{version_id}"
    resolved_key = resolve_api_key(request, api_key, None)
    route = await bridge.load(
        model_id, resolved_key, row_key=model_id, path=request.scope["path"]
    )
    bridge.record_request(route, model_id, request.scope["path"])
    return JSONResponse(
        {"status": 200, "message": "inference session started from local memory."}
    )


async def _run_cv_inference(
    request: Request,
    inference_request,
    bridge: LegacyModelBridge,
    *,
    expected_task_types: Tuple[str, ...],
    active_learning_eligible: bool = False,
) -> Response:
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), inference_request.api_key
    )
    inference_request.api_key = api_key
    alias = request_alias_for(inference_request.model_id)
    route = await bridge.resolve(
        inference_request.model_id,
        api_key,
        row_key=inference_request.model_id,
        path=request.scope["path"],
        alias=alias,
    )
    bridge.record_request(
        route,
        inference_request.model_id,
        request.scope["path"],
        alias=alias,
    )
    if route.is_stub:
        return orjson_response(_stub_response(inference_request, route))

    if route.task_type not in expected_task_types:
        raise LegacyHTTPError(
            400,
            f"Model {inference_request.model_id!r} is a {route.task_type} model.",
        )
    return await _infer_and_repack(
        inference_request,
        bridge,
        route,
        api_key,
        active_learning_eligible=active_learning_eligible,
    )


def _model_monitoring_enabled(inference_request) -> bool:
    return not getattr(inference_request, "disable_model_monitoring", False)


def _stub_response(inference_request, route: Route) -> StubResponse:
    started = time.perf_counter()
    visualization = None
    if getattr(inference_request, "visualize_predictions", False):
        visualization = encode_image_to_jpeg_bytes(
            np.zeros((128, 128, 3), dtype=np.uint8)
        )
    response = StubResponse(
        is_stub=True,
        model_id=resolve_roboflow_model_alias(route.model_id),
        task_type=route.task_type,
        visualization=visualization,
    )
    response.time = time.perf_counter() - started

    return response


async def _infer_and_repack(
    inference_request,
    bridge: LegacyModelBridge,
    route: Route,
    api_key: Optional[str],
    image_format: Optional[str] = None,
    active_learning_eligible: bool = False,
) -> Response:
    ensure_request_supported(inference_request.model_id, inference_request, route)
    visualize = image_format in _VISUALIZATION_FORMATS or bool(
        getattr(inference_request, "visualize_predictions", False)
    )
    images, is_batch = as_image_list(inference_request.image)
    payloads = await load_request_images(images, ndarray_ok=bridge.accepts_ndarray)
    params = build_task_params(route.task_type, route.action, inference_request, route)
    started = time.perf_counter()
    predictions = await bridge.infer(
        route,
        api_key,
        route.action,
        payloads,
        params,
        model_monitoring=_model_monitoring_enabled(inference_request),
    )
    elapsed = time.perf_counter() - started
    responses = []
    for prediction, payload in zip(predictions, payloads):
        response = repack_prediction(
            route.task_type,
            route.action,
            prediction,
            (payload.width, payload.height),
            route,
            inference_request,
        )
        response.time = elapsed
        response.inference_id = inference_request.id
        response.resolved_model = resolved_model_for(route)
        if visualize:
            response.visualization = render_visualization(
                route, inference_request, response, payload
            )
        responses.append(response)
    pingback.record_inference(route.registry_id, inference_request, responses)
    if image_format == "image":
        http_response = Response(
            content=responses[0].visualization if responses else None,
            media_type="image/jpeg",
        )
    else:
        http_response = orjson_response(responses if is_batch else responses[0])
    await register_inference(
        inference_request,
        task_type=route.task_type,
        payloads=payloads,
        responses=responses,
        eligible=active_learning_eligible,
    )
    return http_response


@infer_router.post(
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
@with_legacy_errors
@report_request_usage
async def infer_object_detection(
    request: Request,
    inference_request: ObjectDetectionInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_cv_inference(
        request,
        inference_request,
        bridge,
        expected_task_types=("object-detection",),
        active_learning_eligible=True,
    )


@infer_router.post(
    "/infer/instance_segmentation",
    response_model=Union[
        InstanceSegmentationInferenceResponse,
        List[InstanceSegmentationInferenceResponse],
        StubResponse,
    ],
    summary="Instance segmentation infer",
    description="Run inference with the specified instance segmentation model",
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def infer_instance_segmentation(
    request: Request,
    inference_request: InstanceSegmentationInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_cv_inference(
        request,
        inference_request,
        bridge,
        expected_task_types=("instance-segmentation",),
        active_learning_eligible=True,
    )


@infer_router.post(
    "/infer/semantic_segmentation",
    response_model=Union[
        SemanticSegmentationInferenceResponse,
        List[SemanticSegmentationInferenceResponse],
        StubResponse,
    ],
    summary="Semantic segmentation infer",
    description="Run inference with the specified semantic segmentation model",
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def infer_semantic_segmentation(
    request: Request,
    inference_request: SemanticSegmentationInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    inference_request.response_mask_format = "base64_png"
    return await _run_cv_inference(
        request,
        inference_request,
        bridge,
        expected_task_types=("semantic-segmentation",),
        active_learning_eligible=True,
    )


@infer_router.post(
    "/infer/classification",
    response_model=Union[
        ClassificationInferenceResponse,
        List[ClassificationInferenceResponse],
        MultiLabelClassificationInferenceResponse,
        List[MultiLabelClassificationInferenceResponse],
        AnomalyDetectionResponse,
        List[AnomalyDetectionResponse],
        StubResponse,
    ],
    summary="Classification infer",
    description="Run inference with the specified classification model",
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def infer_classification(
    request: Request,
    inference_request: ClassificationInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_cv_inference(
        request,
        inference_request,
        bridge,
        expected_task_types=("classification", "multi-label-classification"),
        active_learning_eligible=True,
    )


@infer_router.post(
    "/infer/keypoints_detection",
    response_model=Union[
        KeypointsDetectionInferenceResponse,
        List[KeypointsDetectionInferenceResponse],
        StubResponse,
    ],
    summary="Keypoints detection infer",
    description="Run inference with the specified keypoints detection model",
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def infer_keypoints(
    request: Request,
    inference_request: KeypointsDetectionInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_cv_inference(
        request,
        inference_request,
        bridge,
        expected_task_types=("keypoint-detection",),
    )


@action_recognition_router.post(
    "/infer/action_recognition",
    response_model=Union[ActionRecognitionInferenceResponse, StubResponse],
    summary="Action Recognition",
    description=(
        "Classify the actions in a video clip. The model states how the clip is "
        "cut and how its frames are sampled, so a caller sends the clip and "
        "nothing else. Frame indices in the response count from the first frame "
        "of the clip, and windows_classified reports how many calls the clip was "
        "cut into. A fine-tuned model reports its own classes. A zero-shot model "
        "names the events it finds in its own words. Frames are chosen by the "
        "clip's nominal frame rate, so a variable-frame-rate source is sampled at "
        "different instants than the model trained on. Send the clip as a URL. "
        "Base64 grows it by a third and holds the whole request in memory, so it "
        "suits short clips only."
    ),
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def infer_action_recognition(
    request: Request,
    inference_request: ActionRecognitionInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), inference_request.api_key
    )
    inference_request.api_key = api_key
    alias = request_alias_for(inference_request.model_id)
    route = await bridge.resolve(
        inference_request.model_id,
        api_key,
        row_key=inference_request.model_id,
        path=request.scope["path"],
        alias=alias,
    )
    bridge.record_request(
        route,
        inference_request.model_id,
        request.scope["path"],
        alias=alias,
    )
    if route.is_stub:
        return orjson_response(_stub_response(inference_request, route))

    ensure_action_recognition_route(inference_request.model_id, route)
    http_response = await _classify_and_repack(inference_request, bridge, route)

    return http_response


async def _classify_and_repack(
    inference_request: ActionRecognitionInferenceRequest,
    bridge: LegacyModelBridge,
    route: Route,
) -> Response:
    started = time.perf_counter()
    response = await classify_video(
        route,
        inference_request.api_key,
        bridge,
        video_type=inference_request.video.type,
        video_value=inference_request.video.value,
        class_filter=inference_request.class_filter or None,
    )
    response.time = time.perf_counter() - started
    response.inference_id = inference_request.id
    response.resolved_model = resolved_model_for(route)
    pingback.record_inference(route.registry_id, inference_request, response)
    http_response = orjson_response(response)

    return http_response


def _parse_legacy_class_filter(class_filter: Optional[str]) -> Optional[List[str]]:
    if not class_filter:
        return None
    classes = [entry.strip() for entry in class_filter.split(",") if entry.strip()]
    return classes or None


@sam3_3d_router.post("/sam3_3d/infer")
@with_legacy_errors
@report_request_usage
async def infer_sam3_3d(request: Request) -> Response:
    raise LegacyHTTPError(501, _TASK_UNAVAILABLE_MESSAGE.format(route="/sam3_3d/infer"))


@owlv2_router.post(
    "/owlv2/infer",
    response_model=Union[
        ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]
    ],
    summary="Owlv2 image prompting",
    description="Run the google owlv2 model to few-shot object detect",
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def infer_owlv2(
    request: Request,
    inference_request: OwlV2InferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_few_shot_detection(request, inference_request, bridge, "owlv2")


async def _catch_all_image(
    request: Request, image: Optional[str], image_type: Optional[str]
) -> InferenceRequestImage:
    content_type = request.headers.get("Content-Type")
    part = None
    if content_type is not None and "multipart/form-data" in content_type:
        form = await request.form()
        if "file" not in form:
            raise image_load_error(_MULTIPART_PART_MISSING_MESSAGE)
        part = form["file"]
    if image is not None:
        return InferenceRequestImage(type="url", value=image)
    if content_type is None:
        raise LegacyHTTPError(400, _CONTENT_TYPE_MISSING_MESSAGE)
    if isinstance(part, UploadFile):
        data = await part.read()
        return InferenceRequestImage(
            type="base64", value=base64.b64encode(data).decode("ascii")
        )
    if part is not None:
        raise LegacyHTTPError(400, _CONTENT_TYPE_INVALID_MESSAGE)

    body = await request.body()
    return InferenceRequestImage(type=image_type, value=body)


@catch_all_router.get("/{dataset_id}/{version_id}")
@catch_all_router.post("/{dataset_id}/{version_id}")
@with_legacy_errors
@report_request_usage
async def legacy_infer_from_request(
    request: Request,
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
    confidence: Confidence = Query(
        configuration.DEFAULT_CONFIDENCE,
        description=(
            "The confidence threshold used to filter out predictions. "
            'Pass a float in [0, 1], or "best" to use F1-optimal thresholds from '
            'model evaluation, or "default" to use the model\'s built-in default.'
        ),
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
    class_filter: Optional[str] = Query(
        None,
        description=(
            "Action recognition only: comma separated classes. The subset of a "
            "fine-tuned model's classes to report."
        ),
    ),
    labels: Optional[bool] = Query(
        False,
        description="If true, labels will be include in any inference visualization.",
    ),
    mask_decode_mode: Optional[str] = Query(
        "accurate",
        description="One of 'accurate' or 'fast'. If 'accurate' the mask will be decoded using the original image size.",
    ),
    tradeoff_factor: Optional[float] = Query(
        0.0, description="The amount to tradeoff between 0='fast' and 1='accurate'"
    ),
    max_detections: int = Query(
        300, description="The maximum number of detections to return."
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
        description="Shared secret used to authenticate requests to the inference server from internal services",
        include_in_schema=False,
    ),
    disable_preproc_auto_orient: Optional[bool] = Query(
        False, description="If true, disables automatic image orientation"
    ),
    disable_preproc_contrast: Optional[bool] = Query(
        False, description="If true, disables automatic contrast adjustment"
    ),
    disable_preproc_grayscale: Optional[bool] = Query(
        False, description="If true, disables automatic grayscale conversion"
    ),
    disable_preproc_static_crop: Optional[bool] = Query(
        False, description="If true, disables automatic static crop"
    ),
    disable_active_learning: Optional[bool] = Query(
        False,
        description="If true, the predictions will be prevented from registration by Active Learning",
    ),
    active_learning_target_dataset: Optional[str] = Query(
        None,
        description="Parameter to be used when Active Learning data registration should happen against different dataset than the one pointed by model_id",
    ),
    include_anomaly_map: Optional[bool] = Query(
        False,
        description="Anomaly detection only: include the raw anomaly heatmap in original image coordinates",
    ),
    source: Optional[str] = Query(
        "external", description="The source of the inference request"
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
    response_mask_format: Optional[Literal["polygon", "rle"]] = Query(
        "polygon",
        description="The format of the prediction mask - polygon (default) or rle",
    ),
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    model_id = f"{dataset_id}/{version_id}"
    resolved_key = resolve_api_key(request, api_key, None)
    request_image = await _catch_all_image(request, image, image_type)
    if not countinference and not service_secret_is_valid(service_secret):
        raise MissingServiceSecretError()
    route = await bridge.resolve(
        model_id, resolved_key, row_key=model_id, path=request.scope["path"]
    )
    bridge.record_request(route, model_id, request.scope["path"])
    if route.task_type == ACTION_RECOGNITION_TASK and not route.is_stub:
        inference_request = ActionRecognitionInferenceRequest(
            api_key=resolved_key,
            model_id=model_id,
            video=InferenceRequestVideo(
                type=request_image.type, value=request_image.value
            ),
            class_filter=_parse_legacy_class_filter(class_filter),
        )
        http_response = await _classify_and_repack(inference_request, bridge, route)

        return http_response

    if isinstance(confidence, (int, float)):
        if confidence >= 1:
            confidence /= 100
        if confidence < configuration.CONFIDENCE_LOWER_BOUND_OOM_PREVENTION:
            confidence = configuration.CONFIDENCE_LOWER_BOUND_OOM_PREVENTION
    if overlap >= 1:
        overlap /= 100
    request_type = ObjectDetectionInferenceRequest
    extra_args: dict = {}
    if route.task_type == "instance-segmentation":
        request_type = InstanceSegmentationInferenceRequest
        extra_args = {
            "mask_decode_mode": mask_decode_mode,
            "tradeoff_factor": tradeoff_factor,
        }
        if response_mask_format:
            extra_args["response_mask_format"] = response_mask_format
    elif route.task_type == "classification":
        request_type = ClassificationInferenceRequest
        extra_args = {"include_anomaly_map": include_anomaly_map}
    elif route.task_type == "keypoint-detection":
        request_type = KeypointsDetectionInferenceRequest
        extra_args = {"keypoint_confidence": keypoint_confidence}
    elif route.task_type == "semantic-segmentation":
        request_type = SemanticSegmentationInferenceRequest
    try:
        inference_request = request_type(
            api_key=resolved_key,
            model_id=model_id,
            image=request_image,
            confidence=confidence,
            iou_threshold=overlap,
            max_detections=max_detections,
            visualization_labels=labels,
            visualization_stroke_width=stroke,
            visualize_predictions=format in _VISUALIZATION_FORMATS,
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
            **extra_args,
        )
    except ValidationError as error:
        raise LegacyHTTPError(400, str(error)) from error
    if route.is_stub:
        stub_response = _stub_response(inference_request, route)
        if format == "image":
            return Response(
                content=stub_response.visualization, media_type="image/jpeg"
            )
        return orjson_response(stub_response)

    return await _infer_and_repack(
        inference_request,
        bridge,
        route,
        resolved_key,
        image_format=format if format in _VISUALIZATION_FORMATS else None,
        active_learning_eligible=True,
    )


async def _resolve_core_model(
    request: Request,
    inference_request,
    bridge: LegacyModelBridge,
    core: str,
) -> Tuple[Route, str, Optional[str]]:
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), inference_request.api_key
    )
    inference_request.api_key = api_key
    core_model_id = f"{core}/{getattr(inference_request, f'{core}_version_id')}"
    route = await bridge.resolve(
        core_model_id, api_key, row_key=core_model_id, path=request.scope["path"]
    )
    bridge.record_request(route, core_model_id, request.scope["path"])
    return route, core_model_id, api_key


async def _load_images(inference_request, bridge: LegacyModelBridge):
    images, is_batch = as_image_list(inference_request.image)
    payloads = await load_request_images(images, ndarray_ok=bridge.accepts_ndarray)
    return payloads, is_batch


async def _run_embedding(
    request: Request,
    inference_request,
    bridge: LegacyModelBridge,
    core: str,
) -> Any:
    route, _, api_key = await _resolve_core_model(
        request, inference_request, bridge, core
    )
    action = resolve_request_action(route, inference_request)
    calls, prompt_keys = build_embedding_calls(action, inference_request)
    image_positions = [
        position for position, call in enumerate(calls) if call["image"] is not None
    ]
    payloads = await load_request_images(
        [calls[position]["image"] for position in image_positions],
        ndarray_ok=bridge.accepts_ndarray,
    )
    payload_by_position = dict(zip(image_positions, payloads))
    await bridge.ensure_loaded(route, api_key)
    model_monitoring = _model_monitoring_enabled(inference_request)
    started = time.perf_counter()
    results = []
    with measure_inference(
        route.registry_id,
        responses=1,
        monitoring=model_monitoring,
    ):
        for position, call in enumerate(calls):
            payload = payload_by_position.get(position)
            if payload is None:
                results.append(
                    await bridge.infer_params_only(
                        route,
                        api_key,
                        call["action"],
                        call["params"],
                        model_monitoring=model_monitoring,
                        record=False,
                    )
                )
                continue
            results.extend(
                await bridge.infer(
                    route,
                    api_key,
                    call["action"],
                    [payload],
                    call["params"],
                    model_monitoring=model_monitoring,
                    record=False,
                )
            )
    elapsed = time.perf_counter() - started
    record_telemetry(
        telemetry.record_inference, requested_model_id_for(route.registry_id), elapsed
    )
    response = repack_embedding_response(
        action, inference_request, results, prompt_keys
    )
    response.time = elapsed
    response.resolved_model = resolved_model_for(route)
    pingback.record_inference(route.registry_id, inference_request, response)
    return response


async def _run_ocr(
    request: Request,
    inference_request,
    bridge: LegacyModelBridge,
    core: str,
    *,
    structured: bool,
    generate_bounding_boxes: Optional[bool] = None,
    class_from_text: bool = False,
    params: Optional[dict] = None,
) -> Response:
    ensure_ocr_request_supported(inference_request)
    route, _, api_key = await _resolve_core_model(
        request, inference_request, bridge, core
    )
    payloads, is_batch = await _load_images(inference_request, bridge)
    started = time.perf_counter()
    predictions = await bridge.infer(
        route,
        api_key,
        route.action,
        payloads,
        params or {},
        model_monitoring=_model_monitoring_enabled(inference_request),
    )
    elapsed = time.perf_counter() - started
    responses = []
    for prediction, payload in zip(predictions, payloads):
        dims = (payload.width, payload.height)
        if structured:
            response = repack_structured_ocr_response(
                prediction,
                dims,
                route.class_names,
                inference_request,
                generate_bounding_boxes=generate_bounding_boxes,
                class_from_text=class_from_text,
            )
        else:
            response = repack_text_ocr_response(prediction, dims)
        response.time = elapsed
        response.resolved_model = resolved_model_for(route)
        responses.append(response)
    pingback.record_inference(route.registry_id, inference_request, responses)
    return orjson_response(responses if is_batch else responses[0], keep_parent_id=True)


async def _run_open_vocabulary_detection(
    request: Request,
    inference_request,
    bridge: LegacyModelBridge,
    core: str,
) -> Any:
    params = build_open_vocabulary_params(inference_request)
    route, core_model_id, api_key = await _resolve_core_model(
        request, inference_request, bridge, core
    )
    ensure_request_supported(core_model_id, inference_request, route)
    class_names = requested_open_vocabulary_classes(inference_request)
    payloads, is_batch = await _load_images(inference_request, bridge)
    started = time.perf_counter()
    predictions = await bridge.infer(
        route,
        api_key,
        route.action,
        payloads,
        params,
        model_monitoring=_model_monitoring_enabled(inference_request),
    )
    elapsed = time.perf_counter() - started
    responses = []
    for prediction, payload in zip(predictions, payloads):
        response = repack_object_detection_response(
            prediction, (payload.width, payload.height), class_names, inference_request
        )
        response.time = elapsed
        response.inference_id = inference_request.id
        response.resolved_model = resolved_model_for(route)
        responses.append(response)
    pingback.record_inference(route.registry_id, inference_request, responses)
    return responses if is_batch else responses[0]


async def _run_few_shot_detection(
    request: Request,
    inference_request,
    bridge: LegacyModelBridge,
    core: str,
) -> Any:
    route, _, api_key = await _resolve_core_model(
        request, inference_request, bridge, core
    )
    payloads, is_batch = await _load_images(inference_request, bridge)
    references = await load_request_images(
        [example.image for example in inference_request.training_data],
        ndarray_ok=False,
    )
    params = build_few_shot_params(
        inference_request, [reference.data for reference in references]
    )
    started = time.perf_counter()
    predictions = await bridge.infer(
        route,
        api_key,
        "infer_with_reference_examples",
        payloads,
        params,
        model_monitoring=_model_monitoring_enabled(inference_request),
    )
    elapsed = time.perf_counter() - started
    responses = []
    for prediction, payload in zip(predictions, payloads):
        response = repack_object_detection_response(
            prediction,
            (payload.width, payload.height),
            few_shot_class_names(prediction, inference_request),
            inference_request,
        )
        response.time = elapsed
        response.inference_id = inference_request.id
        response.resolved_model = resolved_model_for(route)
        if inference_request.visualize_predictions:
            response.visualization = render_visualization(
                route, inference_request, response, payload
            )
        responses.append(response)
    pingback.record_inference(route.registry_id, inference_request, responses)
    return responses if is_batch else responses[0]


@clip_router.post(
    "/clip/embed_image",
    response_model=ClipEmbeddingResponse,
    summary="CLIP Image Embeddings",
    description="Run the Open AI CLIP model to embed image data.",
)
@with_legacy_errors
@report_request_usage
async def clip_embed_image(
    request: Request,
    inference_request: ClipImageEmbeddingRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_embedding(request, inference_request, bridge, "clip")


@clip_router.post(
    "/clip/embed_text",
    response_model=ClipEmbeddingResponse,
    summary="CLIP Text Embeddings",
    description="Run the Open AI CLIP model to embed text data.",
)
@with_legacy_errors
@report_request_usage
async def clip_embed_text(
    request: Request,
    inference_request: ClipTextEmbeddingRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_embedding(request, inference_request, bridge, "clip")


@clip_router.post(
    "/clip/compare",
    response_model=ClipCompareResponse,
    summary="CLIP Compare",
    description="Run the Open AI CLIP model to compute similarity scores.",
)
@with_legacy_errors
@report_request_usage
async def clip_compare(
    request: Request,
    inference_request: ClipCompareRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_embedding(request, inference_request, bridge, "clip")


@perception_encoder_router.post(
    "/perception_encoder/embed_image",
    response_model=PerceptionEncoderEmbeddingResponse,
    summary="PE Image Embeddings",
    description="Run the Meta Perception Encoder model to embed image data.",
)
@with_legacy_errors
@report_request_usage
async def perception_encoder_embed_image(
    request: Request,
    inference_request: PerceptionEncoderImageEmbeddingRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_embedding(
        request, inference_request, bridge, "perception_encoder"
    )


@perception_encoder_router.post(
    "/perception_encoder/embed_text",
    response_model=PerceptionEncoderEmbeddingResponse,
    summary="PE Text Embeddings",
    description="Run the Meta Perception Encoder model to embed text data.",
)
@with_legacy_errors
@report_request_usage
async def perception_encoder_embed_text(
    request: Request,
    inference_request: PerceptionEncoderTextEmbeddingRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_embedding(
        request, inference_request, bridge, "perception_encoder"
    )


@perception_encoder_router.post(
    "/perception_encoder/compare",
    response_model=PerceptionEncoderCompareResponse,
    summary="PE Compare",
    description="Run the Meta Perception Encoder model to compute similarity scores.",
)
@with_legacy_errors
@report_request_usage
async def perception_encoder_compare(
    request: Request,
    inference_request: PerceptionEncoderCompareRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_embedding(
        request, inference_request, bridge, "perception_encoder"
    )


@doctr_router.post(
    "/doctr/ocr",
    response_model=Union[OCRInferenceResponse, List[OCRInferenceResponse]],
    summary="DocTR OCR response",
    description="Run the DocTR OCR model to retrieve text in an image.",
)
@with_legacy_errors
@report_request_usage
async def doctr_retrieve_text(
    request: Request,
    inference_request: DoctrOCRInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_ocr(request, inference_request, bridge, "doctr", structured=True)


@easy_ocr_router.post(
    "/easy_ocr/ocr",
    response_model=Union[OCRInferenceResponse, List[OCRInferenceResponse]],
    summary="EasyOCR OCR response",
    description="Run the EasyOCR model to retrieve text in an image.",
)
@with_legacy_errors
@report_request_usage
async def easy_ocr_retrieve_text(
    request: Request,
    inference_request: EasyOCRInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_ocr(
        request,
        inference_request,
        bridge,
        "easy_ocr",
        structured=True,
        generate_bounding_boxes=True,
        class_from_text=True,
        params={"confidence": 0.0},
    )


@trocr_router.post(
    "/ocr/trocr",
    response_model=Union[OCRInferenceResponse, List[OCRInferenceResponse]],
    summary="TrOCR OCR response",
    description="Run the TrOCR model to retrieve text in an image.",
)
@with_legacy_errors
@report_request_usage
async def trocr_retrieve_text(
    request: Request,
    inference_request: TrOCRInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_ocr(request, inference_request, bridge, "trocr", structured=False)


@pp_ocr_router.post(
    "/ocr/pp-ocr",
    response_model=Union[OCRInferenceResponse, List[OCRInferenceResponse]],
    summary="PP-OCRv6 OCR response",
    description="Run PP-OCRv6 two-stage OCR to retrieve text in an image.",
)
@with_legacy_errors
@report_request_usage
async def pp_ocr_retrieve_text(
    request: Request,
    inference_request: PPOCRInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_ocr(
        request,
        inference_request,
        bridge,
        "pp_ocr",
        structured=True,
        generate_bounding_boxes=True,
        class_from_text=True,
    )


@yolo_world_router.post(
    "/yolo_world/infer",
    response_model=Union[
        ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]
    ],
    summary="YOLO-World inference.",
    description="Run the YOLO-World zero-shot object detection model.",
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def yolo_world_infer(
    request: Request,
    inference_request: YOLOWorldInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    raise LegacyHTTPError(404, _YOLO_WORLD_UNSUPPORTED_MESSAGE)


@grounding_dino_router.post(
    "/grounding_dino/infer",
    response_model=Union[
        ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]
    ],
    summary="Grounding DINO inference.",
    description="Run the Grounding DINO zero-shot object detection model.",
)
@with_legacy_errors
@report_request_usage
async def grounding_dino_infer(
    request: Request,
    inference_request: GroundingDINOInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_open_vocabulary_detection(
        request, inference_request, bridge, "grounding_dino"
    )


@gaze_router.post(
    "/gaze/gaze_detection",
    summary="Gaze Detection (deprecated)",
    description=(
        "Deprecated. Always returns HTTP 410 Gone. The endpoint stub will be "
        "removed end of Q2 2026."
    ),
    deprecated=True,
)
async def gaze_detection_deprecated() -> Response:
    return JSONResponse(status_code=410, content=_GAZE_DEPRECATION_BODY)


async def _run_lmm(
    request: Request,
    inference_request: LMMInferenceRequest,
    bridge: LegacyModelBridge,
    model_id: Optional[str] = None,
) -> Any:
    if model_id is not None:
        if (
            inference_request.model_id is not None
            and inference_request.model_id != model_id
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model ID mismatch: path specifies '{model_id}' but request "
                    f"body specifies '{inference_request.model_id}'"
                ),
            )
        inference_request.model_id = model_id
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), inference_request.api_key
    )
    inference_request.api_key = api_key
    alias = request_alias_for(inference_request.model_id)
    route = await bridge.resolve(
        inference_request.model_id,
        api_key,
        row_key=inference_request.model_id,
        path=request.scope["path"],
        alias=alias,
    )
    bridge.record_request(
        route,
        inference_request.model_id,
        request.scope["path"],
        alias=alias,
    )
    ensure_request_supported(inference_request.model_id, inference_request, route)
    action = resolve_request_action(route, inference_request)
    if action == "detect":
        params = {"classes": [getattr(inference_request, "prompt", None)]}
    else:
        params = build_vlm_params(
            inference_request, model_class_name=route.model_class_name
        )
    payloads, is_batch = await _load_images(inference_request, bridge)
    started = time.perf_counter()
    predictions = await bridge.infer(
        route,
        api_key,
        action,
        payloads,
        params,
        model_monitoring=_model_monitoring_enabled(inference_request),
    )
    elapsed = time.perf_counter() - started
    responses = []
    for prediction, payload in zip(predictions, payloads):
        dims = (payload.width, payload.height)
        if action == "detect":
            response = repack_moondream_detection(prediction, inference_request, dims)
        else:
            response = repack_vlm_response(prediction, dims)
        response.time = elapsed
        response.inference_id = inference_request.id
        response.resolved_model = resolved_model_for(route)
        responses.append(response)
    pingback.record_inference(route.registry_id, inference_request, responses)
    return responses if is_batch else responses[0]


@lmm_router.post(
    "/infer/lmm",
    response_model=Union[
        LMMInferenceResponse,
        List[LMMInferenceResponse],
        ObjectDetectionInferenceResponse,
        List[ObjectDetectionInferenceResponse],
    ],
    summary="Large multi-modal model infer",
    description="Run inference with the specified large multi-modal model",
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def infer_lmm(
    request: Request,
    inference_request: LMMInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_lmm(request, inference_request, bridge)


@lmm_router.post(
    "/infer/lmm/{model_id:path}",
    response_model=Union[
        LMMInferenceResponse,
        List[LMMInferenceResponse],
        ObjectDetectionInferenceResponse,
        List[ObjectDetectionInferenceResponse],
    ],
    summary="Large multi-modal model infer with model ID in path",
    description=(
        "Run inference with the specified large multi-modal model. Model ID is "
        "specified in the URL path (can contain slashes)."
    ),
    response_model_exclude_none=True,
)
@with_legacy_errors
@report_request_usage
async def infer_lmm_with_model_id(
    request: Request,
    inference_request: LMMInferenceRequest,
    model_id: str = Path(description="Identifier of the model to run"),
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_lmm(request, inference_request, bridge, model_id=model_id)


async def _run_depth_estimation(
    request: Request,
    inference_request: DepthEstimationRequest,
    bridge: LegacyModelBridge,
    model_id: Optional[str] = None,
) -> Any:
    images, is_batch = as_image_list(inference_request.image)
    if is_batch:
        raise LegacyHTTPError(400, _DEPTH_SINGLE_IMAGE_MESSAGE)
    if model_id is not None:
        fields_set = getattr(inference_request, "model_fields_set", set())
        if (
            "model_id" in fields_set
            and inference_request.model_id is not None
            and inference_request.model_id != model_id
        ):
            raise LegacyHTTPError(
                400,
                f"Model ID mismatch: path specifies '{model_id}' but request body "
                f"specifies '{inference_request.model_id}'",
            )
        inference_request.model_id = model_id
    if inference_request.model_id is None:
        inference_request.model_id = (
            f"depth-anything-v2/{inference_request.depth_version_id}"
        )
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), inference_request.api_key
    )
    inference_request.api_key = api_key
    route = await bridge.resolve(
        inference_request.model_id,
        api_key,
        row_key=inference_request.model_id,
        path=request.scope["path"],
    )
    bridge.record_request(route, inference_request.model_id, request.scope["path"])
    payloads = await load_request_images(images, ndarray_ok=bridge.accepts_ndarray)
    started = time.perf_counter()
    predictions = await bridge.infer(
        route,
        api_key,
        route.action,
        payloads,
        {},
        model_monitoring=_model_monitoring_enabled(inference_request),
    )
    elapsed = time.perf_counter() - started
    depth = repack_depth_estimation(
        predictions[0], invert=is_metric_depth_model_class(route.model_class_name)
    )
    normalized_depth = depth["normalized_depth"]
    if inference_request.depth_map_format == "png8":
        serialized_depth = encode_normalized_depth_to_png8(normalized_depth)
    elif inference_request.depth_map_format == "png16":
        serialized_depth = encode_normalized_depth_to_png16(normalized_depth)
    else:
        serialized_depth = normalized_depth.tolist()
    response = DepthEstimationResponse(
        normalized_depth=serialized_depth,
        depth_map_format=inference_request.depth_map_format,
        image=depth["image"]["base64_image"],
    )
    response.time = elapsed
    response.inference_id = inference_request.id
    response.resolved_model = resolved_model_for(route)
    pingback.record_inference(route.registry_id, inference_request, response)
    return response


@depth_router.post(
    "/infer/depth-estimation",
    response_model=DepthEstimationResponse,
    summary="Depth Estimation",
    description="Run the depth estimation model to generate a depth map.",
)
@with_legacy_errors
@report_request_usage
async def depth_estimation(
    request: Request,
    inference_request: DepthEstimationRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_depth_estimation(request, inference_request, bridge)


@depth_router.post(
    "/infer/depth-estimation/{model_id:path}",
    response_model=DepthEstimationResponse,
    summary="Depth Estimation with model ID in path",
    description=(
        "Run depth estimation. Model ID is specified in the URL path and can "
        "contain slashes."
    ),
)
@with_legacy_errors
@report_request_usage
async def depth_estimation_with_model_id(
    request: Request,
    inference_request: DepthEstimationRequest,
    model_id: str = Path(description="Identifier of the model to run"),
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_depth_estimation(
        request, inference_request, bridge, model_id=model_id
    )


def _sam3_remote_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if configuration.ROBOFLOW_INTERNAL_SERVICE_NAME:
        headers["X-Roboflow-Internal-Service-Name"] = (
            configuration.ROBOFLOW_INTERNAL_SERVICE_NAME
        )
    if configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET:
        headers["X-Roboflow-Internal-Service-Secret"] = (
            configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET
        )
    all_headers = platform_http.build_api_headers(explicit_headers=headers)

    return all_headers


def _post_sam3_remote(
    proxy_path: str, payload: dict, api_key: Optional[str], response_model: type
) -> Any:
    url = platform_http.wrap_url(
        f"{configuration.API_BASE_URL}/inferenceproxy/{proxy_path}?api_key={api_key}"
    )
    response = platform_http._platform_request(
        "post", url, headers=_sam3_remote_headers(), json=payload, timeout=60
    )
    if response.status_code >= 400:
        raise RuntimeError(f"platform answered {response.status_code}")
    parsed = response_model(**response.json())

    return parsed


async def _sam3_remote(
    proxy_path: str,
    payload: dict,
    api_key: Optional[str],
    response_model: type,
    failure_message: str,
) -> Any:
    try:
        parsed = await asyncio.to_thread(
            _post_sam3_remote, proxy_path, payload, api_key, response_model
        )
    except Exception as error:
        logger.error("%s %s", failure_message, redact_text(str(error), (api_key,)))
        raise HTTPException(status_code=500, detail=failure_message) from None

    return parsed


def _sam3_concept_remote_payload(inference_request: Sam3SegmentationRequest) -> dict:
    prompts = []
    for prompt in inference_request.prompts:
        prompt_data = prompt.model_dump(exclude_none=True)
        if "type" not in prompt_data and "text" in prompt_data:
            prompt_data["type"] = "text"
        prompts.append(prompt_data)
    payload = {
        "image": {
            "type": inference_request.image.type,
            "value": inference_request.image.value,
        },
        "prompts": prompts,
        "output_prob_thresh": inference_request.output_prob_thresh,
        "source": inference_request.source,
        "source_info": inference_request.source_info,
    }

    return payload


def _sam3_visual_remote_payload(inference_request: Sam2SegmentationRequest) -> dict:
    prompts = (
        inference_request.prompts.model_dump(exclude_none=True)
        if inference_request.prompts
        else None
    )
    payload = {
        "image": {
            "type": inference_request.image.type,
            "value": inference_request.image.value,
        },
        "prompts": prompts,
        "multimask_output": inference_request.multimask_output,
        "source": inference_request.source,
        "source_info": inference_request.source_info,
    }

    return payload


async def _run_interactive_segmentation(
    request: Request,
    inference_request,
    bridge: LegacyModelBridge,
    model_id: str,
) -> Any:
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), inference_request.api_key
    )
    inference_request.api_key = api_key
    route = await bridge.resolve(
        model_id, api_key, row_key=model_id, path=request.scope["path"]
    )
    bridge.record_request(route, model_id, request.scope["path"])
    action = resolve_request_action(route, inference_request)
    params = build_interactive_segmentation_params(
        action, inference_request, api_key, model_id=model_id
    )
    image = getattr(inference_request, "image", None)
    model_monitoring = _model_monitoring_enabled(inference_request)
    started = time.perf_counter()
    if image is None:
        prediction = await bridge.infer_params_only(
            route, api_key, action, params, model_monitoring=model_monitoring
        )
    else:
        images, _ = as_image_list(image)
        payloads = await load_request_images(images, ndarray_ok=bridge.accepts_ndarray)
        prediction = (
            await bridge.infer(
                route,
                api_key,
                action,
                payloads,
                params,
                model_monitoring=model_monitoring,
            )
        )[0]
    elapsed = time.perf_counter() - started
    response = repack_interactive_segmentation_response(
        action, prediction, inference_request, api_key
    )
    if isinstance(response, bytes):
        return Response(
            content=response, headers={"Content-Type": "application/octet-stream"}
        )
    response.time = elapsed
    response.inference_id = inference_request.id
    response.resolved_model = resolved_model_for(route)
    pingback.record_inference(route.registry_id, inference_request, response)
    return response


@sam_router.post(
    "/sam/embed_image",
    response_model=SamEmbeddingResponse,
    summary="SAM Image Embeddings",
    description="Run the Meta AI Segment Anything Model to embed image data.",
)
@with_legacy_errors
@report_request_usage
async def sam_embed_image(
    request: Request,
    inference_request: SamEmbeddingRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    response = await _run_interactive_segmentation(
        request,
        inference_request,
        bridge,
        f"sam/{inference_request.sam_version_id}",
    )
    if inference_request.format == "binary":
        return Response(
            content=response.embeddings,
            headers={"Content-Type": "application/octet-stream"},
        )
    return response


@sam_router.post(
    "/sam/segment_image",
    response_model=SamSegmentationResponse,
    summary="SAM Image Segmentation",
    description=(
        "Run the Meta AI Segment Anything Model to generate segmentations for "
        "image data."
    ),
)
@with_legacy_errors
@report_request_usage
async def sam_segment_image(
    request: Request,
    inference_request: SamSegmentationRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_interactive_segmentation(
        request,
        inference_request,
        bridge,
        f"sam/{inference_request.sam_version_id}",
    )


@sam2_router.post(
    "/sam2/embed_image",
    response_model=Sam2EmbeddingResponse,
    summary="SAM2 Image Embeddings",
    description="Run the Meta AI Segment Anything 2 Model to embed image data.",
)
@with_legacy_errors
@report_request_usage
async def sam2_embed_image(
    request: Request,
    inference_request: Sam2EmbeddingRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_interactive_segmentation(
        request,
        inference_request,
        bridge,
        f"sam2/{inference_request.sam2_version_id}",
    )


@sam2_router.post(
    "/sam2/segment_image",
    response_model=Sam2SegmentationResponse,
    summary="SAM2 Image Segmentation",
    description=(
        "Run the Meta AI Segment Anything 2 Model to generate segmentations for "
        "image data."
    ),
)
@with_legacy_errors
@report_request_usage
async def sam2_segment_image(
    request: Request,
    inference_request: Sam2SegmentationRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_interactive_segmentation(
        request,
        inference_request,
        bridge,
        f"sam2/{inference_request.sam2_version_id}",
    )


@sam3_router.post(
    "/sam3/embed_image",
    response_model=Sam3EmbeddingResponse,
    summary="SAM3 Image Embeddings",
    description="Run the SAM3 interactive model to embed image data.",
)
@with_legacy_errors
@report_request_usage
async def sam3_embed_image(
    request: Request,
    inference_request: Sam2EmbeddingRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    if configuration.SAM3_EXEC_MODE == "remote":
        raise HTTPException(
            status_code=501, detail=SAM3_EMBEDDING_REMOTE_UNSUPPORTED_MESSAGE
        )
    return await _run_interactive_segmentation(
        request, inference_request, bridge, SAM3_INTERACTIVE_MODEL_ID
    )


@sam3_router.post(
    "/sam3/concept_segment",
    response_model=Sam3SegmentationResponse,
    summary="SAM3 PCS (promptable concept segmentation)",
    description=(
        "Run the SAM3 PCS (promptable concept segmentation) to generate "
        "segmentations for image data."
    ),
)
@with_legacy_errors
@report_request_usage
async def sam3_concept_segment(
    request: Request,
    inference_request: Sam3SegmentationRequest,
    request_source: Optional[str] = Query(
        None, alias="source", description="The source of the inference request"
    ),
    request_source_info: Optional[str] = Query(
        None,
        alias="source_info",
        description="The detailed source information of the inference request",
    ),
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    if request_source is not None:
        inference_request.source = request_source
    if request_source_info is not None:
        inference_request.source_info = request_source_info
    if not configuration.SAM3_FINE_TUNED_MODELS_ENABLED:
        if not inference_request.model_id.startswith("sam3/"):
            raise LegacyHTTPError(501, FINE_TUNED_SAM3_DEPLOYMENT_ERROR)
    if configuration.SAM3_EXEC_MODE == "remote":
        api_key = resolve_api_key(
            request, request.query_params.get("api_key"), inference_request.api_key
        )
        inference_request.api_key = api_key
        return await _sam3_remote(
            "seg-preview",
            _sam3_concept_remote_payload(inference_request),
            api_key,
            Sam3SegmentationResponse,
            SAM3_CONCEPT_REMOTE_FAILURE_MESSAGE,
        )
    return await _run_interactive_segmentation(
        request, inference_request, bridge, inference_request.model_id
    )


@sam3_router.post(
    "/sam3/visual_segment",
    response_model=Sam2SegmentationResponse,
    summary="SAM3 PVS (promptable visual segmentation)",
    description=(
        "Run the SAM3 PVS (promptable visual segmentation) to generate "
        "segmentations for image data."
    ),
)
@with_legacy_errors
@report_request_usage
async def sam3_visual_segment(
    request: Request,
    inference_request: Sam2SegmentationRequest,
    request_source: Optional[str] = Query(
        None, alias="source", description="The source of the inference request"
    ),
    request_source_info: Optional[str] = Query(
        None,
        alias="source_info",
        description="The detailed source information of the inference request",
    ),
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    if request_source is not None:
        inference_request.source = request_source
    if request_source_info is not None:
        inference_request.source_info = request_source_info
    if configuration.SAM3_EXEC_MODE == "remote":
        api_key = resolve_api_key(
            request, request.query_params.get("api_key"), inference_request.api_key
        )
        inference_request.api_key = api_key
        return await _sam3_remote(
            "sam3-pvs",
            _sam3_visual_remote_payload(inference_request),
            api_key,
            Sam2SegmentationResponse,
            SAM3_VISUAL_REMOTE_FAILURE_MESSAGE,
        )
    return await _run_interactive_segmentation(
        request, inference_request, bridge, SAM3_INTERACTIVE_MODEL_ID
    )
