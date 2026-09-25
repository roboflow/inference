import base64
import logging
import time
from typing import Any, List, Literal, Optional, Tuple, Union

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
from starlette.datastructures import UploadFile

from inference_server import configuration
from inference_server.dependencies import get_model_manager
from inference_server.legacy.bridge import LegacyModelBridge, Route, resolved_model_for
from inference_server.legacy.common import (
    as_image_list,
    load_request_images,
    orjson_response,
    resolve_api_key,
)
from inference_server.legacy.cuda_health import check_cuda_health
from inference_server.legacy.entities import (
    AddModelRequest,
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
    TrOCRInferenceRequest,
    YOLOWorldInferenceRequest,
)
from inference_server.legacy.errors import LegacyHTTPError, with_legacy_errors
from inference_server.legacy.translation import (
    build_embedding_calls,
    build_interactive_segmentation_params,
    build_open_vocabulary_params,
    build_task_params,
    build_vlm_params,
    encode_normalized_depth_to_png8,
    encode_normalized_depth_to_png16,
    ensure_ocr_request_supported,
    ensure_request_supported,
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
from inference_server.legacy.visualization import render_visualization

logger = logging.getLogger(__name__)

router = APIRouter(tags=["legacy"])
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
    (("LMM_ENABLED", "MOONDREAM2_ENABLED"), lmm_router),
    (("DEPTH_ESTIMATION_ENABLED",), depth_router),
    (("CORE_MODEL_SAM_ENABLED",), sam_router),
    (("CORE_MODEL_SAM2_ENABLED",), sam2_router),
    (("CORE_MODEL_SAM3_ENABLED",), sam3_router),
    (("SAM3_3D_OBJECTS_ENABLED",), sam3_3d_router),
    (("ACTION_RECOGNITION_ENABLED",), action_recognition_router),
)

_VISUALIZATION_FORMATS = ("image", "image_and_json")
_CONTENT_TYPE_MISSING_MESSAGE = "Request must include a Content-Type header"
_MULTIPART_PART_MISSING_MESSAGE = (
    "Expected image to be send in part named 'file' of multipart/form-data request"
)
_EMPTY_BODY_MESSAGE = "Image not found in request body."
_YOLO_WORLD_UNSUPPORTED_MESSAGE = (
    "YOLO-World is not supported by this inference server configuration."
)

_TASK_UNAVAILABLE_MESSAGE = (
    "{route} is not available on inference_server: no model class for this task is "
    "registered with the model manager"
)
SAM3_REMOTE_UNSUPPORTED_MESSAGE = (
    "SAM3_EXEC_MODE=remote proxying is not available on inference_server"
)
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
    app.include_router(router)
    if configuration.CORE_MODELS_ENABLED:
        for flag_names, group_router in _CORE_MODEL_ROUTER_GROUPS:
            if any(getattr(configuration, name) for name in flag_names):
                app.include_router(group_router)
    if configuration.LEGACY_CONTROL_PLANE_ROUTES_ENABLED:
        app.include_router(control_plane_router)
        if configuration.GET_MODEL_REGISTRY_ENABLED:
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
    return ServerVersionInfo(
        name="Roboflow Inference Server",
        version=configuration.SERVER_VERSION,
        uuid=configuration.INFERENCE_SERVER_ID or configuration.SERVER_ID,
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
async def readiness(model_manager: Any = Depends(get_model_manager)) -> Response:
    try:
        stats = await model_manager.stats()
    except Exception:
        return JSONResponse(content={"status": "not ready"}, status_code=503)
    models = stats.get("models", {})
    for model_id in configuration.preload_model_ids():
        if models.get(model_id, {}).get("state") != "loaded":
            return JSONResponse(content={"status": "not ready"}, status_code=503)
    return JSONResponse(content={"status": "ready"})


async def _models_descriptions(bridge: LegacyModelBridge) -> ModelsDescriptions:
    return ModelsDescriptions.from_models_descriptions(
        [
            ModelDescriptionEntity(
                model_id=route.registry_id,
                task_type=route.task_type,
                request_aliases=sorted(route.request_aliases - {route.registry_id}),
                request_paths=sorted(route.request_paths),
            )
            for route in await bridge.describe()
        ]
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
    route = await bridge.load_pinned(add_model_request.model_id, api_key)
    bridge.record_request(route, add_model_request.model_id, request.scope["path"])
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
    await bridge.unload(clear_model_request.model_id)
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
    route = await bridge.load_pinned(model_id, resolved_key)
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
) -> Response:
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), inference_request.api_key
    )
    inference_request.api_key = api_key
    route = await bridge.resolve(inference_request.model_id, api_key)
    bridge.record_request(route, inference_request.model_id, request.scope["path"])
    if route.task_type not in expected_task_types:
        raise LegacyHTTPError(
            400,
            f"Model {inference_request.model_id!r} is a {route.task_type} model.",
        )
    return await _infer_and_repack(inference_request, bridge, route, api_key)


async def _infer_and_repack(
    inference_request,
    bridge: LegacyModelBridge,
    route: Route,
    api_key: Optional[str],
    image_format: Optional[str] = None,
) -> Response:
    ensure_request_supported(inference_request.model_id, inference_request, route)
    visualize = image_format in _VISUALIZATION_FORMATS or bool(
        getattr(inference_request, "visualize_predictions", False)
    )
    images, is_batch = as_image_list(inference_request.image)
    payloads = await load_request_images(images, ndarray_ok=bridge.accepts_ndarray)
    params = build_task_params(route.task_type, route.action, inference_request, route)
    started = time.perf_counter()
    predictions = await bridge.infer(route, api_key, route.action, payloads, params)
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
    if image_format == "image":
        return Response(
            content=responses[0].visualization if responses else None,
            media_type="image/jpeg",
        )
    return orjson_response(responses if is_batch else responses[0])


@router.post(
    "/infer/object_detection",
    response_model=Union[
        ObjectDetectionInferenceResponse, List[ObjectDetectionInferenceResponse]
    ],
    summary="Object detection infer",
    description="Run inference with the specified object detection model",
    response_model_exclude_none=True,
)
@with_legacy_errors
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
    )


@router.post(
    "/infer/instance_segmentation",
    response_model=Union[
        InstanceSegmentationInferenceResponse,
        List[InstanceSegmentationInferenceResponse],
    ],
    summary="Instance segmentation infer",
    description="Run inference with the specified instance segmentation model",
    response_model_exclude_none=True,
)
@with_legacy_errors
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
    )


@router.post(
    "/infer/semantic_segmentation",
    response_model=Union[
        SemanticSegmentationInferenceResponse,
        List[SemanticSegmentationInferenceResponse],
    ],
    summary="Semantic segmentation infer",
    description="Run inference with the specified semantic segmentation model",
    response_model_exclude_none=True,
)
@with_legacy_errors
async def infer_semantic_segmentation(
    request: Request,
    inference_request: SemanticSegmentationInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_cv_inference(
        request,
        inference_request,
        bridge,
        expected_task_types=("semantic-segmentation",),
    )


@router.post(
    "/infer/classification",
    response_model=Union[
        ClassificationInferenceResponse,
        List[ClassificationInferenceResponse],
        MultiLabelClassificationInferenceResponse,
        List[MultiLabelClassificationInferenceResponse],
    ],
    summary="Classification infer",
    description="Run inference with the specified classification model",
    response_model_exclude_none=True,
)
@with_legacy_errors
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
    )


@router.post(
    "/infer/keypoints_detection",
    response_model=Union[
        KeypointsDetectionInferenceResponse, List[KeypointsDetectionInferenceResponse]
    ],
    summary="Keypoints detection infer",
    description="Run inference with the specified keypoints detection model",
    response_model_exclude_none=True,
)
@with_legacy_errors
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


@action_recognition_router.post("/infer/action_recognition")
@with_legacy_errors
async def infer_action_recognition(request: Request) -> Response:
    raise LegacyHTTPError(
        501,
        _TASK_UNAVAILABLE_MESSAGE.format(route="/infer/action_recognition"),
    )


@sam3_3d_router.post("/sam3_3d/infer")
@with_legacy_errors
async def infer_sam3_3d(request: Request) -> Response:
    raise LegacyHTTPError(501, _TASK_UNAVAILABLE_MESSAGE.format(route="/sam3_3d/infer"))


@owlv2_router.post("/owlv2/infer")
@with_legacy_errors
async def infer_owlv2(request: Request) -> Response:
    raise LegacyHTTPError(
        501,
        "/owlv2/infer few-shot detection with training_data is not available "
        "on inference_server",
    )


async def _catch_all_image(
    request: Request, image: Optional[str], image_type: Optional[str]
) -> InferenceRequestImage:
    if image is not None:
        return InferenceRequestImage(type="url", value=image)
    content_type = request.headers.get("Content-Type")
    if content_type is None:
        raise LegacyHTTPError(400, _CONTENT_TYPE_MISSING_MESSAGE)
    if "multipart/form-data" in content_type:
        form = await request.form()
        part = form.get("file")
        if part is None:
            raise LegacyHTTPError(400, _MULTIPART_PART_MISSING_MESSAGE)
        data = await part.read() if isinstance(part, UploadFile) else part.encode()
        return InferenceRequestImage(
            type="base64", value=base64.b64encode(data).decode("ascii")
        )
    body = await request.body()
    if not body:
        raise LegacyHTTPError(400, _EMPTY_BODY_MESSAGE)
    return InferenceRequestImage(type=image_type, value=body)


@catch_all_router.get("/{dataset_id}/{version_id}")
@catch_all_router.post("/{dataset_id}/{version_id}")
@with_legacy_errors
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
    if isinstance(confidence, (int, float)):
        if confidence >= 1:
            confidence /= 100
        if confidence < configuration.CONFIDENCE_LOWER_BOUND_OOM_PREVENTION:
            confidence = configuration.CONFIDENCE_LOWER_BOUND_OOM_PREVENTION
    if overlap >= 1:
        overlap /= 100
    request_image = await _catch_all_image(request, image, image_type)
    route = await bridge.resolve(model_id, resolved_key)
    bridge.record_request(route, model_id, request.scope["path"])
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
    return await _infer_and_repack(
        inference_request,
        bridge,
        route,
        resolved_key,
        image_format=format if format in _VISUALIZATION_FORMATS else None,
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
    route = await bridge.resolve(core_model_id, api_key)
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
    started = time.perf_counter()
    results = []
    for position, call in enumerate(calls):
        payload = payload_by_position.get(position)
        if payload is None:
            results.append(
                await bridge.infer_params_only(
                    route, api_key, call["action"], call["params"]
                )
            )
            continue
        results.extend(
            await bridge.infer(
                route, api_key, call["action"], [payload], call["params"]
            )
        )
    elapsed = time.perf_counter() - started
    response = repack_embedding_response(
        action, inference_request, results, prompt_keys
    )
    response.time = elapsed
    response.resolved_model = resolved_model_for(route)
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
) -> Response:
    ensure_ocr_request_supported(inference_request)
    route, _, api_key = await _resolve_core_model(
        request, inference_request, bridge, core
    )
    payloads, is_batch = await _load_images(inference_request, bridge)
    started = time.perf_counter()
    predictions = await bridge.infer(route, api_key, route.action, payloads, {})
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
    predictions = await bridge.infer(route, api_key, route.action, payloads, params)
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
    return responses if is_batch else responses[0]


@clip_router.post(
    "/clip/embed_image",
    response_model=ClipEmbeddingResponse,
    summary="CLIP Image Embeddings",
    description="Run the Open AI CLIP model to embed image data.",
)
@with_legacy_errors
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
async def easy_ocr_retrieve_text(
    request: Request,
    inference_request: EasyOCRInferenceRequest,
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Response:
    return await _run_ocr(
        request, inference_request, bridge, "easy_ocr", structured=True
    )


@trocr_router.post(
    "/ocr/trocr",
    response_model=Union[OCRInferenceResponse, List[OCRInferenceResponse]],
    summary="TrOCR OCR response",
    description="Run the TrOCR model to retrieve text in an image.",
)
@with_legacy_errors
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
    route = await bridge.resolve(inference_request.model_id, api_key)
    bridge.record_request(route, inference_request.model_id, request.scope["path"])
    ensure_request_supported(inference_request.model_id, inference_request, route)
    action = resolve_request_action(route, inference_request)
    if action == "detect":
        params = {"classes": [getattr(inference_request, "prompt", None)]}
    else:
        params = build_vlm_params(inference_request)
    payloads, is_batch = await _load_images(inference_request, bridge)
    started = time.perf_counter()
    predictions = await bridge.infer(route, api_key, action, payloads, params)
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
    route = await bridge.resolve(inference_request.model_id, api_key)
    bridge.record_request(route, inference_request.model_id, request.scope["path"])
    payloads = await load_request_images(images, ndarray_ok=bridge.accepts_ndarray)
    started = time.perf_counter()
    predictions = await bridge.infer(route, api_key, route.action, payloads, {})
    elapsed = time.perf_counter() - started
    depth = repack_depth_estimation(predictions[0])
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
    return response


@depth_router.post(
    "/infer/depth-estimation",
    response_model=DepthEstimationResponse,
    summary="Depth Estimation",
    description="Run the depth estimation model to generate a depth map.",
)
@with_legacy_errors
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
async def depth_estimation_with_model_id(
    request: Request,
    inference_request: DepthEstimationRequest,
    model_id: str = Path(description="Identifier of the model to run"),
    bridge: LegacyModelBridge = Depends(get_bridge),
) -> Any:
    return await _run_depth_estimation(
        request, inference_request, bridge, model_id=model_id
    )


def _ensure_sam3_local_execution() -> None:
    if configuration.SAM3_EXEC_MODE == "remote":
        raise LegacyHTTPError(501, SAM3_REMOTE_UNSUPPORTED_MESSAGE)


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
    route = await bridge.resolve(model_id, api_key)
    bridge.record_request(route, model_id, request.scope["path"])
    action = resolve_request_action(route, inference_request)
    params = build_interactive_segmentation_params(action, inference_request, api_key)
    image = getattr(inference_request, "image", None)
    started = time.perf_counter()
    if image is None:
        prediction = await bridge.infer_params_only(route, api_key, action, params)
    else:
        images, _ = as_image_list(image)
        payloads = await load_request_images(images, ndarray_ok=bridge.accepts_ndarray)
        prediction = (await bridge.infer(route, api_key, action, payloads, params))[0]
    elapsed = time.perf_counter() - started
    response = repack_interactive_segmentation_response(
        action, prediction, inference_request, api_key
    )
    response.time = elapsed
    response.inference_id = inference_request.id
    response.resolved_model = resolved_model_for(route)
    return response


@sam_router.post(
    "/sam/embed_image",
    response_model=SamEmbeddingResponse,
    summary="SAM Image Embeddings",
    description="Run the Meta AI Segment Anything Model to embed image data.",
)
@with_legacy_errors
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
    _ensure_sam3_local_execution()
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
    _ensure_sam3_local_execution()
    return await _run_interactive_segmentation(
        request, inference_request, bridge, SAM3_INTERACTIVE_MODEL_ID
    )
