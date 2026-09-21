import base64
import logging
import time
from typing import Any, List, Literal, Optional, Tuple, Union

from fastapi import APIRouter, Depends, FastAPI, Path, Query, Request, Response
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
    Confidence,
    InferenceRequestImage,
    InstanceSegmentationInferenceRequest,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceRequest,
    KeypointsDetectionInferenceResponse,
    ModelDescriptionEntity,
    ModelsDescriptions,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceRequest,
    ObjectDetectionInferenceResponse,
    SemanticSegmentationInferenceRequest,
    SemanticSegmentationInferenceResponse,
    ServerVersionInfo,
)
from inference_server.legacy.errors import LegacyHTTPError, with_legacy_errors
from inference_server.legacy.translation import (
    build_task_params,
    ensure_request_supported,
    repack_prediction,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["legacy"])
control_plane_router = APIRouter(tags=["legacy"])
registry_router = APIRouter(tags=["legacy"])
catch_all_router = APIRouter(tags=["legacy"])

VISUALIZATION_UNAVAILABLE_MESSAGE = (
    "Visualization is not available yet on inference_server"
)
_VISUALIZATION_FORMATS = ("image", "image_and_json")
_CONTENT_TYPE_MISSING_MESSAGE = "Request must include a Content-Type header"
_MULTIPART_PART_MISSING_MESSAGE = (
    "Expected image to be send in part named 'file' of multipart/form-data request"
)
_EMPTY_BODY_MESSAGE = "Image not found in request body."
_TASK_UNAVAILABLE_MESSAGE = (
    "{route} is not available on inference_server: no model class for this task is "
    "registered with the model manager"
)


def get_bridge(request: Request) -> LegacyModelBridge:
    return request.app.state.legacy_bridge


def include_legacy_routers(app: FastAPI) -> None:
    app.include_router(router)
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
) -> Response:
    ensure_request_supported(inference_request.model_id, inference_request, route)
    if getattr(inference_request, "visualize_predictions", False):
        raise LegacyHTTPError(501, VISUALIZATION_UNAVAILABLE_MESSAGE)
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
        responses.append(response)
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


if configuration.CORE_MODELS_ENABLED:

    if configuration.ACTION_RECOGNITION_ENABLED:

        @router.post("/infer/action_recognition")
        @with_legacy_errors
        async def infer_action_recognition(request: Request) -> Response:
            raise LegacyHTTPError(
                501,
                _TASK_UNAVAILABLE_MESSAGE.format(route="/infer/action_recognition"),
            )

    if configuration.SAM3_3D_OBJECTS_ENABLED:

        @router.post("/sam3_3d/infer")
        @with_legacy_errors
        async def infer_sam3_3d(request: Request) -> Response:
            raise LegacyHTTPError(
                501, _TASK_UNAVAILABLE_MESSAGE.format(route="/sam3_3d/infer")
            )

    if configuration.CORE_MODEL_OWLV2_ENABLED:

        @router.post("/owlv2/infer")
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
    if format in _VISUALIZATION_FORMATS:
        raise LegacyHTTPError(501, VISUALIZATION_UNAVAILABLE_MESSAGE)
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
    return await _infer_and_repack(inference_request, bridge, route, resolved_key)
