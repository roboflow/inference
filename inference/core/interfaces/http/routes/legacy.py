import base64
from typing import Annotated, Any, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Depends, Path, Query, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile

from inference.core import logger
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    InstanceSegmentationInferenceRequest,
    InferenceRequestImage,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
    SemanticSegmentationInferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    SemanticSegmentationInferenceResponse,
    StubResponse,
)
from inference.core.env import (
    CONFIDENCE_LOWER_BOUND_OOM_PREVENTION,
    GCP_SERVERLESS,
    LAMBDA,
    LEGACY_ROUTE_ENABLED,
    ROBOFLOW_SERVICE_SECRET,
)
from inference.core.exceptions import (
    ContentTypeInvalid,
    ContentTypeMissing,
    InputImageLoadError,
    MissingServiceSecretError,
)
from inference.core.interfaces.http.dependencies import (
    parse_body_content_for_legacy_request_handler,
)
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.interfaces.http.orjson_utils import orjson_response
from inference.core.managers.base import ModelManager
from inference.usage_tracking.collector import usage_collector

if LAMBDA:
    from inference.core.usage import trackUsage


def create_legacy_router(model_manager: ModelManager) -> APIRouter:
    """Create router for legacy inference and cache endpoints.
    Infer route is added when LEGACY_ROUTE_ENABLED; clear_cache/start when not (LAMBDA or GCP_SERVERLESS).
    """
    router = APIRouter()

    if LEGACY_ROUTE_ENABLED:
        # Legacy object detection inference path for backwards compatibility
        @router.get(
            "/{dataset_id}/{version_id:str}",
            # Order matters in this response model Union. It will use the first matching model. For example, Object Detection Inference Response is a subset of Instance segmentation inference response, so instance segmentation must come first in order for the matching logic to work.
            response_model=Union[
                InstanceSegmentationInferenceResponse,
                KeypointsDetectionInferenceResponse,
                ObjectDetectionInferenceResponse,
                ClassificationInferenceResponse,
                MultiLabelClassificationInferenceResponse,
                SemanticSegmentationInferenceResponse,
                StubResponse,
                Any,
            ],
            response_model_exclude_none=True,
        )
        @router.post(
            "/{dataset_id}/{version_id:str}",
            # Order matters in this response model Union. It will use the first matching model. For example, Object Detection Inference Response is a subset of Instance segmentation inference response, so instance segmentation must come first in order for the matching logic to work.
            response_model=Union[
                InstanceSegmentationInferenceResponse,
                KeypointsDetectionInferenceResponse,
                ObjectDetectionInferenceResponse,
                ClassificationInferenceResponse,
                MultiLabelClassificationInferenceResponse,
                SemanticSegmentationInferenceResponse,
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
                background_tasks: FastAPI background tasks.
                dataset_id: Roboflow dataset ID or workspace ID.
                version_id: Dataset version ID or model ID.
                api_key: Optional API key for artifact retrieval.

            Returns:
                Inference result (type varies by model: detection, segmentation, classification, etc.).
            """
            logger.debug(
                f"Reached legacy route /:dataset_id/:version_id with {dataset_id}/{version_id}"
            )
            model_id = f"{dataset_id}/{version_id}"
            if confidence >= 1:
                confidence /= 100
            if confidence < CONFIDENCE_LOWER_BOUND_OOM_PREVENTION:
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
                f"State of model registry: {model_manager.describe_models()}"
            )
            model_manager.add_model(
                request_model_id,
                api_key,
                model_id_alias=model_id,
                countinference=countinference,
                service_secret=service_secret,
            )

            task_type = model_manager.get_task_type(model_id, api_key=api_key)
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
            elif task_type == "semantic-segmentation":
                inference_request_type = SemanticSegmentationInferenceRequest
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
            inference_response = model_manager.infer_from_request_sync(
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
        @router.get("/clear_cache", response_model=str)
        @with_route_exceptions
        def legacy_clear_cache():
            """
            Clears the model cache.

            This endpoint provides a way to clear the cache of loaded models.

            Returns:
                str: A string indicating that the cache has been cleared.
            """
            logger.debug(f"Reached /clear_cache")
            model_manager.clear()
            return "Cache Cleared"

        # Legacy add model endpoint for backwards compatibility
        @router.get("/start/{dataset_id}/{version_id}")
        @with_route_exceptions
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
            model_manager.add_model(
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

    return router
