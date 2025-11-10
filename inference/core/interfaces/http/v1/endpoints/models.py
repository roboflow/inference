"""
V1 API endpoints for model inference.

This module provides clean, efficient endpoints for running inference on various
model types using multipart form data and header-based authentication.
"""

from typing import List, Optional, Union

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse

from inference.core import logger
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    ObjectDetectionInferenceResponse,
    StubResponse,
)
from inference.core.exceptions import InputImageLoadError
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.interfaces.http.orjson_utils import orjson_response
from inference.core.interfaces.http.v1.auth import get_validated_api_key
from inference.core.interfaces.http.v1.multipart import (
    parse_multipart_with_images,
    upload_file_to_inference_request_image,
)
from inference.core.managers.base import ModelManager
from inference.models.aliases import resolve_roboflow_model_alias
from inference.usage_tracking.collector import usage_collector

router = APIRouter()


def create_model_endpoints(model_manager: ModelManager) -> APIRouter:
    """
    Create and configure v1 model inference endpoints.

    Args:
        model_manager: The model manager instance

    Returns:
        Configured APIRouter with all model endpoints
    """

    @router.post(
        "/object-detection/{model_id}",
        response_model=Union[
            ObjectDetectionInferenceResponse,
            List[ObjectDetectionInferenceResponse],
            StubResponse,
        ],
        summary="Object Detection (v1)",
        description="Run object detection inference using multipart form data. "
        "Images are uploaded as binary data avoiding base64 encoding overhead.",
        response_model_exclude_none=True,
        tags=["v1", "Object Detection"],
    )
    @with_route_exceptions
    @usage_collector("request")
    async def object_detection_v1(
        model_id: str = Path(..., description="Model ID (e.g., 'project-id/version-id')"),
        request: Request = None,
        background_tasks: BackgroundTasks = None,
        api_key: str = Depends(get_validated_api_key),
        countinference: Optional[bool] = Query(
            True,
            description="If false, does not track inference against usage",
            include_in_schema=False,
        ),
        service_secret: Optional[str] = Query(
            None,
            description="Service secret for internal requests",
            include_in_schema=False,
        ),
    ):
        """
        Object detection inference endpoint (v1).

        Expected multipart/form-data format:
        - image: Binary image file (required)
        - config: JSON string with inference parameters (optional)

        Config JSON schema:
        {
            "confidence": 0.5,           # Detection confidence threshold (0-1)
            "iou_threshold": 0.4,        # IoU threshold for NMS (0-1)
            "max_detections": 300,       # Maximum number of detections
            "visualize_predictions": false,  # Return visualization image
            "visualization_labels": true,    # Show labels in visualization
            "visualization_stroke_width": 1  # Stroke width for visualization
        }

        Authentication:
        - Header: Authorization: Bearer <api_key>
        - Header: X-Roboflow-Api-Key: <api_key>
        - Query param: ?api_key=<api_key>

        Returns:
            Object detection predictions with bounding boxes and class labels
        """
        logger.debug(f"[V1] Object detection request for model: {model_id}")

        # Parse multipart form data
        form_data = await request.form()
        images, config = await parse_multipart_with_images(form_data)

        # Validate we have at least one image
        if not images:
            raise InputImageLoadError(
                message="No image provided in multipart request",
                public_message="At least one image is required",
            )

        # Get the primary image (first one, or one named "image")
        image_key = "image" if "image" in images else list(images.keys())[0]
        image_upload = images[image_key]

        # Convert UploadFile to InferenceRequestImage (with raw bytes, no base64)
        request_image = await upload_file_to_inference_request_image(image_upload)

        # Resolve model alias
        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)

        # Build inference request
        inference_request = ObjectDetectionInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=request_image,
            confidence=config.get("confidence", 0.4),
            iou_threshold=config.get("iou_threshold", 0.3),
            max_detections=config.get("max_detections", 300),
            visualization_labels=config.get("visualization_labels", False),
            visualization_stroke_width=config.get("visualization_stroke_width", 1),
            visualize_predictions=config.get("visualize_predictions", False),
            disable_preproc_auto_orient=config.get("disable_preproc_auto_orient", False),
            disable_preproc_contrast=config.get("disable_preproc_contrast", False),
            disable_preproc_grayscale=config.get("disable_preproc_grayscale", False),
            disable_preproc_static_crop=config.get("disable_preproc_static_crop", False),
            disable_active_learning=config.get("disable_active_learning", False),
            active_learning_target_dataset=config.get("active_learning_target_dataset"),
            source=config.get("source", "v1_api"),
            source_info=config.get("source_info", "v1_object_detection"),
        )

        # Add model to manager
        model_manager.add_model(
            de_aliased_model_id,
            api_key,
            countinference=countinference,
            service_secret=service_secret,
        )

        # Run inference
        response = model_manager.infer_from_request_sync(
            de_aliased_model_id,
            inference_request,
            active_learning_eligible=True,
            background_tasks=background_tasks,
        )

        logger.debug(f"[V1] Object detection completed for model: {model_id}")
        return orjson_response(response)

    @router.post(
        "/instance-segmentation/{model_id}",
        response_model=Union[
            InstanceSegmentationInferenceResponse,
            List[InstanceSegmentationInferenceResponse],
            StubResponse,
        ],
        summary="Instance Segmentation (v1)",
        description="Run instance segmentation inference using multipart form data",
        response_model_exclude_none=True,
        tags=["v1", "Instance Segmentation"],
    )
    @with_route_exceptions
    @usage_collector("request")
    async def instance_segmentation_v1(
        model_id: str = Path(..., description="Model ID"),
        request: Request = None,
        background_tasks: BackgroundTasks = None,
        api_key: str = Depends(get_validated_api_key),
        countinference: Optional[bool] = Query(True, include_in_schema=False),
        service_secret: Optional[str] = Query(None, include_in_schema=False),
    ):
        """
        Instance segmentation inference endpoint (v1).

        Similar to object detection but returns instance masks in addition to bounding boxes.
        """
        logger.debug(f"[V1] Instance segmentation request for model: {model_id}")

        # Parse multipart form data
        form_data = await request.form()
        images, config = await parse_multipart_with_images(form_data)

        if not images:
            raise InputImageLoadError(
                message="No image provided in multipart request",
                public_message="At least one image is required",
            )

        image_key = "image" if "image" in images else list(images.keys())[0]
        image_upload = images[image_key]
        request_image = await upload_file_to_inference_request_image(image_upload)

        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)

        inference_request = InstanceSegmentationInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=request_image,
            confidence=config.get("confidence", 0.4),
            iou_threshold=config.get("iou_threshold", 0.3),
            max_detections=config.get("max_detections", 300),
            mask_decode_mode=config.get("mask_decode_mode", "accurate"),
            tradeoff_factor=config.get("tradeoff_factor", 0.0),
            visualization_labels=config.get("visualization_labels", False),
            visualization_stroke_width=config.get("visualization_stroke_width", 1),
            visualize_predictions=config.get("visualize_predictions", False),
        )

        model_manager.add_model(
            de_aliased_model_id,
            api_key,
            countinference=countinference,
            service_secret=service_secret,
        )

        response = model_manager.infer_from_request_sync(
            de_aliased_model_id,
            inference_request,
            active_learning_eligible=True,
            background_tasks=background_tasks,
        )

        logger.debug(f"[V1] Instance segmentation completed for model: {model_id}")
        return orjson_response(response)

    @router.post(
        "/classification/{model_id}",
        response_model=Union[
            ClassificationInferenceResponse,
            List[ClassificationInferenceResponse],
            StubResponse,
        ],
        summary="Classification (v1)",
        description="Run classification inference using multipart form data",
        response_model_exclude_none=True,
        tags=["v1", "Classification"],
    )
    @with_route_exceptions
    @usage_collector("request")
    async def classification_v1(
        model_id: str = Path(..., description="Model ID"),
        request: Request = None,
        background_tasks: BackgroundTasks = None,
        api_key: str = Depends(get_validated_api_key),
        countinference: Optional[bool] = Query(True, include_in_schema=False),
        service_secret: Optional[str] = Query(None, include_in_schema=False),
    ):
        """
        Classification inference endpoint (v1).

        Returns predicted class labels and confidence scores.
        """
        logger.debug(f"[V1] Classification request for model: {model_id}")

        form_data = await request.form()
        images, config = await parse_multipart_with_images(form_data)

        if not images:
            raise InputImageLoadError(
                message="No image provided in multipart request",
                public_message="At least one image is required",
            )

        image_key = "image" if "image" in images else list(images.keys())[0]
        image_upload = images[image_key]
        request_image = await upload_file_to_inference_request_image(image_upload)

        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)

        inference_request = ClassificationInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=request_image,
            confidence=config.get("confidence", 0.4),
            visualize_predictions=config.get("visualize_predictions", False),
        )

        model_manager.add_model(
            de_aliased_model_id,
            api_key,
            countinference=countinference,
            service_secret=service_secret,
        )

        response = model_manager.infer_from_request_sync(
            de_aliased_model_id,
            inference_request,
            active_learning_eligible=True,
            background_tasks=background_tasks,
        )

        logger.debug(f"[V1] Classification completed for model: {model_id}")
        return orjson_response(response)

    @router.post(
        "/keypoint-detection/{model_id}",
        response_model=Union[
            KeypointsDetectionInferenceResponse,
            List[KeypointsDetectionInferenceResponse],
            StubResponse,
        ],
        summary="Keypoint Detection (v1)",
        description="Run keypoint detection inference using multipart form data",
        response_model_exclude_none=True,
        tags=["v1", "Keypoint Detection"],
    )
    @with_route_exceptions
    @usage_collector("request")
    async def keypoint_detection_v1(
        model_id: str = Path(..., description="Model ID"),
        request: Request = None,
        background_tasks: BackgroundTasks = None,
        api_key: str = Depends(get_validated_api_key),
        countinference: Optional[bool] = Query(True, include_in_schema=False),
        service_secret: Optional[str] = Query(None, include_in_schema=False),
    ):
        """
        Keypoint detection inference endpoint (v1).

        Returns detected keypoints with confidence scores.
        """
        logger.debug(f"[V1] Keypoint detection request for model: {model_id}")

        form_data = await request.form()
        images, config = await parse_multipart_with_images(form_data)

        if not images:
            raise InputImageLoadError(
                message="No image provided in multipart request",
                public_message="At least one image is required",
            )

        image_key = "image" if "image" in images else list(images.keys())[0]
        image_upload = images[image_key]
        request_image = await upload_file_to_inference_request_image(image_upload)

        de_aliased_model_id = resolve_roboflow_model_alias(model_id=model_id)

        inference_request = KeypointsDetectionInferenceRequest(
            api_key=api_key,
            model_id=model_id,
            image=request_image,
            confidence=config.get("confidence", 0.4),
            iou_threshold=config.get("iou_threshold", 0.3),
            keypoint_confidence=config.get("keypoint_confidence", 0.0),
            max_detections=config.get("max_detections", 300),
            visualize_predictions=config.get("visualize_predictions", False),
        )

        model_manager.add_model(
            de_aliased_model_id,
            api_key,
            countinference=countinference,
            service_secret=service_secret,
        )

        response = model_manager.infer_from_request_sync(
            de_aliased_model_id,
            inference_request,
            active_learning_eligible=True,
            background_tasks=background_tasks,
        )

        logger.debug(f"[V1] Keypoint detection completed for model: {model_id}")
        return orjson_response(response)

    return router
