from typing import List, Optional, Union

from fastapi import APIRouter, BackgroundTasks

from inference.core import logger
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    DepthEstimationRequest,
    InferenceRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    DepthEstimationResponse,
    InferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    ObjectDetectionInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    StubResponse,
)
from inference.core.env import DEPTH_ESTIMATION_ENABLED
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.interfaces.http.orjson_utils import orjson_response
from inference.core.managers.base import ModelManager
from inference.core.utils.model_alias import resolve_roboflow_model_alias
from inference.usage_tracking.collector import usage_collector


def create_inference_router(
    model_manager: ModelManager,
) -> APIRouter:
    router = APIRouter()

    def process_inference_request(
        inference_request: InferenceRequest,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
        **kwargs,
    ) -> InferenceResponse:
        de_aliased_model_id = resolve_roboflow_model_alias(
            model_id=inference_request.model_id
        )
        model_manager.add_model(
            de_aliased_model_id,
            inference_request.api_key,
            countinference=countinference,
            service_secret=service_secret,
        )
        resp = model_manager.infer_from_request_sync(
            de_aliased_model_id,
            inference_request,
            **kwargs,
        )
        return orjson_response(resp)

    @router.post(
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
        logger.debug("Reached /infer/object_detection")
        return process_inference_request(
            inference_request,
            active_learning_eligible=True,
            background_tasks=background_tasks,
            countinference=countinference,
            service_secret=service_secret,
        )

    @router.post(
        "/infer/instance_segmentation",
        response_model=Union[InstanceSegmentationInferenceResponse, StubResponse],
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
        logger.debug("Reached /infer/instance_segmentation")
        return process_inference_request(
            inference_request,
            active_learning_eligible=True,
            background_tasks=background_tasks,
            countinference=countinference,
            service_secret=service_secret,
        )

    @router.post(
        "/infer/semantic_segmentation",
        response_model=Union[InstanceSegmentationInferenceResponse, StubResponse],
        summary="Semantic segmentation infer",
        description="Run inference with the specified semantic segmentation model",
    )
    @with_route_exceptions
    @usage_collector("request")
    def infer_semantic_segmentation(
        inference_request,
        background_tasks: BackgroundTasks,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ):
        logger.debug("Reached /infer/semantic_segmentation")
        return process_inference_request(
            inference_request,
            active_learning_eligible=True,
            background_tasks=background_tasks,
            countinference=countinference,
            service_secret=service_secret,
        )

    @router.post(
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
        logger.debug("Reached /infer/classification")
        return process_inference_request(
            inference_request,
            active_learning_eligible=True,
            background_tasks=background_tasks,
            countinference=countinference,
            service_secret=service_secret,
        )

    @router.post(
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
        logger.debug("Reached /infer/keypoints_detection")
        return process_inference_request(
            inference_request,
            countinference=countinference,
            service_secret=service_secret,
        )

    if DEPTH_ESTIMATION_ENABLED:

        @router.post(
            "/infer/depth-estimation",
            response_model=DepthEstimationResponse,
            summary="Depth Estimation",
            description="Run the depth estimation model to generate a depth map.",
        )
        @with_route_exceptions
        def depth_estimation(
            inference_request: DepthEstimationRequest,
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            logger.debug("Reached /infer/depth-estimation")
            depth_model_id = inference_request.model_id
            model_manager.add_model(
                depth_model_id,
                inference_request.api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            response = model_manager.infer_from_request_sync(
                depth_model_id, inference_request
            )
            return response

    return router

