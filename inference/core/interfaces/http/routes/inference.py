"""Roboflow trained-model inference HTTP routes (/infer/*)."""

from typing import List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Query, Request, HTTPException

from inference.core import logger
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    DepthEstimationRequest,
    InferenceRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
    LMMInferenceRequest,
    SemanticSegmentationInferenceRequest,
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
    LMMInferenceResponse,
    SemanticSegmentationInferenceResponse,
)
from inference.core.env import DEPTH_ESTIMATION_ENABLED, LMM_ENABLED, MOONDREAM2_ENABLED
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.interfaces.http.orjson_utils import orjson_response
from inference.core.managers.base import ModelManager
from inference.models.aliases import resolve_roboflow_model_alias
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
        response_model=Union[SemanticSegmentationInferenceResponse, StubResponse],
        summary="Semantic segmentation infer",
        description="Run inference with the specified semantic segmentation model",
    )
    @with_route_exceptions
    @usage_collector("request")
    def infer_semantic_segmentation(
        inference_request: SemanticSegmentationInferenceRequest,
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

    if LMM_ENABLED or MOONDREAM2_ENABLED:
        @router.post(
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
            """Run inference with the specified large multi-modal model.

            Args:
                inference_request (LMMInferenceRequest): The request containing the necessary details for LMM inference.

            Returns:
                Union[LMMInferenceResponse, List[LMMInferenceResponse]]: The response containing the inference results.
            """
            logger.debug(f"Reached /infer/lmm")
            return process_inference_request(
                inference_request,
                countinference=countinference,
                service_secret=service_secret,
            )

        @router.post(
            "/infer/lmm/{model_id:path}",
            response_model=Union[
                LMMInferenceResponse,
                List[LMMInferenceResponse],
                StubResponse,
            ],
            summary="Large multi-modal model infer with model ID in path",
            description="Run inference with the specified large multi-modal model. Model ID is specified in the URL path (can contain slashes).",
            response_model_exclude_none=True,
        )
        @with_route_exceptions
        @usage_collector("request")
        def infer_lmm_with_model_id(
            model_id: str,
            inference_request: LMMInferenceRequest,
            countinference: Optional[bool] = None,
            service_secret: Optional[str] = None,
        ):
            """Run inference with the specified large multi-modal model.

            The model_id can be specified in the URL path. If model_id is also provided
            in the request body, it must match the path parameter.

            Args:
                model_id (str): The model identifier from the URL path.
                inference_request (LMMInferenceRequest): The request containing the necessary details for LMM inference.

            Returns:
                Union[LMMInferenceResponse, List[LMMInferenceResponse]]: The response containing the inference results.

            Raises:
                HTTPException: If model_id in path and request body don't match.
            """
            logger.debug(f"Reached /infer/lmm/{model_id}")

            # Validate model_id consistency between path and request body
            if (
                inference_request.model_id is not None
                and inference_request.model_id != model_id
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Model ID mismatch: path specifies '{model_id}' but request body specifies '{inference_request.model_id}'",
                )

            # Set the model_id from path if not in request body
            inference_request.model_id = model_id

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

