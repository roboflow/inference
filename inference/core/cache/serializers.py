from typing import List, Union

from fastapi.encoders import jsonable_encoder

from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
)
from inference.core.env import TINY_CACHE
from inference.core.logger import logger
from inference.core.version import __version__


def to_cachable_inference_item(
    infer_request: InferenceRequest,
    infer_response: Union[InferenceResponse, List[InferenceResponse]],
) -> dict:
    if not TINY_CACHE:
        return {
            "inference_id": infer_request.id,
            "inference_server_version": __version__,
            "inference_server_id": GLOBAL_INFERENCE_SERVER_ID,
            "request": jsonable_encoder(infer_request),
            "response": jsonable_encoder(infer_response),
        }

    included_request_fields = {
        "api_key",
        "confidence",
        "model_id",
        "model_type",
        "source",
        "source_info",
    }
    request = infer_request.dict(include=included_request_fields)
    response = build_condensed_response(infer_response)
    return {
        "inference_id": infer_request.id,
        "inference_server_version": __version__,
        "inference_server_id": GLOBAL_INFERENCE_SERVER_ID,
        "request": jsonable_encoder(request),
        "response": jsonable_encoder(response),
    }


def build_condensed_response(responses):
    if not isinstance(responses, list):
        responses = [responses]

    response_handlers = {
        ClassificationInferenceResponse: from_classification_response,
        MultiLabelClassificationInferenceResponse: from_multilabel_classification_response,
        ObjectDetectionInferenceResponse: from_object_detection_response,
        InstanceSegmentationInferenceResponse: from_instance_segmentation_response,
        KeypointsDetectionInferenceResponse: from_keypoints_detection_response,
    }

    formatted_responses = []
    for response in responses:
        if not getattr(response, "predictions", None):
            continue
        try:
            handler = None
            for cls, h in response_handlers.items():
                if isinstance(response, cls):
                    handler = h
                    break
            if handler:
                predictions = handler(response)
                formatted_responses.append(
                    {
                        "predictions": predictions,
                        "time": response.time,
                    }
                )
        except Exception as e:
            logger.warning(f"Error formatting response, skipping caching: {e}")

    return formatted_responses


def from_classification_response(response: ClassificationInferenceResponse):
    return [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in response.predictions
    ]


def from_multilabel_classification_response(
    response: MultiLabelClassificationInferenceResponse,
):
    return [
        {"class": cls, "confidence": pred.confidence}
        for cls, pred in response.predictions.items()
    ]


def from_object_detection_response(response: ObjectDetectionInferenceResponse):
    return [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in response.predictions
    ]


def from_instance_segmentation_response(
    response: InstanceSegmentationInferenceResponse,
):
    return [
        {"class": pred.class_name, "confidence": pred.confidence}
        for pred in response.predictions
    ]


def from_keypoints_detection_response(response: KeypointsDetectionInferenceResponse):
    return [
        {"class": keypoint.class_name, "confidence": keypoint.confidence}
        for pred in response.predictions
        for keypoint in pred.keypoints
    ]
