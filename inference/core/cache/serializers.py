from typing import List, Optional, Union

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
    *,
    model_id: Optional[str] = None,
) -> dict:
    """Build the model monitoring cache entry that the pingback reports.

    Args:
        infer_request: Request the model served.
        infer_response: Response or responses the model returned.
        model_id: Id of the model that served the request. It fills the cached
            request's ``model_id`` when the request leaves it empty, as core model
            requests such as YOLO-World and Grounding DINO do.

    Returns:
        Dict with the inference id, the server identity, the request and the
        response.
    """
    if not TINY_CACHE:
        request = jsonable_encoder(infer_request)
        response = jsonable_encoder(infer_response)
    else:
        included_request_fields = {
            "api_key",
            "confidence",
            "model_id",
            "model_type",
            "source",
            "source_info",
        }
        request = jsonable_encoder(infer_request.dict(include=included_request_fields))
        response = jsonable_encoder(build_condensed_response(infer_response))

    if request.get("model_id") is None:
        request["model_id"] = model_id

    return {
        "inference_id": infer_request.id,
        "inference_server_version": __version__,
        "inference_server_id": GLOBAL_INFERENCE_SERVER_ID,
        "request": request,
        "response": response,
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
