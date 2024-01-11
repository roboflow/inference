from typing import Union

from fastapi.encoders import jsonable_encoder

from inference.core.version import __version__
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.env import CONDENSED_CACHED
from inference.core.logger import logger


def to_cachable_inference_item(
    infer_request: InferenceRequest,
    infer_response: Union[InferenceResponse, list[InferenceResponse]],
) -> dict:
    request = jsonable_encoder(infer_request.dict())
    response = jsonable_encoder(infer_response)
    logger.debug(f"request: {request}")
    logger.debug(f"response: {response}")
    if CONDENSED_CACHED:
        logger.debug("Condensing cached inference item...")
        return {
            "request": build_condensed_request(request),
            "response": build_condensed_response(response),
        }
    logger.debug("Not condensing cached inference item...")
    return {
        "request": request,
        "response": response,
    }


def build_condensed_request(request):
    return {
        "api_key": request.get("api_key"),
        "confidence": request.get("confidence"),
        "model_id": request.get("model_id"),
        "model_type": request.get("model_type"),
        "image_urls": [img for img in request.get("image") if img.get("type") == "url"],
    }


def build_condensed_response(responses):
    if isinstance(responses, list):
        return [_build_condensed_response(r) for r in responses]
    return _build_condensed_response(responses)


def _build_condensed_response(response):
    predictions = []
    for p in response.get("predictions", []):
        predictions.append(
            {
                "class": p.get("class"),
                "confidence": p.get("confidence"),
            }
        )
    return {
        "time": response.get("time"),
        "inference_server_version": __version__,
        "inference_server_id": GLOBAL_INFERENCE_SERVER_ID,
        "predictions": predictions,
    }
