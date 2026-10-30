from unittest import mock

import pytest

from inference.core.cache import serializers
from inference.core.cache.memory import MemoryCache
from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest
from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.entities.responses.inference import (
    ObjectDetectionInferenceResponse,
    ObjectDetectionPrediction,
)
from inference.core.managers import base as base_module
from inference.core.managers import pingback as pingback_module
from inference.core.managers.base import ModelManager


def _detection_response(
    class_name: str, confidence: float
) -> ObjectDetectionInferenceResponse:
    response = ObjectDetectionInferenceResponse(
        predictions=[
            ObjectDetectionPrediction(
                **{
                    "x": 5,
                    "y": 5,
                    "width": 2,
                    "height": 2,
                    "confidence": confidence,
                    "class": class_name,
                    "class_id": 0,
                }
            )
        ],
        image={"width": 10, "height": 10},
        time=0.1,
    )

    return response


@pytest.mark.parametrize("tiny_cache", [True, False], ids=["tiny", "full"])
def test_post_data_sends_single_image_results_in_the_shape_roboflow_ingests(
    monkeypatch: pytest.MonkeyPatch,
    tiny_cache: bool,
) -> None:
    monkeypatch.setattr(serializers, "TINY_CACHE", tiny_cache)
    monkeypatch.setattr(
        base_module.model_monitoring_cache_module,
        "model_monitoring_cache",
        MemoryCache(),
    )
    monkeypatch.setattr(base_module, "METRICS_ENABLED", True)
    monkeypatch.setattr(base_module, "OFFLINE_MODE", False)
    monkeypatch.setattr(base_module, "DISABLE_INFERENCE_CACHE", False)
    monkeypatch.setattr(pingback_module, "OFFLINE_MODE", False)
    monkeypatch.setattr(pingback_module, "get_system_info", dict)
    monkeypatch.setattr(pingback_module.PingbackInfo, "start", mock.MagicMock())
    image = {"type": "url", "value": "https://some/image.jpg"}
    served = [
        (
            "some-project/1",
            ObjectDetectionInferenceRequest(
                model_id="some-project/1", image=image, confidence=0.4
            ),
            _detection_response("car", 0.9),
        ),
        (
            "yolo_world/l",
            YOLOWorldInferenceRequest(image=image, text=["person"], confidence=0.2),
            _detection_response("person", 0.8),
        ),
        (
            "grounding_dino/default",
            GroundingDINOInferenceRequest(image=image, text=["dog"], box_threshold=0.3),
            _detection_response("dog", 0.7),
        ),
    ]
    models = {model_id: mock.MagicMock() for model_id, _, _ in served}
    model_manager = ModelManager(model_registry=mock.MagicMock(), models=models)
    model_manager.init_pingback()

    for model_id, request, response in served:
        models[model_id].infer_from_request.return_value = response
        model_manager.infer_from_request_sync(model_id=model_id, request=request)
    with mock.patch.object(pingback_module.requests, "post") as post_mock:
        model_manager.pingback.post_data(model_manager)

    # What Roboflow's pingback mapper reads, one row per prediction
    rows = [
        (
            result["inference"]["request"]["model_id"],
            result["inference"]["request"]["confidence"],
            response["time"],
            prediction["class"],
            prediction["confidence"],
        )
        for result in post_mock.call_args.kwargs["json"]["inference_results"]
        for response in result["inference"]["response"]
        for prediction in response["predictions"]
    ]
    assert rows == [
        ("some-project/1", 0.4, 0.1, "car", 0.9),
        ("yolo_world/l", 0.2, 0.1, "person", 0.8),
        ("grounding_dino/default", 0.3, 0.1, "dog", 0.7),
    ]
