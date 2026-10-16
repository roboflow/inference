"""`anomaly_threshold` is only meaningful for anomaly detection models; the
model manager refuses it for every other model instead of silently ignoring it."""

from unittest.mock import MagicMock

import pytest

from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.exceptions import RequestDataContradiction
from inference.core.managers.base import (
    ModelManager,
    reject_anomaly_threshold_for_other_models,
)
from inference.core.models.base import Model

IMAGE = {"type": "url", "value": "https://example.com/test.jpg"}
SENTINEL = object()


class _PlainClassifier(Model):
    def infer_from_request(self, request):
        return SENTINEL


class _AnomalyDetector(Model):
    accepts_anomaly_threshold = True

    def infer_from_request(self, request):
        return SENTINEL


def _manager(model: Model) -> ModelManager:
    return ModelManager(model_registry=MagicMock(), models={"m/1": model})


def test_plain_classifier_rejects_anomaly_threshold() -> None:
    request = ClassificationInferenceRequest(
        model_id="m/1", image=IMAGE, anomaly_threshold=1.25
    )

    with pytest.raises(RequestDataContradiction, match="m/1"):
        _manager(_PlainClassifier()).model_infer_sync(model_id="m/1", request=request)


@pytest.mark.asyncio
async def test_plain_classifier_rejects_anomaly_threshold_on_async_path() -> None:
    request = ClassificationInferenceRequest(
        model_id="m/1", image=IMAGE, anomaly_threshold=1.25
    )

    with pytest.raises(RequestDataContradiction):
        await _manager(_PlainClassifier()).model_infer(model_id="m/1", request=request)


def test_plain_classifier_runs_without_anomaly_threshold() -> None:
    request = ClassificationInferenceRequest(model_id="m/1", image=IMAGE)

    result = _manager(_PlainClassifier()).model_infer_sync(
        model_id="m/1", request=request
    )

    assert result is SENTINEL


def test_anomaly_detector_accepts_anomaly_threshold() -> None:
    request = ClassificationInferenceRequest(
        model_id="m/1", image=IMAGE, anomaly_threshold=0.0
    )

    result = _manager(_AnomalyDetector()).model_infer_sync(
        model_id="m/1", request=request
    )

    assert result is SENTINEL


def test_guard_ignores_requests_of_other_task_types() -> None:
    request = ObjectDetectionInferenceRequest(model_id="m/1", image=IMAGE)

    assert (
        reject_anomaly_threshold_for_other_models(
            model=_PlainClassifier(), request=request
        )
        is None
    )
