from unittest.mock import Mock, patch

import pytest

from inference.core.entities.requests.inference import ObjectDetectionInferenceRequest
from inference.core.managers import active_learning as active_learning_module
from inference.core.managers.active_learning import ActiveLearningManager
from inference.core.models.base import Model


@pytest.mark.parametrize("backend", [None, "onnx"])
def test_shared_package_keeps_active_learning_credentials_separate(
    monkeypatch, backend
):
    monkeypatch.setattr(active_learning_module, "USE_INFERENCE_MODELS", True)
    manager = ActiveLearningManager(
        model_registry=Mock(), cache=Mock(), content_addressed_artifact_cache=Mock()
    )
    model_id = "project/1:package:shared"
    manager._models[model_id] = Mock(spec=Model, task_type="object-detection")
    first_middleware, second_middleware = Mock(), Mock()
    with patch(
        "inference.core.managers.active_learning.ActiveLearningMiddleware.init",
        side_effect=[first_middleware, second_middleware],
    ) as initialize:
        for api_key in ("first-key", "second-key", "first-key"):
            request = ObjectDetectionInferenceRequest(
                model_id="project/1",
                backend=backend,
                api_key=api_key,
                image=[],
                id="request-id",
            )
            manager.register(prediction=Mock(), model_id=model_id, request=request)

    assert [call.kwargs["api_key"] for call in initialize.call_args_list] == [
        "first-key",
        "second-key",
    ]
    assert all(
        call.kwargs["model_id"] == "project/1" for call in initialize.call_args_list
    )
    assert first_middleware.register_batch.call_count == 2
    assert second_middleware.register_batch.call_count == 1
