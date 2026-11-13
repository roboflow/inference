import json
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from roboflow_workflows.prototypes.platform_errors import (
    RoboflowAPIConnectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIUnsuccessfulRequestError,
)
from roboflow_workflows.utils.in_memory_cache import InMemoryWorkflowsCache

from inference_server.active_learning.cache_operations import (
    get_current_strategy_limit_usage,
)
from inference_server.active_learning.configuration import (
    get_roboflow_project_metadata,
    prepare_active_learning_configuration,
    prepare_active_learning_configuration_inplace,
)
from inference_server.active_learning.core import (
    prepare_image_to_registration_with_metadata,
)
from inference_server.active_learning.entities import ImageDimensions, StrategyLimitType
from inference_server.active_learning.middlewares import ActiveLearningMiddleware

PLANTED_API_KEY = "planted-secret-api-key-0123456789"


def _project_configuration(**overrides: Any) -> Dict[str, Any]:
    configuration = {
        "enabled": True,
        "persist_predictions": True,
        "sampling_strategies": [
            {
                "name": "everything",
                "type": "random",
                "traffic_percentage": 1.1,
                "tags": ["strategy-tag"],
                "limits": [{"type": "hourly", "value": 2}],
            }
        ],
        "batching_strategy": {
            "batches_name_prefix": "al_batch",
            "recreation_interval": "never",
        },
        "tags": ["project-tag"],
    }
    configuration.update(overrides)

    return configuration


def _platform_client(
    project_type: str = "object-detection",
    configuration: Optional[Dict[str, Any]] = None,
) -> MagicMock:
    client = MagicMock()
    client.get_roboflow_workspace.return_value = "my-workspace"
    client.get_roboflow_dataset_type.return_value = project_type
    client.get_roboflow_active_learning_configuration.return_value = (
        configuration if configuration is not None else _project_configuration()
    )
    client.register_image_at_roboflow.return_value = {"success": True, "id": "rf-id"}
    client.annotate_image_at_roboflow.return_value = {"success": True}

    return client


def _middleware(
    client: MagicMock,
    cache: Optional[InMemoryWorkflowsCache] = None,
    model_id: str = "target/3",
) -> ActiveLearningMiddleware:
    middleware = ActiveLearningMiddleware.init(
        api_key=PLANTED_API_KEY,
        target_dataset="target",
        model_id=model_id,
        cache=cache if cache is not None else InMemoryWorkflowsCache(),
        platform_client=client,
    )

    return middleware


def _hourly_usage(cache: InMemoryWorkflowsCache) -> Optional[int]:
    return get_current_strategy_limit_usage(
        cache=cache,
        workspace="my-workspace",
        project="target",
        strategy_name="everything",
        limit_type=StrategyLimitType.HOURLY,
    )


def _detection_prediction() -> Dict[str, Any]:
    return {
        "image": {"width": 200, "height": 100},
        "predictions": [
            {
                "x": 100.0,
                "y": 50.0,
                "width": 40.0,
                "height": 20.0,
                "class": "a",
                "confidence": 0.9,
            }
        ],
    }


def test_detection_is_registered_with_tags_batch_and_annotation() -> None:
    client = _platform_client()
    cache = InMemoryWorkflowsCache()
    middleware = _middleware(client, cache=cache)
    prediction = _detection_prediction()

    middleware.register(
        image=np.zeros((100, 200, 3), dtype=np.uint8),
        prediction=prediction,
        prediction_type="object-detection",
        inference_id="inference-1",
    )

    upload = client.register_image_at_roboflow.call_args.kwargs
    assert upload["api_key"] == PLANTED_API_KEY
    assert upload["dataset_id"] == "target"
    assert upload["batch_name"] == "al_batch"
    assert upload["tags"] == ["project-tag", "strategy-tag", "target-3"]
    assert upload["inference_id"] == "inference-1"
    decoded = cv2.imdecode(
        np.frombuffer(upload["image_bytes"], dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert decoded.shape == (100, 200, 3)
    client.annotate_image_at_roboflow.assert_called_once_with(
        api_key=PLANTED_API_KEY,
        dataset_id="target",
        local_image_id=upload["local_image_id"],
        roboflow_image_id="rf-id",
        annotation_content=json.dumps(prediction),
        annotation_file_type="json",
        is_prediction=True,
    )
    assert _hourly_usage(cache) == 1


def test_downscaled_image_rescales_the_registered_prediction() -> None:
    client = _platform_client(
        configuration=_project_configuration(max_image_size=[50, 100])
    )
    middleware = _middleware(client)
    prediction = _detection_prediction()

    middleware.register(
        image=np.zeros((100, 200, 3), dtype=np.uint8),
        prediction=prediction,
        prediction_type="object-detection",
    )

    upload = client.register_image_at_roboflow.call_args.kwargs
    decoded = cv2.imdecode(
        np.frombuffer(upload["image_bytes"], dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert decoded.shape == (50, 100, 3)
    annotation = json.loads(
        client.annotate_image_at_roboflow.call_args.kwargs["annotation_content"]
    )
    assert annotation["image"] == {"width": 400, "height": 200}
    assert annotation["predictions"][0]["x"] == 200.0
    assert annotation["predictions"][0]["height"] == 40.0


def test_image_smaller_than_the_limit_is_not_upscaled() -> None:
    image = np.zeros((60, 80, 3), dtype=np.uint8)

    result = prepare_image_to_registration_with_metadata(
        image=image,
        desired_size=ImageDimensions(height=600, width=800),
        jpeg_compression_level=95,
    )

    decoded = cv2.imdecode(
        np.frombuffer(result.encoded_image, dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert decoded.shape == (60, 80, 3)
    assert result.final_size_wh == (80, 60)
    assert result.scaling_factor == 1.0


def test_image_larger_than_the_limit_is_downscaled_with_its_scaling_factor() -> None:
    image = np.zeros((600, 800, 3), dtype=np.uint8)

    result = prepare_image_to_registration_with_metadata(
        image=image,
        desired_size=ImageDimensions(height=150, width=400),
        jpeg_compression_level=95,
    )

    assert result.final_size_wh == (200, 150)
    assert result.scaling_factor == 0.25


def test_jpeg_compression_level_of_the_configuration_is_applied() -> None:
    image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)

    low = prepare_image_to_registration_with_metadata(
        image=image, desired_size=None, jpeg_compression_level=10
    )
    high = prepare_image_to_registration_with_metadata(
        image=image, desired_size=None, jpeg_compression_level=95
    )

    assert len(low.encoded_image) < len(high.encoded_image)


@pytest.mark.parametrize(
    "prediction_type", ["multi-label-classification", "classification"]
)
def test_multi_label_prediction_registers_the_image_without_annotation(
    prediction_type: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = _platform_client(project_type="multi-label-classification")
    middleware = _middleware(client)
    prediction = {
        "image": {"width": 200, "height": 100},
        "predictions": {"cat": {"confidence": 0.9}, "dog": {"confidence": 0.2}},
        "predicted_classes": ["cat"],
    }

    with caplog.at_level(logging.WARNING):
        middleware.register(
            image=np.zeros((100, 200, 3), dtype=np.uint8),
            prediction=prediction,
            prediction_type=prediction_type,
        )

    client.register_image_at_roboflow.assert_called_once()
    client.annotate_image_at_roboflow.assert_not_called()
    assert "Image registered without annotation" in caplog.text


def test_multi_class_prediction_is_annotated_with_its_top_class() -> None:
    client = _platform_client(project_type="classification")
    middleware = _middleware(client)

    middleware.register(
        image=np.zeros((100, 200, 3), dtype=np.uint8),
        prediction={
            "top": "cat",
            "confidence": 0.9,
            "predictions": [{"class": "cat", "confidence": 0.9}],
        },
        prediction_type="classification",
    )

    annotation = client.annotate_image_at_roboflow.call_args.kwargs
    assert annotation["annotation_content"] == "cat"
    assert annotation["annotation_file_type"] == "txt"


def test_duplicate_upload_returns_the_strategy_credit() -> None:
    client = _platform_client()
    client.register_image_at_roboflow.return_value = {"duplicate": True, "id": "x"}
    cache = InMemoryWorkflowsCache()
    middleware = _middleware(client, cache=cache)

    middleware.register(
        image=np.zeros((100, 200, 3), dtype=np.uint8),
        prediction=_detection_prediction(),
        prediction_type="object-detection",
    )

    client.annotate_image_at_roboflow.assert_not_called()
    assert _hourly_usage(cache) == 0


def test_failed_upload_returns_the_strategy_credit_and_raises() -> None:
    client = _platform_client()
    client.register_image_at_roboflow.side_effect = RoboflowAPIConnectionError("down")
    cache = InMemoryWorkflowsCache()
    middleware = _middleware(client, cache=cache)

    with pytest.raises(RoboflowAPIConnectionError):
        middleware.register(
            image=np.zeros((100, 200, 3), dtype=np.uint8),
            prediction=_detection_prediction(),
            prediction_type="object-detection",
        )

    assert _hourly_usage(cache) == 0


def test_failed_annotation_keeps_the_strategy_credit_and_raises() -> None:
    client = _platform_client()
    client.annotate_image_at_roboflow.side_effect = RoboflowAPIUnsuccessfulRequestError(
        "rejected"
    )
    cache = InMemoryWorkflowsCache()
    middleware = _middleware(client, cache=cache)

    with pytest.raises(RoboflowAPIUnsuccessfulRequestError):
        middleware.register(
            image=np.zeros((100, 200, 3), dtype=np.uint8),
            prediction=_detection_prediction(),
            prediction_type="object-detection",
        )

    assert _hourly_usage(cache) == 1


def test_strategy_limit_stops_registration() -> None:
    client = _platform_client()
    cache = InMemoryWorkflowsCache()
    middleware = _middleware(client, cache=cache)
    images: List[np.ndarray] = [np.zeros((10, 10, 3), dtype=np.uint8)] * 4

    middleware.register_batch(
        images=images,
        predictions=[_detection_prediction() for _ in images],
        prediction_type="object-detection",
    )

    assert client.register_image_at_roboflow.call_count == 2
    assert _hourly_usage(cache) == 2


def test_batch_image_limit_stops_registration() -> None:
    configuration = _project_configuration()
    configuration["batching_strategy"]["max_batch_images"] = 3
    client = _platform_client(configuration=configuration)
    client.get_roboflow_labeling_batches.return_value = {
        "batches": [{"name": "al_batch", "numJobs": 0, "images": 3, "id": "b"}]
    }
    middleware = _middleware(client)

    middleware.register(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        prediction=_detection_prediction(),
        prediction_type="object-detection",
    )

    client.register_image_at_roboflow.assert_not_called()


def test_configuration_is_cached_between_middlewares() -> None:
    client = _platform_client()
    cache = InMemoryWorkflowsCache()

    first = _middleware(client, cache=cache)
    calls_after_first = len(client.mock_calls)
    second = _middleware(client, cache=cache)

    assert first._configuration is not None
    assert second._configuration is not None
    assert len(client.mock_calls) == calls_after_first


def test_classification_model_into_detection_project_disables_active_learning() -> None:
    client = _platform_client()
    client.get_roboflow_dataset_type.side_effect = [
        "object-detection",
        "classification",
    ]

    middleware = _middleware(client, model_id="classifier/1")

    assert middleware._configuration is None
    client.get_roboflow_active_learning_configuration.assert_not_called()


def test_incompatible_types_disable_a_configuration_given_inplace() -> None:
    client = _platform_client()
    client.get_roboflow_dataset_type.side_effect = [
        "object-detection",
        "multi-label-classification",
    ]

    result = prepare_active_learning_configuration_inplace(
        api_key=PLANTED_API_KEY,
        target_dataset="target",
        model_id="classifier/1",
        active_learning_configuration=_project_configuration(),
        platform_client=client,
    )

    assert result is None


@pytest.mark.parametrize(
    "error_class", [RoboflowAPINotAuthorizedError, RoboflowAPINotNotFoundError]
)
def test_refused_configuration_read_disables_active_learning(error_class) -> None:
    client = _platform_client()
    client.get_roboflow_active_learning_configuration.side_effect = error_class("no")
    cache = InMemoryWorkflowsCache()

    metadata = get_roboflow_project_metadata(
        api_key=PLANTED_API_KEY,
        target_dataset="target",
        model_id="target/3",
        cache=cache,
        platform_client=client,
    )

    assert metadata.active_learning_configuration == {"enabled": False}


def test_target_workspace_and_project_override_the_registration_target() -> None:
    client = _platform_client(
        configuration=_project_configuration(
            target_workspace="other-workspace", target_project="other-project"
        )
    )
    middleware = _middleware(client)

    middleware.register(
        image=np.zeros((10, 10, 3), dtype=np.uint8),
        prediction=_detection_prediction(),
        prediction_type="object-detection",
    )

    assert middleware._configuration.workspace_id == "other-workspace"
    upload = client.register_image_at_roboflow.call_args.kwargs
    assert upload["dataset_id"] == "other-project"


def test_failing_platform_call_logs_no_api_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _platform_client()
    client.get_roboflow_dataset_type.side_effect = RoboflowAPIConnectionError(
        f"GET https://api.roboflow.com/ws/target?api_key={PLANTED_API_KEY} failed"
    )

    with caplog.at_level(logging.DEBUG):
        result = prepare_active_learning_configuration(
            api_key=PLANTED_API_KEY,
            target_dataset="target",
            model_id="target/3",
            cache=InMemoryWorkflowsCache(),
            platform_client=client,
        )

    assert result is None
    assert "RoboflowAPIConnectionError" in caplog.text
    assert PLANTED_API_KEY not in caplog.text
    assert "api_key=" not in caplog.text


def test_registration_logs_no_api_key(caplog: pytest.LogCaptureFixture) -> None:
    client = _platform_client()
    client.register_image_at_roboflow.return_value = {
        "duplicate": True,
        "url": f"https://api.roboflow.com/dataset/target/upload?api_key={PLANTED_API_KEY}",
    }

    with caplog.at_level(logging.DEBUG):
        middleware = _middleware(client)
        middleware.register(
            image=np.zeros((10, 10, 3), dtype=np.uint8),
            prediction=_detection_prediction(),
            prediction_type="object-detection",
        )

    assert "Image duplication detected" in caplog.text
    assert PLANTED_API_KEY not in caplog.text
