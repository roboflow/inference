import hashlib
import logging
from dataclasses import asdict, replace
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from roboflow_workflows.utils.in_memory_cache import InMemoryWorkflowsCache

from inference_server.active_learning import configuration
from inference_server.active_learning.configuration import (
    get_roboflow_project_metadata,
    initialize_sampling_methods,
    predictions_incompatible_with_dataset,
    prepare_active_learning_configuration,
)
from inference_server.active_learning.entities import (
    ActiveLearningConfiguration,
    ActiveLearningConfigurationDecodingError,
    ActiveLearningConfigurationError,
    BatchReCreationInterval,
    ImageDimensions,
    RoboflowProjectMetadata,
    StrategyLimit,
    StrategyLimitType,
)


def test_initialize_sampling_methods() -> None:
    sampling_strategies_configs = [
        {
            "name": "default_strategy",
            "type": "random",
            "traffic_percentage": 0.5,
            "tags": ["a"],
            "limits": [
                {"type": "minutely", "value": 10},
                {"type": "hourly", "value": 100},
                {"type": "daily", "value": 1000},
            ],
        },
        {"type": "non-existing"},
        {
            "name": "hard_examples",
            "type": "close_to_threshold",
            "selected_class_names": ["a", "b"],
            "threshold": 0.25,
            "epsilon": 0.1,
            "probability": 0.5,
            "tags": ["b"],
            "limits": [
                {"type": "minutely", "value": 10},
                {"type": "hourly", "value": 100},
                {"type": "daily", "value": 1000},
            ],
        },
        {
            "name": "underrepresented_classes",
            "type": "classes_based",
            "selected_class_names": ["a"],
            "probability": 0.5,
            "tags": ["hard_classes"],
            "limits": [
                {"type": "minutely", "value": 10},
                {"type": "hourly", "value": 100},
                {"type": "daily", "value": 1000},
            ],
        },
        {
            "name": "low_detections",
            "type": "detections_number_based",
            "probability": 0.5,
            "less_than": 3,
            "tags": ["empty"],
            "limits": [
                {"type": "minutely", "value": 10},
                {"type": "hourly", "value": 100},
                {"type": "daily", "value": 1000},
            ],
        },
    ]

    result = initialize_sampling_methods(
        sampling_strategies_configs=sampling_strategies_configs
    )

    assert len(result) == 4
    assert [r.name for r in result] == [
        "default_strategy",
        "hard_examples",
        "underrepresented_classes",
        "low_detections",
    ]
    for strategy in result:
        _ = strategy.sample(
            np.zeros((128, 128, 3), dtype=np.ndarray),
            {"is_stub": "True"},
            "object-detection",
        )


@mock.patch.object(configuration, "get_model_type")
def test_get_roboflow_project_metadata_when_cache_miss_encountered(
    get_model_type_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    platform_client.get_roboflow_workspace.return_value = "my-workspace"
    platform_client.get_roboflow_dataset_type.return_value = "object-detection"
    platform_client.get_roboflow_active_learning_configuration.return_value = {
        "some": "config"
    }
    cache = InMemoryWorkflowsCache()

    result = get_roboflow_project_metadata(
        api_key="api-key",
        target_dataset="some",
        model_id="some/1",
        cache=cache,
        platform_client=platform_client,
    )

    expected_configuration = RoboflowProjectMetadata(
        dataset_id="some",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={"some": "config"},
    )
    assert (
        result == expected_configuration
    ), "Returned configuration must contain values defined in mocks"
    api_key_hash = hashlib.md5(b"api-key").hexdigest()
    assert cache.get(
        f"active_learning:configurations:{api_key_hash}:some:some/1"
    ) == asdict(
        expected_configuration
    ), "Configuration (serialised to dict) must be saved in cache"
    platform_client.get_roboflow_workspace.assert_called_once_with(api_key="api-key")
    platform_client.get_roboflow_dataset_type.assert_called_once_with(
        api_key="api-key",
        workspace_id="my-workspace",
        dataset_id="some",
    )
    platform_client.get_roboflow_active_learning_configuration.assert_called_once_with(
        api_key="api-key",
        workspace_id="my-workspace",
        dataset_id="some",
    )
    get_model_type_mock.assert_not_called()


@mock.patch.object(configuration, "get_model_type")
def test_get_roboflow_project_metadata_when_cache_miss_encountered_and_model_dataset_differs_from_target_dataset(
    get_model_type_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    platform_client.get_roboflow_workspace.return_value = "my-workspace"
    platform_client.get_roboflow_dataset_type.return_value = "object-detection"
    get_model_type_mock.return_value = "object-detection"
    platform_client.get_roboflow_active_learning_configuration.return_value = {
        "some": "config"
    }
    cache = InMemoryWorkflowsCache()

    result = get_roboflow_project_metadata(
        api_key="api-key",
        target_dataset="other",
        model_id="some/1",
        cache=cache,
        platform_client=platform_client,
    )

    expected_configuration = RoboflowProjectMetadata(
        dataset_id="other",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={"some": "config"},
    )
    assert (
        result == expected_configuration
    ), "Returned configuration must contain values defined in mocks"
    api_key_hash = hashlib.md5(b"api-key").hexdigest()
    assert cache.get(
        f"active_learning:configurations:{api_key_hash}:other:some/1"
    ) == asdict(
        expected_configuration
    ), "Configuration (serialised to dict) must be saved in cache"
    platform_client.get_roboflow_workspace.assert_called_once_with(api_key="api-key")
    platform_client.get_roboflow_dataset_type.assert_called_once_with(
        api_key="api-key",
        workspace_id="my-workspace",
        dataset_id="other",
    )
    platform_client.get_roboflow_active_learning_configuration.assert_called_once_with(
        api_key="api-key",
        workspace_id="my-workspace",
        dataset_id="other",
    )
    get_model_type_mock.assert_called_once_with(
        model_id="some/1", api_key="api-key", platform_client=platform_client
    )


@mock.patch.object(configuration, "get_model_type")
def test_get_roboflow_project_metadata_when_cache_miss_encountered_and_missmatch_in_registration_types_encountered(
    get_model_type_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    platform_client.get_roboflow_workspace.return_value = "my-workspace"
    platform_client.get_roboflow_dataset_type.return_value = "object-detection"
    get_model_type_mock.return_value = "classification"
    platform_client.get_roboflow_active_learning_configuration.return_value = {
        "some": "config"
    }
    cache = InMemoryWorkflowsCache()

    result = get_roboflow_project_metadata(
        api_key="api-key",
        target_dataset="other",
        model_id="some/1",
        cache=cache,
        platform_client=platform_client,
    )

    expected_configuration = RoboflowProjectMetadata(
        dataset_id="other",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={"enabled": False},
    )
    assert (
        result == expected_configuration
    ), "Returned configuration must contain values defined in mocks"
    api_key_hash = hashlib.md5(b"api-key").hexdigest()
    assert cache.get(
        f"active_learning:configurations:{api_key_hash}:other:some/1"
    ) == asdict(
        expected_configuration
    ), "Configuration (serialised to dict) must be saved in cache"
    platform_client.get_roboflow_workspace.assert_called_once_with(api_key="api-key")
    platform_client.get_roboflow_dataset_type.assert_called_once_with(
        api_key="api-key",
        workspace_id="my-workspace",
        dataset_id="other",
    )
    platform_client.get_roboflow_active_learning_configuration.assert_not_called()
    get_model_type_mock.assert_called_once_with(
        model_id="some/1", api_key="api-key", platform_client=platform_client
    )


def test_get_roboflow_project_metadata_when_cache_hit_encountered() -> None:
    platform_client = MagicMock()
    cache = InMemoryWorkflowsCache()
    api_key_hash = hashlib.md5(b"api-key").hexdigest()
    cache.set(
        key=f"active_learning:configurations:{api_key_hash}:other:some/1",
        value={
            "dataset_id": "some",
            "workspace_id": "my-workspace",
            "dataset_type": "object-detection",
            "active_learning_configuration": {"some": "config"},
        },
    )

    result = get_roboflow_project_metadata(
        api_key="api-key",
        target_dataset="other",
        model_id="some/1",
        cache=cache,
        platform_client=platform_client,
    )

    assert result == RoboflowProjectMetadata(
        dataset_id="some",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={"some": "config"},
    ), "Result must be cache content after de-serialisation"
    assert platform_client.mock_calls == []


def test_get_roboflow_project_metadata_when_cache_hit_encountered_but_content_is_malformed() -> (
    None
):
    cache = InMemoryWorkflowsCache()
    api_key_hash = hashlib.md5(b"api-key").hexdigest()
    cache.set(
        key=f"active_learning:configurations:{api_key_hash}:other:some/1",
        value={
            "dataset_id": "some",
            "workspace_id": "my-workspace",
            "dataset_type": "object-detection",
        },
    )

    with pytest.raises(ActiveLearningConfigurationDecodingError):
        _ = get_roboflow_project_metadata(
            api_key="api-key",
            target_dataset="other",
            model_id="some/1",
            cache=cache,
            platform_client=MagicMock(),
        )


@mock.patch.object(configuration, "get_roboflow_project_metadata")
def test_prepare_active_learning_configuration_when_active_learning_disabled_by_configuration(
    get_roboflow_project_metadata_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    cache = InMemoryWorkflowsCache()
    get_roboflow_project_metadata_mock.return_value = RoboflowProjectMetadata(
        dataset_id="some",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={"enabled": False},
    )

    result = prepare_active_learning_configuration(
        api_key="api-key",
        target_dataset="other",
        model_id="some/1",
        cache=cache,
        platform_client=platform_client,
    )

    assert result is None, "Expected null config when AL is disabled"


@mock.patch.object(configuration, "get_roboflow_project_metadata")
def test_prepare_active_learning_configuration_does_not_log_configuration_content(
    get_roboflow_project_metadata_mock: MagicMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    get_roboflow_project_metadata_mock.return_value = RoboflowProjectMetadata(
        dataset_id="some",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={
            "enabled": True,
            "target_workspace": "another-workspace",
            "target_project": "another-project",
            "max_image_size": (1200, 1300),
            "jpeg_compression_level": 75,
            "persist_predictions": True,
            "sampling_strategies": [
                {
                    "name": "SECRET123-strategy",
                    "type": "random",
                    "traffic_percentage": 0.1,
                    "tags": ["c"],
                    "limits": [],
                },
                {"name": "other", "type": "SECRET123-type"},
            ],
            "batching_strategy": {
                "batches_name_prefix": "SECRET123-batch",
                "recreation_interval": "daily",
                "max_batch_images": 400,
            },
            "tags": ["https://example.com/?api_key=SECRET123"],
            "unknown_key": "SECRET123",
        },
    )

    with caplog.at_level(logging.DEBUG):
        result = prepare_active_learning_configuration(
            api_key="api-key",
            target_dataset="other",
            model_id="some/1",
            cache=InMemoryWorkflowsCache(),
            platform_client=MagicMock(),
        )

    assert result is not None
    assert "SECRET123" not in caplog.text
    assert all("SECRET123" not in r.getMessage() for r in caplog.records)


@mock.patch.object(configuration, "get_roboflow_project_metadata")
def test_prepare_active_learning_configuration_when_active_learning_enabled(
    get_roboflow_project_metadata_mock: MagicMock,
) -> None:
    platform_client = MagicMock()
    cache = InMemoryWorkflowsCache()
    get_roboflow_project_metadata_mock.return_value = RoboflowProjectMetadata(
        dataset_id="some",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={
            "enabled": True,
            "target_workspace": "another-workspace",
            "target_project": "another-project",
            "max_image_size": (1200, 1300),
            "jpeg_compression_level": 75,
            "persist_predictions": True,
            "sampling_strategies": [
                {
                    "name": "default_strategy",
                    "type": "random",
                    "traffic_percentage": 0.1,
                    "tags": ["c", "d"],
                    "limits": [
                        {"type": "minutely", "value": 10},
                        {"type": "hourly", "value": 100},
                        {"type": "daily", "value": 1000},
                    ],
                }
            ],
            "batching_strategy": {
                "batches_name_prefix": "al_batch",
                "recreation_interval": "daily",
                "max_batch_images": 400,
            },
            "tags": ["a", "b"],
        },
    )

    result = prepare_active_learning_configuration(
        api_key="api-key",
        target_dataset="other",
        model_id="some/1",
        cache=cache,
        platform_client=platform_client,
    )

    sampling_methods = result.sampling_methods
    assert len(sampling_methods) == 1, "One sampling method was defined"
    assert (
        sampling_methods[0].name == "default_strategy"
    ), "Name of sampling method must match config"
    sampling_methods_mock = MagicMock()
    result = replace(result, sampling_methods=sampling_methods_mock)
    assert result == ActiveLearningConfiguration(
        max_image_size=ImageDimensions(height=1200, width=1300),
        jpeg_compression_level=75,
        persist_predictions=True,
        sampling_methods=sampling_methods_mock,
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=400,
        workspace_id="another-workspace",
        dataset_id="another-project",
        model_id="some/1",
        strategies_limits={
            "default_strategy": [
                StrategyLimit(limit_type=StrategyLimitType.MINUTELY, value=10),
                StrategyLimit(limit_type=StrategyLimitType.HOURLY, value=100),
                StrategyLimit(limit_type=StrategyLimitType.DAILY, value=1000),
            ]
        },
        tags=["a", "b"],
        strategies_tags={"default_strategy": ["c", "d"]},
    ), "Configuration must be parsed correctly according to project metadata"


def test_test_initialize_sampling_methods_when_duplicate_names_detected() -> None:
    sampling_strategies_configs = [
        {
            "name": "default_strategy",
            "type": "random",
            "traffic_percentage": 0.5,
            "tags": ["a"],
            "limits": [
                {"type": "minutely", "value": 10},
                {"type": "hourly", "value": 100},
                {"type": "daily", "value": 1000},
            ],
        },
        {
            "name": "default_strategy",
            "type": "close_to_threshold",
            "selected_class_names": ["a", "b"],
            "threshold": 0.25,
            "epsilon": 0.1,
            "probability": 0.5,
            "tags": ["b"],
            "limits": [
                {"type": "minutely", "value": 10},
                {"type": "hourly", "value": 100},
                {"type": "daily", "value": 1000},
            ],
        },
    ]

    with pytest.raises(ActiveLearningConfigurationError):
        _ = initialize_sampling_methods(
            sampling_strategies_configs=sampling_strategies_configs
        )


@pytest.mark.parametrize(
    "model_type, dataset_type",
    [
        ("classification", "classification"),
        ("object-detection", "object-detection"),
        ("object-detection", "keypoint-detection"),
        ("object-detection", "instance-segmentation"),
        ("instance-segmentation", "keypoint-detection"),
        ("keypoint-detection", "keypoint-detection"),
        ("instance-segmentation", "instance-segmentation"),
    ],
)
def test_predictions_incompatible_with_dataset_when_elements_are_compatible(
    model_type: str,
    dataset_type: str,
) -> None:
    result = predictions_incompatible_with_dataset(
        model_type=model_type,
        dataset_type=dataset_type,
    )

    assert result is False


@pytest.mark.parametrize(
    "model_type, dataset_type",
    [
        ("classification", "object-detection"),
        ("classification", "keypoint-detection"),
        ("classification", "instance-segmentation"),
        ("object-detection", "classification"),
        ("keypoint-detection", "classification"),
        ("instance-segmentation", "classification"),
    ],
)
def test_predictions_incompatible_with_dataset_when_elements_are_incompatible(
    model_type: str,
    dataset_type: str,
) -> None:
    result = predictions_incompatible_with_dataset(
        model_type=model_type,
        dataset_type=dataset_type,
    )

    assert result is True
