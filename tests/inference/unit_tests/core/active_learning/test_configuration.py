from dataclasses import replace
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.active_learning import configuration
from inference.core.active_learning.configuration import (
    get_roboflow_project_metadata,
    initialize_sampling_methods,
    prepare_active_learning_configuration,
)
from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    BatchReCreationInterval,
    ImageDimensions,
    RoboflowProjectMetadata,
    StrategyLimit,
    StrategyLimitType,
)
from inference.core.exceptions import ActiveLearningConfigurationError


def test_initialize_sampling_methods() -> None:
    # given
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

    # when
    result = initialize_sampling_methods(
        sampling_strategies_configs=sampling_strategies_configs
    )

    # then
    assert len(result) == 4
    assert [r.name for r in result] == [
        "default_strategy",
        "hard_examples",
        "underrepresented_classes",
        "low_detections",
    ]
    for strategy in result:
        _ = strategy.sample(  # test if sampling executed correctly
            np.zeros((128, 128, 3), dtype=np.ndarray),
            {"is_stub": "True"},
            "object-detection",
        )


@mock.patch.object(configuration, "get_roboflow_active_learning_configuration")
@mock.patch.object(configuration, "get_roboflow_dataset_type")
@mock.patch.object(configuration, "get_roboflow_workspace")
def test_get_roboflow_project_metadata(
    get_roboflow_workspace_mock: MagicMock,
    get_roboflow_dataset_type_mock: MagicMock,
    get_roboflow_active_learning_configuration_mock: MagicMock,
) -> None:
    # given
    get_roboflow_workspace_mock.return_value = "my-workspace"
    get_roboflow_dataset_type_mock.return_value = "object-detection"
    get_roboflow_active_learning_configuration_mock.return_value = {"some": "config"}

    # when
    result = get_roboflow_project_metadata(api_key="api-key", model_id="some/1")

    # then
    assert result == RoboflowProjectMetadata(
        dataset_id="some",
        version_id="1",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={"some": "config"},
    )
    get_roboflow_workspace_mock.assert_called_once_with(api_key="api-key")
    get_roboflow_dataset_type_mock.assert_called_once_with(
        api_key="api-key",
        workspace_id="my-workspace",
        dataset_id="some",
    )
    get_roboflow_active_learning_configuration_mock.assert_called_once_with(
        api_key="api-key",
        workspace_id="my-workspace",
        dataset_id="some",
    )


@mock.patch.object(configuration, "ACTIVE_LEARNING_ENABLED", False)
def test_prepare_active_learning_configuration_when_active_learning_disabled_by_env() -> (
    None
):
    # when
    result = prepare_active_learning_configuration(
        api_key="api-key",
        model_id="some/1",
    )

    # then
    assert result is None


@mock.patch.object(configuration, "ACTIVE_LEARNING_ENABLED", True)
@mock.patch.object(configuration, "get_roboflow_project_metadata")
def test_prepare_active_learning_configuration_when_active_learning_disabled_by_configuration(
    get_roboflow_project_metadata_mock: MagicMock,
) -> None:
    # given
    get_roboflow_project_metadata_mock.return_value = RoboflowProjectMetadata(
        dataset_id="some",
        version_id="1",
        workspace_id="my-workspace",
        dataset_type="object-detection",
        active_learning_configuration={"enabled": False},
    )

    # when
    result = prepare_active_learning_configuration(
        api_key="api-key",
        model_id="some/1",
    )

    # then
    assert result is None


@mock.patch.object(configuration, "ACTIVE_LEARNING_ENABLED", True)
@mock.patch.object(configuration, "get_roboflow_project_metadata")
def test_prepare_active_learning_configuration_when_active_learning_enabled(
    get_roboflow_project_metadata_mock: MagicMock,
) -> None:
    # given
    get_roboflow_project_metadata_mock.return_value = RoboflowProjectMetadata(
        dataset_id="some",
        version_id="1",
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

    # when
    result = prepare_active_learning_configuration(
        api_key="api-key",
        model_id="some/1",
    )

    # then
    sampling_methods = result.sampling_methods
    assert len(sampling_methods) == 1
    assert sampling_methods[0].name == "default_strategy"
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
    )


def test_test_initialize_sampling_methods_when_duplicate_names_detected() -> None:
    # given
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
