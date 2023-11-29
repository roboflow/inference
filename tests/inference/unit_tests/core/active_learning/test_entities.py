from unittest.mock import MagicMock

from inference.core.active_learning.entities import (
    ActiveLearningConfiguration,
    BatchReCreationInterval,
    ImageDimensions,
    StrategyLimit,
    StrategyLimitType,
)


def test_init_active_learning_configuration_when_all_values_provided() -> None:
    # given
    configuration = {
        "enabled": True,
        "max_image_size": (1200, 1300),
        "jpeg_compression_level": 75,
        "persist_predictions": True,
        "sampling_strategies": [
            {
                "name": "default_strategy",
                "type": "random_sampling",
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
    }
    sampling_methods = MagicMock()

    # when
    result = ActiveLearningConfiguration.init(
        roboflow_api_configuration=configuration,
        sampling_methods=sampling_methods,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
    )

    # then
    assert result == ActiveLearningConfiguration(
        max_image_size=ImageDimensions(height=1200, width=1300),
        jpeg_compression_level=75,
        persist_predictions=True,
        sampling_methods=sampling_methods,  # type: ignore
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=400,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
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


def test_init_active_learning_configuration_when_all_optional_values_missing() -> None:
    # given
    configuration = {
        "enabled": True,
        "persist_predictions": True,
        "sampling_strategies": [
            {
                "name": "default_strategy",
                "type": "random_sampling",
                "traffic_percentage": 0.1,
            }
        ],
        "batching_strategy": {
            "batches_name_prefix": "al_batch",
            "recreation_interval": "daily",
        },
    }
    sampling_methods = MagicMock()

    # when
    result = ActiveLearningConfiguration.init(
        roboflow_api_configuration=configuration,
        sampling_methods=sampling_methods,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
    )

    # then
    assert result == ActiveLearningConfiguration(
        max_image_size=None,
        jpeg_compression_level=95,
        persist_predictions=True,
        sampling_methods=sampling_methods,  # type: ignore
        batches_name_prefix="al_batch",
        batch_recreation_interval=BatchReCreationInterval.DAILY,
        max_batch_images=None,
        workspace_id="my_workspace",
        dataset_id="coin-detection",
        model_id="coin-detection/3",
        strategies_limits={"default_strategy": []},
        tags=[],
        strategies_tags={"default_strategy": []},
    )
