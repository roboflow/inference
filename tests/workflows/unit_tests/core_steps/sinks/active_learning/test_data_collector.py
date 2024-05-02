from typing import Any

import pytest
from pydantic import ValidationError

from inference.enterprise.workflows.core_steps.sinks.active_learning.data_collector import (
    BlockManifest,
)
from inference.enterprise.workflows.core_steps.sinks.active_learning.entities import (
    ActiveLearningBatchingStrategy,
    ClassesBasedSampling,
    CloseToThresholdSampling,
    DetectionsBasedSampling,
    DisabledActiveLearningConfiguration,
    EnabledActiveLearningConfiguration,
    LimitDefinition,
    RandomSamplingConfig,
)


def test_validate_al_data_collector_when_valid_input_given() -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "prediction_type": "$steps.detection.prediction_type",
        "target_dataset": "some",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="ActiveLearningDataCollector",
        name="some",
        image="$inputs.image",
        predictions="$steps.detection.predictions",
        prediction_type="$steps.detection.prediction_type",
        target_dataset="some",
        target_dataset_api_key=None,
        disable_active_learning=False,
        active_learning_configuration=None,
    )


def test_validate_al_data_collector_when_valid_input_with_disabled_al_config_given() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "prediction_type": "$steps.detection.prediction_type",
        "target_dataset": "some",
        "active_learning_configuration": {"enabled": False},
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="ActiveLearningDataCollector",
        name="some",
        image="$inputs.image",
        predictions="$steps.detection.predictions",
        prediction_type="$steps.detection.prediction_type",
        target_dataset="some",
        target_dataset_api_key=None,
        disable_active_learning=False,
        active_learning_configuration=DisabledActiveLearningConfiguration(
            enabled=False
        ),
    )


def test_validate_al_data_collector_when_valid_input_with_enabled_al_config_given() -> (
    None
):
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "prediction_type": "$steps.detection.prediction_type",
        "target_dataset": "some",
        "active_learning_configuration": {
            "enabled": True,
            "persist_predictions": True,
            "sampling_strategies": [
                {
                    "type": "random",
                    "name": "a",
                    "traffic_percentage": 0.6,
                    "limits": [{"type": "daily", "value": 100}],
                },
                {
                    "type": "close_to_threshold",
                    "name": "b",
                    "probability": 0.7,
                    "threshold": 0.5,
                    "epsilon": 0.25,
                    "tags": ["some"],
                    "limits": [{"type": "daily", "value": 200}],
                },
                {
                    "type": "classes_based",
                    "name": "c",
                    "probability": 0.8,
                    "selected_class_names": ["a", "b", "c"],
                    "limits": [{"type": "daily", "value": 300}],
                },
                {
                    "type": "detections_number_based",
                    "name": "d",
                    "probability": 0.9,
                    "more_than": 3,
                    "less_than": 5,
                    "limits": [{"type": "daily", "value": 400}],
                },
            ],
            "batching_strategy": {
                "batches_name_prefix": "my_batches",
                "recreation_interval": "monthly",
            },
        },
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type="ActiveLearningDataCollector",
        name="some",
        image="$inputs.image",
        predictions="$steps.detection.predictions",
        prediction_type="$steps.detection.prediction_type",
        target_dataset="some",
        target_dataset_api_key=None,
        disable_active_learning=False,
        active_learning_configuration=EnabledActiveLearningConfiguration(
            enabled=True,
            persist_predictions=True,
            sampling_strategies=[
                RandomSamplingConfig(
                    type="random",
                    name="a",
                    traffic_percentage=0.6,
                    tags=[],
                    limits=[LimitDefinition(type="daily", value=100)],
                ),
                CloseToThresholdSampling(
                    type="close_to_threshold",
                    name="b",
                    probability=0.7,
                    threshold=0.5,
                    epsilon=0.25,
                    max_batch_images=None,
                    only_top_classes=True,
                    minimum_objects_close_to_threshold=1,
                    selected_class_names=None,
                    tags=["some"],
                    limits=[LimitDefinition(type="daily", value=200)],
                ),
                ClassesBasedSampling(
                    type="classes_based",
                    name="c",
                    probability=0.8,
                    selected_class_names=["a", "b", "c"],
                    tags=[],
                    limits=[LimitDefinition(type="daily", value=300)],
                ),
                DetectionsBasedSampling(
                    type="detections_number_based",
                    name="d",
                    probability=0.9,
                    more_than=3,
                    less_than=5,
                    selected_class_names=None,
                    tags=[],
                    limits=[LimitDefinition(type="daily", value=400)],
                ),
            ],
            batching_strategy=ActiveLearningBatchingStrategy(
                batches_name_prefix="my_batches",
                recreation_interval="monthly",
            ),
            tags=[],
            max_image_size=None,
            jpeg_compression_level=95,
        ),
    )


@pytest.mark.parametrize("image_selector", [1, None, "some", 1.3, True])
def test_validate_al_data_collector_image_field_when_field_does_not_hold_selector(
    image_selector: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": image_selector,
        "predictions": "$steps.detection.predictions",
        "prediction_type": "$steps.detection.prediction_type",
        "target_dataset": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("predictions_selector", [1, None, "some", 1.3, True])
def test_validate_al_data_collector_predictions_field_when_field_does_not_hold_selector(
    predictions_selector: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": predictions_selector,
        "prediction_type": "$steps.detection.prediction_type",
        "target_dataset": "some",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("target_dataset", [1, None, 1.3, True])
def test_validate_al_data_collector_target_dataset_field_when_field_contains_invalid_value(
    target_dataset: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "prediction_type": "$steps.detection.prediction_type",
        "target_dataset": target_dataset,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("target_dataset_api_key", [1, 1.3, True])
def test_validate_al_data_collector_target_dataset_api_key_field_when_field_contains_invalid_value(
    target_dataset_api_key: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "prediction_type": "$steps.detection.prediction_type",
        "target_dataset": "some",
        "target_dataset_api_key": target_dataset_api_key,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("disable_active_learning", ["some"])
def test_validate_al_data_collector_disable_active_learning_field_when_field_contains_invalid_value(
    disable_active_learning: Any,
) -> None:
    # given
    specification = {
        "type": "ActiveLearningDataCollector",
        "name": "some",
        "image": "$inputs.image",
        "predictions": "$steps.detection.predictions",
        "prediction_type": "$steps.detection.prediction_type",
        "target_dataset": "some",
        "disable_active_learning": disable_active_learning,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)
