from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.active_learning.samplers import contains_classes
from inference.core.active_learning.samplers.contains_classes import (
    initialize_classes_based_sampling,
    sample_based_on_classes,
)
from inference.core.constants import CLASSIFICATION_TASK, OBJECT_DETECTION_TASK
from inference.core.exceptions import ActiveLearningConfigurationError

MULTI_LABEL_CLASSIFICATION_PREDICTION = {
    "image": {"width": 416, "height": 416},
    "predictions": {
        "cat": {"confidence": 0.97},
        "dog": {"confidence": 0.03},
    },
    "predicted_classes": ["cat"],
}


def test_sample_based_on_classes_for_detection_predictions() -> None:
    # given
    prediction = {
        "predictions": [
            {
                "x": 784.5,
                "y": 397.5,
                "width": 187.0,
                "height": 309.0,
                "confidence": 0.9,
                "class": "a",
                "class_id": 1,
            }
        ]
    }

    # when
    result = sample_based_on_classes(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=prediction,
        prediction_type=OBJECT_DETECTION_TASK,
        selected_class_names={"a", "b", "c"},
        probability=1.0,
    )

    # then
    assert result is False


def test_sample_based_on_classes_for_classification_prediction_when_classes_detected() -> (
    None
):
    # when
    result = sample_based_on_classes(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=MULTI_LABEL_CLASSIFICATION_PREDICTION,
        prediction_type=CLASSIFICATION_TASK,
        selected_class_names={"cat"},
        probability=1.0,
    )

    # then
    assert result is True


def test_sample_based_on_classes_for_classification_prediction_when_classes_not_detected() -> (
    None
):
    # when
    result = sample_based_on_classes(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=MULTI_LABEL_CLASSIFICATION_PREDICTION,
        prediction_type=CLASSIFICATION_TASK,
        selected_class_names={"dog"},
        probability=1.0,
    )

    # then
    assert result is False


def test_initialize_classes_based_sampling() -> None:
    # given
    strategy_config = {
        "name": "detect_dogs",
        "probability": 1.0,
        "selected_class_names": ["dog"],
    }

    # when
    sampling_method = initialize_classes_based_sampling(strategy_config=strategy_config)
    result = sampling_method.sample(
        np.zeros((128, 128, 3), dtype=np.uint8),
        MULTI_LABEL_CLASSIFICATION_PREDICTION,
        CLASSIFICATION_TASK,
    )

    # then
    assert result is False


@mock.patch.object(contains_classes, "partial")
def test_initialize_classes_based_sampling_against_parameters_correctness(
    partial_mock: MagicMock,
) -> None:
    # given
    strategy_config = {
        "name": "detect_dogs",
        "probability": 1.0,
        "selected_class_names": ["dog"],
    }

    # when
    _ = initialize_classes_based_sampling(strategy_config=strategy_config)

    # then
    partial_mock.assert_called_once_with(
        sample_based_on_classes,
        selected_class_names={"dog"},
        probability=1.0,
    )


def test_initialize_classes_based_sampling_when_configuration_key_missing() -> None:
    # given
    strategy_config = {
        "name": "detect_dogs",
        "selected_class_names": ["dog"],
        "minimum_objects": 10,
    }

    # when
    with pytest.raises(ActiveLearningConfigurationError):
        _ = initialize_classes_based_sampling(strategy_config=strategy_config)
