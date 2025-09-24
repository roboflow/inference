from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.active_learning.samplers import number_of_detections
from inference.core.active_learning.samplers.number_of_detections import (
    initialize_detections_number_based_sampling,
    is_in_range,
    sample_based_on_detections_number,
)
from inference.core.constants import CLASSIFICATION_TASK, OBJECT_DETECTION_TASK
from inference.core.exceptions import ActiveLearningConfigurationError

OBJECT_DETECTION_PREDICTION = {
    "predictions": [
        {
            "x": 784.5,
            "y": 397.5,
            "width": 187.0,
            "height": 309.0,
            "confidence": 0.9,
            "class": "a",
            "class_id": 1,
        },
        {
            "x": 784.5,
            "y": 397.5,
            "width": 187.0,
            "height": 309.0,
            "confidence": 0.7,
            "class": "b",
            "class_id": 1,
        },
    ]
}


@pytest.mark.parametrize(
    "value, more_than, less_than",
    [
        (1, None, 2),
        (2, 1, 5),
        (5, 4, None),
        (1, None, None),
    ],
)
def test_is_in_range_value_meets_condition(
    value: int, more_than: Optional[int], less_than: Optional[int]
) -> None:
    # when
    result = is_in_range(value=value, more_than=more_than, less_than=less_than)

    # then
    assert result is True


@pytest.mark.parametrize(
    "value, more_than, less_than",
    [
        (1, 2, None),
        (2, 5, 1),
        (5, None, 4),
    ],
)
def test_is_in_range_value_does_not_meet_condition(
    value: int, more_than: Optional[int], less_than: Optional[int]
) -> None:
    # when
    result = is_in_range(value=value, more_than=more_than, less_than=less_than)

    # then
    assert result is False


def test_sample_based_on_detections_number_when_classification_prediction_given() -> (
    None
):
    # given
    prediction = {
        "image": {"width": 416, "height": 416},
        "predictions": {
            "cat": {"confidence": 0.97},
            "dog": {"confidence": 0.03},
        },
        "predicted_classes": ["cat"],
    }

    # when
    result = sample_based_on_detections_number(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=prediction,
        prediction_type=CLASSIFICATION_TASK,
        less_than=None,
        more_than=None,
        selected_class_names=None,
        probability=1.0,
    )

    # then
    assert result is False


def test_sample_based_on_detections_number_when_stub_prediction_given() -> None:
    # when
    result = sample_based_on_detections_number(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction={"is_stub": True},
        prediction_type=CLASSIFICATION_TASK,
        less_than=None,
        more_than=None,
        selected_class_names=None,
        probability=1.0,
    )

    # then
    assert result is False


@mock.patch.object(number_of_detections.random, "random")
def test_sample_based_on_detections_number_when_detections_in_range_and_sampling_succeeds(
    random_mock: MagicMock,
) -> None:
    # given
    random_mock.return_value = 0.29

    # when
    result = sample_based_on_detections_number(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=OBJECT_DETECTION_PREDICTION,
        prediction_type=OBJECT_DETECTION_TASK,
        more_than=1,
        less_than=3,
        selected_class_names={"a", "b"},
        probability=0.3,
    )

    # then
    assert result is True


@mock.patch.object(number_of_detections.random, "random")
def test_sample_based_on_detections_number_when_detections_in_range_and_sampling_fails(
    random_mock: MagicMock,
) -> None:
    # given
    random_mock.return_value = 0.31

    # when
    result = sample_based_on_detections_number(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=OBJECT_DETECTION_PREDICTION,
        prediction_type=OBJECT_DETECTION_TASK,
        more_than=1,
        less_than=3,
        selected_class_names={"a", "b"},
        probability=0.3,
    )

    # then
    assert result is False


def test_sample_based_on_detections_number_when_detections_not_in_range() -> None:
    # when
    result = sample_based_on_detections_number(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=OBJECT_DETECTION_PREDICTION,
        prediction_type=OBJECT_DETECTION_TASK,
        more_than=2,
        less_than=3,
        selected_class_names={"a"},
        probability=0.3,
    )

    # then
    assert result is False


def test_initialize_detections_number_based_sampling() -> None:
    # given
    strategy_config = {
        "name": "two_detections",
        "less_than": 3,
        "more_than": 1,
        "selected_class_names": {"a", "b"},
        "probability": 1.0,
    }

    # when
    sampling_method = initialize_detections_number_based_sampling(
        strategy_config=strategy_config
    )
    result = sampling_method.sample(
        np.zeros((128, 128, 3), dtype=np.uint8),
        OBJECT_DETECTION_PREDICTION,
        OBJECT_DETECTION_TASK,
    )

    # then
    assert result is True
    assert sampling_method.name == "two_detections"


@mock.patch.object(number_of_detections, "partial")
def test_test_initialize_detections_number_based_sampling_when_optional_values_not_given(
    partial_mock: MagicMock,
) -> None:
    # given
    strategy_config = {
        "name": "two_detections",
        "probability": 1.0,
    }

    # when
    _ = initialize_detections_number_based_sampling(strategy_config=strategy_config)

    # then
    partial_mock.assert_called_once_with(
        sample_based_on_detections_number,
        less_than=None,
        more_than=None,
        selected_class_names=None,
        probability=1.0,
    )


@mock.patch.object(number_of_detections, "partial")
def test_test_initialize_detections_number_based_sampling_when_optional_values_given(
    partial_mock: MagicMock,
) -> None:
    # given
    strategy_config = {
        "name": "two_detections",
        "probability": 1.0,
        "less_than": 10,
        "more_than": 5,
        "selected_class_names": ["a", "b"],
    }

    # when
    _ = initialize_detections_number_based_sampling(strategy_config=strategy_config)

    # then
    partial_mock.assert_called_once_with(
        sample_based_on_detections_number,
        less_than=10,
        more_than=5,
        selected_class_names={"a", "b"},
        probability=1.0,
    )


def test_initialize_detections_number_based_sampling_when_required_value_missing() -> (
    None
):
    # given
    strategy_config = {
        "name": "two_detections",
        "less_than": 10,
        "more_than": 5,
        "selected_class_names": ["a", "b"],
    }

    # when
    with pytest.raises(ActiveLearningConfigurationError):
        _ = initialize_detections_number_based_sampling(strategy_config=strategy_config)


def test_initialize_detections_number_based_sampling_when_malformed_config_given() -> (
    None
):
    # given
    strategy_config = {
        "name": "two_detections",
        "less_than": 5,
        "more_than": 6,
        "probability": 1.0,
        "selected_class_names": ["a", "b"],
    }

    # when
    with pytest.raises(ActiveLearningConfigurationError):
        _ = initialize_detections_number_based_sampling(strategy_config=strategy_config)
