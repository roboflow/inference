from unittest import mock

import numpy as np
import pytest

from inference_server.active_learning.samplers import close_to_threshold
from inference_server.active_learning.samplers.close_to_threshold import (
    sample_close_to_threshold,
)
from inference_server.active_learning.samplers.contains_classes import (
    sample_based_on_classes,
)
from inference_server.active_learning.samplers.number_of_detections import (
    sample_based_on_detections_number,
)

IMAGE = np.zeros((128, 128, 3), dtype=np.uint8)
MULTI_LABEL_PREDICTION = {
    "image": {"width": 416, "height": 416},
    "predictions": {
        "cat": {"confidence": 0.97},
        "dog": {"confidence": 0.03},
    },
    "predicted_classes": ["cat"],
}
TASK_TYPES_OF_A_MULTI_LABEL_MODEL = ["classification", "multi-label-classification"]


@pytest.fixture(autouse=True)
def always_drawn():
    with mock.patch.object(close_to_threshold.random, "random", return_value=0.0):
        yield


@pytest.mark.parametrize("prediction_type", TASK_TYPES_OF_A_MULTI_LABEL_MODEL)
@pytest.mark.parametrize(
    "threshold, only_top_classes, expected_result",
    [
        (0.95, True, True),
        (0.05, True, False),
        (0.05, False, True),
        (0.5, False, False),
    ],
)
def test_close_to_threshold_sampling_of_a_multi_label_prediction(
    prediction_type: str,
    threshold: float,
    only_top_classes: bool,
    expected_result: bool,
) -> None:
    result = sample_close_to_threshold(
        image=IMAGE,
        prediction=MULTI_LABEL_PREDICTION,
        prediction_type=prediction_type,
        selected_class_names=None,
        threshold=threshold,
        epsilon=0.05,
        only_top_classes=only_top_classes,
        minimum_objects_close_to_threshold=1,
        probability=1.0,
    )

    assert result is expected_result


@pytest.mark.parametrize("prediction_type", TASK_TYPES_OF_A_MULTI_LABEL_MODEL)
@pytest.mark.parametrize(
    "selected_class_names, expected_result",
    [({"cat"}, True), ({"dog"}, False), ({"cat", "dog"}, True)],
)
def test_classes_based_sampling_of_a_multi_label_prediction(
    prediction_type: str,
    selected_class_names: set,
    expected_result: bool,
) -> None:
    result = sample_based_on_classes(
        image=IMAGE,
        prediction=MULTI_LABEL_PREDICTION,
        prediction_type=prediction_type,
        selected_class_names=selected_class_names,
        probability=1.0,
    )

    assert result is expected_result


@pytest.mark.parametrize("prediction_type", TASK_TYPES_OF_A_MULTI_LABEL_MODEL)
def test_detections_number_sampling_skips_a_multi_label_prediction(
    prediction_type: str,
) -> None:
    result = sample_based_on_detections_number(
        image=IMAGE,
        prediction=MULTI_LABEL_PREDICTION,
        prediction_type=prediction_type,
        more_than=None,
        less_than=None,
        selected_class_names=None,
        probability=1.0,
    )

    assert result is False
