from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.active_learning.samplers import close_to_threshold
from inference.core.active_learning.samplers.close_to_threshold import (
    class_to_be_excluded,
    count_detections_close_to_threshold,
    detections_are_close_to_threshold,
    initialize_close_to_threshold_sampling,
    is_close_to_threshold,
    is_prediction_a_stub,
    multi_class_classification_prediction_is_close_to_threshold,
    multi_label_classification_prediction_is_close_to_threshold,
    prediction_is_close_to_threshold,
    sample_close_to_threshold,
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
INSTANCE_SEGMENTATION_PREDICTION = {
    "predictions": [
        {
            "x": 784.5,
            "y": 397.5,
            "width": 187.0,
            "height": 309.0,
            "confidence": 0.9,
            "class": "a",
            "class_id": 1,
            "points": [{"x": 207.0453125, "y": 106.559375}],
        },
        {
            "x": 784.5,
            "y": 397.5,
            "width": 187.0,
            "height": 309.0,
            "confidence": 0.7,
            "class": "b",
            "class_id": 1,
            "points": [{"x": 207.0453125, "y": 106.559375}],
        },
    ]
}
KEYPOINTS_PREDICTION = {
    "predictions": [
        {
            "x": 784.5,
            "y": 397.5,
            "width": 187.0,
            "height": 309.0,
            "confidence": 0.9,
            "class": "a",
            "class_id": 1,
            "keypoints": [
                {
                    "x": 207.0453125,
                    "y": 106.559375,
                    "confidence": 0.92,
                    "class_id": 1,
                    "class": "eye",
                }
            ],
        },
        {
            "x": 784.5,
            "y": 397.5,
            "width": 187.0,
            "height": 309.0,
            "confidence": 0.7,
            "class": "b",
            "class_id": 1,
            "keypoints": [
                {
                    "x": 207.0453125,
                    "y": 106.559375,
                    "confidence": 0.92,
                    "class_id": 1,
                    "class": "eye",
                }
            ],
        },
    ]
}

MULTI_LABEL_CLASSIFICATION_PREDICTION = {
    "image": {"width": 416, "height": 416},
    "predictions": {
        "cat": {"confidence": 0.97},
        "dog": {"confidence": 0.03},
    },
    "predicted_classes": ["cat"],
}

MULTI_CLASS_CLASSIFICATION_PREDICTION = {
    "image": {"width": 3487, "height": 2039},
    "predictions": [
        {"class": "Ambulance", "class_id": 0, "confidence": 0.6},
        {"class": "Limousine", "class_id": 16, "confidence": 0.3},
        {"class": "Helicopter", "class_id": 15, "confidence": 0.1},
    ],
    "top": "Ambulance",
    "confidence": 0.6,
}


@pytest.mark.parametrize(
    "value, threshold, epsilon",
    [
        (0.45, 0.5, 0.06),
        (0.55, 0.5, 0.06),
    ],
)
def test_close_to_threshold_when_value_is_close(
    value: float, threshold: float, epsilon: float
) -> None:
    # when
    result = is_close_to_threshold(value=value, threshold=threshold, epsilon=epsilon)

    # then
    assert result is True


@pytest.mark.parametrize(
    "value, threshold, epsilon",
    [
        (0.44, 0.5, 0.05),
        (0.56, 0.5, 0.05),
    ],
)
def test_close_to_threshold_when_value_is_not_close(
    value: float, threshold: float, epsilon: float
) -> None:
    # when
    result = is_close_to_threshold(value=value, threshold=threshold, epsilon=epsilon)

    # then
    assert result is False


def test_class_to_be_excluded_when_classes_not_selected() -> None:
    # when
    result = class_to_be_excluded(class_name="some", selected_class_names=None)

    # then
    assert result is False


def test_class_to_be_excluded_when_classes_selected_and_specific_class_matches() -> (
    None
):
    # when
    result = class_to_be_excluded(class_name="a", selected_class_names={"a", "b", "c"})

    # then
    assert result is False


def test_class_to_be_excluded_when_classes_selected_and_specific_class_does_not_matche() -> (
    None
):
    # when
    result = class_to_be_excluded(class_name="d", selected_class_names={"a", "b", "c"})

    # then
    assert result is True


def test_count_detections_close_to_threshold_when_no_detections_in_prediction() -> None:
    # given
    prediction = {"predictions": []}

    # when
    result = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names=None,
        threshold=0.5,
        epsilon=1.0,
    )

    # then
    assert result == 0


@pytest.mark.parametrize(
    "prediction",
    [
        OBJECT_DETECTION_PREDICTION,
        INSTANCE_SEGMENTATION_PREDICTION,
        KEYPOINTS_PREDICTION,
    ],
)
def test_count_detections_close_to_threshold_when_no_selected_class_names_pointed(
    prediction: dict,
) -> None:
    # when
    result = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names=None,
        threshold=0.5,
        epsilon=1.0,
    )

    # then
    assert result == 2


@pytest.mark.parametrize(
    "prediction",
    [
        OBJECT_DETECTION_PREDICTION,
        INSTANCE_SEGMENTATION_PREDICTION,
        KEYPOINTS_PREDICTION,
    ],
)
def test_count_detections_close_to_threshold_when_no_selected_class_names_filter_out_predictions(
    prediction: dict,
) -> None:
    # when
    result = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names={"a", "c"},
        threshold=0.5,
        epsilon=1.0,
    )

    # then
    assert result == 1


@pytest.mark.parametrize(
    "prediction",
    [
        OBJECT_DETECTION_PREDICTION,
        INSTANCE_SEGMENTATION_PREDICTION,
        KEYPOINTS_PREDICTION,
    ],
)
def test_count_detections_close_to_threshold_when_no_selected_threshold_filter_out_predictions(
    prediction: dict,
) -> None:
    # when
    result = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names=None,
        threshold=0.6,
        epsilon=0.15,
    )

    # then
    assert result == 1


@pytest.mark.parametrize(
    "prediction",
    [
        OBJECT_DETECTION_PREDICTION,
        INSTANCE_SEGMENTATION_PREDICTION,
        KEYPOINTS_PREDICTION,
    ],
)
def test_detection_prediction_is_close_to_threshold_when_minimum_objects_criterion_met(
    prediction: dict,
) -> None:
    # when
    result = detections_are_close_to_threshold(
        prediction=prediction,
        selected_class_names=None,
        threshold=0.6,
        epsilon=0.15,
        minimum_objects_close_to_threshold=1,
    )

    # then
    assert result is True


@pytest.mark.parametrize(
    "prediction",
    [
        OBJECT_DETECTION_PREDICTION,
        INSTANCE_SEGMENTATION_PREDICTION,
        KEYPOINTS_PREDICTION,
    ],
)
def test_detection_prediction_is_close_to_threshold_when_minimum_objects_criterion_not_met(
    prediction: dict,
) -> None:
    # when
    result = detections_are_close_to_threshold(
        prediction=prediction,
        selected_class_names=None,
        threshold=0.6,
        epsilon=0.15,
        minimum_objects_close_to_threshold=2,
    )

    # then
    assert result is False


def test_multi_label_classification_prediction_is_close_to_threshold_when_top_class_meet_criteria() -> (
    None
):
    # when
    result = multi_label_classification_prediction_is_close_to_threshold(
        prediction=MULTI_LABEL_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.95,
        epsilon=0.05,
        only_top_classes=True,
    )

    # then
    assert result is True


def test_multi_label_classification_prediction_is_close_to_threshold_when_non_top_class_meet_threshold_but_filtered_out_by_top_classes() -> (
    None
):
    # when
    result = multi_label_classification_prediction_is_close_to_threshold(
        prediction=MULTI_LABEL_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.05,
        epsilon=0.05,
        only_top_classes=True,
    )

    # then
    assert result is False


def test_multi_label_classification_prediction_is_close_to_threshold_when_non_top_class_meet_threshold_but_filtered_out_by_class_names() -> (
    None
):
    # when
    result = multi_label_classification_prediction_is_close_to_threshold(
        prediction=MULTI_LABEL_CLASSIFICATION_PREDICTION,
        selected_class_names={"cat", "tiger"},
        threshold=0.05,
        epsilon=0.05,
        only_top_classes=False,
    )

    # then
    assert result is False


def test_multi_label_classification_prediction_is_close_to_threshold_when_non_top_class_meet_criteria() -> (
    None
):
    # when
    result = multi_label_classification_prediction_is_close_to_threshold(
        prediction=MULTI_LABEL_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.05,
        epsilon=0.05,
        only_top_classes=False,
    )

    # then
    assert result is True


def test_multi_class_classification_prediction_is_close_to_threshold_for_top_class_when_classes_are_not_selected_and_threshold_met() -> (
    None
):
    # when
    result = multi_class_classification_prediction_is_close_to_threshold(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.6,
        epsilon=0.1,
        only_top_classes=True,
    )

    # then
    assert result is True


def test_multi_class_classification_prediction_is_close_to_threshold_for_top_class_when_classes_are_not_selected_and_threshold_not_met() -> (
    None
):
    # when
    result = multi_class_classification_prediction_is_close_to_threshold(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.8,
        epsilon=0.1,
        only_top_classes=True,
    )

    # then
    assert result is False


def test_multi_class_classification_prediction_is_close_to_threshold_for_top_class_when_classes_are_selected_and_top_class_does_not_match() -> (
    None
):
    # when
    result = multi_class_classification_prediction_is_close_to_threshold(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names={"Limousine", "Helicopter"},
        threshold=0.6,
        epsilon=0.1,
        only_top_classes=True,
    )

    # then
    assert result is False


def test_multi_class_classification_prediction_is_close_to_threshold_for_top_class_when_classes_are_selected_and_top_class_matches() -> (
    None
):
    # when
    result = multi_class_classification_prediction_is_close_to_threshold(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names={"Ambulance"},
        threshold=0.6,
        epsilon=0.1,
        only_top_classes=True,
    )

    # then
    assert result is True


def test_multi_class_classification_prediction_is_close_to_threshold_not_only_for_top_class_when_classes_not_selected() -> (
    None
):
    # when
    result = multi_class_classification_prediction_is_close_to_threshold(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.3,
        epsilon=0.1,
        only_top_classes=False,
    )

    # then
    assert result is True


def test_multi_class_classification_prediction_is_close_to_threshold_not_only_for_top_class_when_classes_not_selected_and_no_match_expected() -> (
    None
):
    # when
    result = multi_class_classification_prediction_is_close_to_threshold(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.5,
        epsilon=0.05,
        only_top_classes=False,
    )

    # then
    assert result is False


def test_multi_class_classification_prediction_is_close_to_threshold_not_only_for_top_class_when_classes_are_selected() -> (
    None
):
    # when
    result = multi_class_classification_prediction_is_close_to_threshold(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names={"Ambulance", "Helicopter"},
        threshold=0.3,
        epsilon=0.1,
        only_top_classes=False,
    )

    # then
    assert result is False


def test_prediction_is_close_to_threshold_for_detection_prediction() -> None:
    # when
    result = prediction_is_close_to_threshold(
        prediction=OBJECT_DETECTION_PREDICTION,
        prediction_type=OBJECT_DETECTION_TASK,
        selected_class_names=None,
        threshold=0.9,
        epsilon=0.05,
        only_top_classes=False,
        minimum_objects_close_to_threshold=1,
    )

    # then
    assert result is True


def test_prediction_is_close_to_threshold_for_classification_prediction() -> None:
    # when
    result = prediction_is_close_to_threshold(
        prediction=MULTI_LABEL_CLASSIFICATION_PREDICTION,
        prediction_type=CLASSIFICATION_TASK,
        selected_class_names=None,
        threshold=0.05,
        epsilon=0.05,
        only_top_classes=False,
        minimum_objects_close_to_threshold=1,
    )

    # then
    assert result is True


@pytest.mark.parametrize(
    "prediction",
    [
        OBJECT_DETECTION_PREDICTION,
        KEYPOINTS_PREDICTION,
        INSTANCE_SEGMENTATION_PREDICTION,
        MULTI_CLASS_CLASSIFICATION_PREDICTION,
        MULTI_LABEL_CLASSIFICATION_PREDICTION,
    ],
)
def test_is_prediction_a_stub_when_prediction_is_not_a_stub(prediction: dict) -> None:
    # when
    result = is_prediction_a_stub(prediction=prediction)

    # then
    assert result is False


def test_is_prediction_a_stub_when_prediction_is_a_stub() -> None:
    # given
    prediction = {"is_stub": True}

    # when
    result = is_prediction_a_stub(prediction=prediction)

    # then
    assert result is True


def test_sample_close_to_threshold_when_prediction_is_sub() -> None:
    # when
    result = sample_close_to_threshold(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction={"is_stub": True},
        prediction_type=CLASSIFICATION_TASK,
        selected_class_names=None,
        threshold=0.5,
        epsilon=0.1,
        only_top_classes=True,
        minimum_objects_close_to_threshold=1,
        probability=1.0,
    )

    # then
    assert result is False


def test_sample_close_to_threshold_when_prediction_type_is_unknown() -> None:
    # when
    result = sample_close_to_threshold(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction={"is_stub": True},
        prediction_type="unknown",
        selected_class_names=None,
        threshold=0.5,
        epsilon=0.1,
        only_top_classes=True,
        minimum_objects_close_to_threshold=1,
        probability=1.0,
    )

    # then
    assert result is False


def test_sample_close_to_threshold_when_prediction_is_not_close_to_threshold() -> None:
    # when
    result = sample_close_to_threshold(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        prediction_type=CLASSIFICATION_TASK,
        selected_class_names={"Ambulance"},
        threshold=0.8,
        epsilon=0.1,
        only_top_classes=True,
        minimum_objects_close_to_threshold=1,
        probability=1.0,
    )

    # then
    assert result is False


@mock.patch.object(close_to_threshold.random, "random")
def test_sample_close_to_threshold_when_prediction_is_close_to_threshold(
    random_mock: MagicMock,
) -> None:
    # given
    random_mock.return_value = 0.49

    # when
    result = sample_close_to_threshold(
        image=np.zeros((128, 128, 3), dtype=np.uint8),
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        prediction_type=CLASSIFICATION_TASK,
        selected_class_names={"Ambulance"},
        threshold=0.6,
        epsilon=0.1,
        only_top_classes=True,
        minimum_objects_close_to_threshold=1,
        probability=0.5,
    )

    # then
    assert result is True


def test_initialize_close_to_threshold_sampling() -> None:
    # given
    strategy_config = {
        "name": "ambulance_high_confidence",
        "selected_class_names": ["Ambulance"],
        "threshold": 0.75,
        "epsilon": 0.25,
        "probability": 1.0,
    }

    # when
    sampling_method = initialize_close_to_threshold_sampling(
        strategy_config=strategy_config
    )
    result = sampling_method.sample(
        np.zeros((128, 128, 3), dtype=np.uint8),
        MULTI_CLASS_CLASSIFICATION_PREDICTION,
        CLASSIFICATION_TASK,
    )

    # then
    assert result is True
    assert sampling_method.name == "ambulance_high_confidence"


@mock.patch.object(close_to_threshold, "partial")
def test_initialize_close_to_threshold_sampling_when_classes_not_selected(
    partial_mock: MagicMock,
) -> None:
    # given
    strategy_config = {
        "name": "ambulance_high_confidence",
        "threshold": 0.75,
        "epsilon": 0.25,
        "probability": 0.6,
    }

    # when
    _ = initialize_close_to_threshold_sampling(strategy_config=strategy_config)

    # then
    partial_mock.assert_called_once_with(
        sample_close_to_threshold,
        selected_class_names=None,
        threshold=0.75,
        epsilon=0.25,
        only_top_classes=True,
        minimum_objects_close_to_threshold=1,
        probability=0.6,
    )


@mock.patch.object(close_to_threshold, "partial")
def test_initialize_close_to_threshold_sampling_when_classes_selected(
    partial_mock: MagicMock,
) -> None:
    # given
    strategy_config = {
        "name": "ambulance_high_confidence",
        "selected_class_names": ["Ambulance", "Helicopter"],
        "threshold": 0.75,
        "epsilon": 0.25,
        "probability": 0.6,
    }

    # when
    _ = initialize_close_to_threshold_sampling(strategy_config=strategy_config)

    # then
    partial_mock.assert_called_once_with(
        sample_close_to_threshold,
        selected_class_names={"Ambulance", "Helicopter"},
        threshold=0.75,
        epsilon=0.25,
        only_top_classes=True,
        minimum_objects_close_to_threshold=1,
        probability=0.6,
    )


@mock.patch.object(close_to_threshold, "partial")
def test_initialize_close_to_threshold_sampling_when_only_top_classes_mode_enabled(
    partial_mock: MagicMock,
) -> None:
    # given
    strategy_config = {
        "name": "ambulance_high_confidence",
        "selected_class_names": ["Ambulance"],
        "threshold": 0.75,
        "epsilon": 0.25,
        "probability": 0.6,
        "only_top_classes": False,
    }

    # when
    _ = initialize_close_to_threshold_sampling(strategy_config=strategy_config)

    # then
    partial_mock.assert_called_once_with(
        sample_close_to_threshold,
        selected_class_names={"Ambulance"},
        threshold=0.75,
        epsilon=0.25,
        only_top_classes=False,
        minimum_objects_close_to_threshold=1,
        probability=0.6,
    )


@mock.patch.object(close_to_threshold, "partial")
def test_initialize_close_to_threshold_sampling_when_objects_close_to_threshold_specified(
    partial_mock: MagicMock,
) -> None:
    # given
    strategy_config = {
        "name": "ambulance_high_confidence",
        "selected_class_names": ["Ambulance"],
        "threshold": 0.75,
        "epsilon": 0.25,
        "probability": 0.6,
        "minimum_objects_close_to_threshold": 6,
    }

    # when
    _ = initialize_close_to_threshold_sampling(strategy_config=strategy_config)

    # then
    partial_mock.assert_called_once_with(
        sample_close_to_threshold,
        selected_class_names={"Ambulance"},
        threshold=0.75,
        epsilon=0.25,
        only_top_classes=True,
        minimum_objects_close_to_threshold=6,
        probability=0.6,
    )


def test_initialize_close_to_threshold_sampling_when_configuration_key_missing() -> (
    None
):
    # given
    strategy_config = {
        "name": "ambulance_high_confidence",
        "threshold": 0.75,
        "epsilon": 0.25,
    }

    # when
    with pytest.raises(ActiveLearningConfigurationError):
        _ = initialize_close_to_threshold_sampling(strategy_config=strategy_config)
