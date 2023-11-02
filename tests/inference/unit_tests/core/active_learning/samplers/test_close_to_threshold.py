import pytest

from inference.core.active_learning.samplers.close_to_threshold import (
    close_to_threshold,
    count_detections_close_to_threshold,
    detection_prediction_is_close_to_threshold,
    multi_label_classification_prediction_is_close_to_threshold,
    multi_class_classification_prediction_is_close_to_threshold_for_top_class,
)

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
    result = close_to_threshold(value=value, threshold=threshold, epsilon=epsilon)

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
    result = close_to_threshold(value=value, threshold=threshold, epsilon=epsilon)

    # then
    assert result is False


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
    result = detection_prediction_is_close_to_threshold(
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
    result = detection_prediction_is_close_to_threshold(
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
    result = multi_class_classification_prediction_is_close_to_threshold_for_top_class(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.6,
        epsilon=0.1,
    )

    # then
    assert result is True


def test_multi_class_classification_prediction_is_close_to_threshold_for_top_class_when_classes_are_not_selected_and_threshold_not_met() -> (
    None
):
    # when
    result = multi_class_classification_prediction_is_close_to_threshold_for_top_class(
        prediction=MULTI_CLASS_CLASSIFICATION_PREDICTION,
        selected_class_names=None,
        threshold=0.8,
        epsilon=0.1,
    )

    # then
    assert result is False
