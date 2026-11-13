"""Sampling of predictions whose confidence is close to a threshold."""

import random
from functools import partial
from typing import Any, Dict, Optional, Set

import numpy as np

from inference_server.active_learning.entities import (
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    MULTI_LABEL_CLASSIFICATION_TASK,
    OBJECT_DETECTION_TASK,
    ActiveLearningConfigurationError,
    Prediction,
    PredictionType,
    SamplingMethod,
)

ELIGIBLE_PREDICTION_TYPES = {
    CLASSIFICATION_TASK,
    MULTI_LABEL_CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    OBJECT_DETECTION_TASK,
}


def initialize_close_to_threshold_sampling(
    strategy_config: Dict[str, Any],
) -> SamplingMethod:
    """Build the close-to-threshold sampling method of a strategy.

    Args:
        strategy_config: Strategy configuration with ``name``, ``threshold``,
            ``epsilon``, ``probability`` and the optional ``selected_class_names``,
            ``only_top_classes`` and ``minimum_objects_close_to_threshold``.

    Returns:
        The sampling method.

    Raises:
        ActiveLearningConfigurationError: If a required key is missing.
    """
    try:
        selected_class_names = strategy_config.get("selected_class_names")
        if selected_class_names is not None:
            selected_class_names = set(selected_class_names)
        sample_function = partial(
            sample_close_to_threshold,
            selected_class_names=selected_class_names,
            threshold=strategy_config["threshold"],
            epsilon=strategy_config["epsilon"],
            only_top_classes=strategy_config.get("only_top_classes", True),
            minimum_objects_close_to_threshold=strategy_config.get(
                "minimum_objects_close_to_threshold",
                1,
            ),
            probability=strategy_config["probability"],
        )
        sampling_method = SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `close_to_threshold_sampling` missing key detected: {error}."
        ) from error

    return sampling_method


def sample_close_to_threshold(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
    only_top_classes: bool,
    minimum_objects_close_to_threshold: int,
    probability: float,
) -> bool:
    """Select a datapoint whose prediction is close to the threshold.

    Args:
        image: Image the prediction was made for; not read.
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.
        selected_class_names: Classes taken into account, all when None.
        threshold: Confidence the prediction is compared with.
        epsilon: Largest distance from the threshold that still counts as close.
        only_top_classes: For classification, look only at the predicted classes.
        minimum_objects_close_to_threshold: For detections, the number of close
            detections required.
        probability: Probability of selecting a datapoint that qualifies.

    Returns:
        True when the datapoint is selected.
    """
    if is_prediction_a_stub(prediction=prediction):
        return False

    if prediction_type not in ELIGIBLE_PREDICTION_TYPES:
        return False

    close_to_threshold = prediction_is_close_to_threshold(
        prediction=prediction,
        prediction_type=prediction_type,
        selected_class_names=selected_class_names,
        threshold=threshold,
        epsilon=epsilon,
        only_top_classes=only_top_classes,
        minimum_objects_close_to_threshold=minimum_objects_close_to_threshold,
    )
    if not close_to_threshold:
        return False

    return random.random() < probability


def is_prediction_a_stub(prediction: Prediction) -> bool:
    """Tell whether a prediction comes from a stub model.

    Args:
        prediction: Prediction in the response format of the server.

    Returns:
        The ``is_stub`` marker of the prediction, False when absent.
    """
    return prediction.get("is_stub", False)


def prediction_is_close_to_threshold(
    prediction: Prediction,
    prediction_type: PredictionType,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
    only_top_classes: bool,
    minimum_objects_close_to_threshold: int,
) -> bool:
    """Apply the closeness rule of the task type to a prediction.

    Args:
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.
        selected_class_names: Classes taken into account, all when None.
        threshold: Confidence the prediction is compared with.
        epsilon: Largest distance from the threshold that still counts as close.
        only_top_classes: For classification, look only at the predicted classes.
        minimum_objects_close_to_threshold: For detections, the number of close
            detections required.

    Returns:
        True when the prediction is close to the threshold.
    """
    if CLASSIFICATION_TASK not in prediction_type:
        detections_close = detections_are_close_to_threshold(
            prediction=prediction,
            selected_class_names=selected_class_names,
            threshold=threshold,
            epsilon=epsilon,
            minimum_objects_close_to_threshold=minimum_objects_close_to_threshold,
        )

        return detections_close

    checker = multi_label_classification_prediction_is_close_to_threshold
    if "top" in prediction:
        checker = multi_class_classification_prediction_is_close_to_threshold
    classification_close = checker(
        prediction=prediction,
        selected_class_names=selected_class_names,
        threshold=threshold,
        epsilon=epsilon,
        only_top_classes=only_top_classes,
    )

    return classification_close


def multi_class_classification_prediction_is_close_to_threshold(
    prediction: Prediction,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
    only_top_classes: bool,
) -> bool:
    """Apply the closeness rule to a multi-class classification prediction.

    Args:
        prediction: Prediction with ``top``, ``confidence`` and ``predictions``.
        selected_class_names: Classes taken into account, all when None.
        threshold: Confidence the prediction is compared with.
        epsilon: Largest distance from the threshold that still counts as close.
        only_top_classes: Look only at the top class.

    Returns:
        True when the prediction is close to the threshold.
    """
    if only_top_classes:
        top_class_close = (
            multi_class_classification_prediction_is_close_to_threshold_for_top_class(
                prediction=prediction,
                selected_class_names=selected_class_names,
                threshold=threshold,
                epsilon=epsilon,
            )
        )

        return top_class_close

    for prediction_details in prediction["predictions"]:
        if class_to_be_excluded(
            class_name=prediction_details["class"],
            selected_class_names=selected_class_names,
        ):
            continue
        if is_close_to_threshold(
            value=prediction_details["confidence"], threshold=threshold, epsilon=epsilon
        ):
            return True

    return False


def multi_class_classification_prediction_is_close_to_threshold_for_top_class(
    prediction: Prediction,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
) -> bool:
    """Apply the closeness rule to the top class of a prediction.

    Args:
        prediction: Prediction with ``top`` and ``confidence``.
        selected_class_names: Classes taken into account, all when None.
        threshold: Confidence the prediction is compared with.
        epsilon: Largest distance from the threshold that still counts as close.

    Returns:
        True when the top class is selected and close to the threshold.
    """
    if (
        selected_class_names is not None
        and prediction["top"] not in selected_class_names
    ):
        return False

    return abs(prediction["confidence"] - threshold) < epsilon


def multi_label_classification_prediction_is_close_to_threshold(
    prediction: Prediction,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
    only_top_classes: bool,
) -> bool:
    """Apply the closeness rule to a multi-label classification prediction.

    Args:
        prediction: Prediction with ``predicted_classes`` and a ``predictions``
            mapping of class name to confidence.
        selected_class_names: Classes taken into account, all when None.
        threshold: Confidence the prediction is compared with.
        epsilon: Largest distance from the threshold that still counts as close.
        only_top_classes: Look only at the predicted classes.

    Returns:
        True when any class taken into account is close to the threshold.
    """
    predicted_classes = set(prediction["predicted_classes"])
    for class_name, prediction_details in prediction["predictions"].items():
        if only_top_classes and class_name not in predicted_classes:
            continue
        if class_to_be_excluded(
            class_name=class_name, selected_class_names=selected_class_names
        ):
            continue
        if is_close_to_threshold(
            value=prediction_details["confidence"], threshold=threshold, epsilon=epsilon
        ):
            return True

    return False


def detections_are_close_to_threshold(
    prediction: Prediction,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
    minimum_objects_close_to_threshold: int,
) -> bool:
    """Apply the closeness rule to a detection prediction.

    Args:
        prediction: Prediction with a ``predictions`` list of detections.
        selected_class_names: Classes taken into account, all when None.
        threshold: Confidence the detections are compared with.
        epsilon: Largest distance from the threshold that still counts as close.
        minimum_objects_close_to_threshold: Number of close detections required.

    Returns:
        True when enough detections are close to the threshold.
    """
    detections_close_to_threshold = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names=selected_class_names,
        threshold=threshold,
        epsilon=epsilon,
    )

    return detections_close_to_threshold >= minimum_objects_close_to_threshold


def count_detections_close_to_threshold(
    prediction: Prediction,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
) -> int:
    """Count the detections close to the threshold.

    Args:
        prediction: Prediction with a ``predictions`` list of detections.
        selected_class_names: Classes taken into account, all when None.
        threshold: Confidence the detections are compared with.
        epsilon: Largest distance from the threshold that still counts as close.

    Returns:
        Number of detections of the selected classes close to the threshold.
    """
    counter = 0
    for prediction_details in prediction["predictions"]:
        if class_to_be_excluded(
            class_name=prediction_details["class"],
            selected_class_names=selected_class_names,
        ):
            continue
        if is_close_to_threshold(
            value=prediction_details["confidence"], threshold=threshold, epsilon=epsilon
        ):
            counter += 1

    return counter


def class_to_be_excluded(
    class_name: str, selected_class_names: Optional[Set[str]]
) -> bool:
    """Tell whether a class is outside the selected classes.

    Args:
        class_name: Class of a prediction.
        selected_class_names: Classes taken into account, all when None.

    Returns:
        True when the class is not taken into account.
    """
    return selected_class_names is not None and class_name not in selected_class_names


def is_close_to_threshold(value: float, threshold: float, epsilon: float) -> bool:
    """Tell whether a confidence is close to the threshold.

    Args:
        value: Confidence of a prediction.
        threshold: Confidence it is compared with.
        epsilon: Largest distance that still counts as close.

    Returns:
        True when the distance is below ``epsilon``.
    """
    return abs(value - threshold) < epsilon
