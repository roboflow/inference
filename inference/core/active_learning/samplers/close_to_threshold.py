import random
from functools import partial
from typing import Any, Dict, Optional, Set

import numpy as np

from inference.core.active_learning.entities import (
    Prediction,
    PredictionType,
    SamplingMethod,
)
from inference.core.constants import (
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    OBJECT_DETECTION_TASK,
)
from inference.core.exceptions import ActiveLearningConfigurationError

ELIGIBLE_PREDICTION_TYPES = {
    CLASSIFICATION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    OBJECT_DETECTION_TASK,
}


def initialize_close_to_threshold_sampling(
    strategy_config: Dict[str, Any]
) -> SamplingMethod:
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
        return SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `close_to_threshold_sampling` missing key detected: {error}."
        ) from error


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
    if CLASSIFICATION_TASK not in prediction_type:
        return detections_are_close_to_threshold(
            prediction=prediction,
            selected_class_names=selected_class_names,
            threshold=threshold,
            epsilon=epsilon,
            minimum_objects_close_to_threshold=minimum_objects_close_to_threshold,
        )
    checker = multi_label_classification_prediction_is_close_to_threshold
    if "top" in prediction:
        checker = multi_class_classification_prediction_is_close_to_threshold
    return checker(
        prediction=prediction,
        selected_class_names=selected_class_names,
        threshold=threshold,
        epsilon=epsilon,
        only_top_classes=only_top_classes,
    )


def multi_class_classification_prediction_is_close_to_threshold(
    prediction: Prediction,
    selected_class_names: Optional[Set[str]],
    threshold: float,
    epsilon: float,
    only_top_classes: bool,
) -> bool:
    if only_top_classes:
        return (
            multi_class_classification_prediction_is_close_to_threshold_for_top_class(
                prediction=prediction,
                selected_class_names=selected_class_names,
                threshold=threshold,
                epsilon=epsilon,
            )
        )
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
    return selected_class_names is not None and class_name not in selected_class_names


def is_close_to_threshold(value: float, threshold: float, epsilon: float) -> bool:
    return abs(value - threshold) < epsilon
