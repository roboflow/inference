import random
from functools import partial
from typing import Any, Dict, Optional, Set

import numpy as np

from inference.core.active_learning.entities import (
    Prediction,
    PredictionType,
    SamplingMethod,
)
from inference.core.active_learning.samplers.close_to_threshold import (
    count_detections_close_to_threshold,
    is_prediction_a_stub,
)
from inference.core.constants import (
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    OBJECT_DETECTION_TASK,
)
from inference.core.exceptions import ActiveLearningConfigurationError

ELIGIBLE_PREDICTION_TYPES = {
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINTS_DETECTION_TASK,
    OBJECT_DETECTION_TASK,
}


def initialize_detections_number_based_sampling(
    strategy_config: Dict[str, Any],
) -> SamplingMethod:
    try:
        more_than = strategy_config.get("more_than")
        less_than = strategy_config.get("less_than")
        ensure_range_configuration_is_valid(more_than=more_than, less_than=less_than)
        selected_class_names = strategy_config.get("selected_class_names")
        if selected_class_names is not None:
            selected_class_names = set(selected_class_names)
        sample_function = partial(
            sample_based_on_detections_number,
            less_than=less_than,
            more_than=more_than,
            selected_class_names=selected_class_names,
            probability=strategy_config["probability"],
        )
        return SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `detections_number_based_sampling` missing key detected: {error}."
        ) from error


def sample_based_on_detections_number(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    more_than: Optional[int],
    less_than: Optional[int],
    selected_class_names: Optional[Set[str]],
    probability: float,
) -> bool:
    if is_prediction_a_stub(prediction=prediction):
        return False
    if prediction_type not in ELIGIBLE_PREDICTION_TYPES:
        return False
    detections_close_to_threshold = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names=selected_class_names,
        threshold=0.5,
        epsilon=1.0,
    )
    if is_in_range(
        value=detections_close_to_threshold, less_than=less_than, more_than=more_than
    ):
        return random.random() < probability
    return False


def is_in_range(
    value: int,
    more_than: Optional[int],
    less_than: Optional[int],
) -> bool:
    # calculates value > more_than and value < less_than, with optional borders of range
    less_than_satisfied, more_than_satisfied = less_than is None, more_than is None
    if less_than is not None and value < less_than:
        less_than_satisfied = True
    if more_than is not None and value > more_than:
        more_than_satisfied = True
    return less_than_satisfied and more_than_satisfied


def ensure_range_configuration_is_valid(
    more_than: Optional[int],
    less_than: Optional[int],
) -> None:
    if more_than is None or less_than is None:
        return None
    if more_than >= less_than:
        raise ActiveLearningConfigurationError(
            f"Misconfiguration of detections number sampling: "
            f"`more_than` parameter ({more_than}) >= `less_than` ({less_than})."
        )
