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
    sample_close_to_threshold,
)
from inference.core.constants import CLASSIFICATION_TASK
from inference.core.exceptions import ActiveLearningConfigurationError


def initialize_detections_number_based_sampling(
    strategy_config: Dict[str, Any]
) -> SamplingMethod:
    less_than_objects = strategy_config.get("less_than_objects")
    more_than_objects = strategy_config.get("more_than_objects")
    both_nones = less_than_objects is None and more_than_objects is None
    both_has_values = less_than_objects is not None and more_than_objects is not None
    if both_nones or both_has_values:
        raise ActiveLearningConfigurationError(
            f"Only one from `less_than_objects` and `more_than_objects` values must be set."
        )
    selected_class_names = strategy_config.get("selected_class_names")
    if selected_class_names is not None:
        selected_class_names = set(selected_class_names)
    sample_function = partial(
        sample_close_to_threshold,
        less_than_objects=less_than_objects,
        more_than_objects=more_than_objects,
        selected_class_names=selected_class_names,
        probability=strategy_config["probability"],
    )
    return SamplingMethod(
        name=strategy_config["name"],
        sample=sample_function,
    )


def sample_based_on_detections_number(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    less_than_objects: Optional[int],
    more_than_objects: Optional[int],
    selected_class_names: Optional[Set[str]],
    probability: float,
) -> bool:
    if CLASSIFICATION_TASK in prediction_type:
        return False
    if is_prediction_a_stub(prediction=prediction):
        return False
    detections_close_to_threshold = count_detections_close_to_threshold(
        prediction=prediction,
        selected_class_names=selected_class_names,
        threshold=0.5,
        epsilon=1.0,
    )
    if (
        less_than_objects is not None
        and detections_close_to_threshold >= less_than_objects
    ):
        return False
    if (
        more_than_objects is not None
        and detections_close_to_threshold <= more_than_objects
    ):
        return False
    return random.random() < probability
