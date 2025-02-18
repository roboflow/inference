from functools import partial
from typing import Any, Dict, Set

import numpy as np

from inference.core.active_learning.entities import (
    Prediction,
    PredictionType,
    SamplingMethod,
)
from inference.core.active_learning.samplers.close_to_threshold import (
    sample_close_to_threshold,
)
from inference.core.constants import CLASSIFICATION_TASK
from inference.core.exceptions import ActiveLearningConfigurationError

ELIGIBLE_PREDICTION_TYPES = {CLASSIFICATION_TASK}


def initialize_classes_based_sampling(
    strategy_config: Dict[str, Any],
) -> SamplingMethod:
    try:
        sample_function = partial(
            sample_based_on_classes,
            selected_class_names=set(strategy_config["selected_class_names"]),
            probability=strategy_config["probability"],
        )
        return SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `classes_based_sampling` missing key detected: {error}."
        ) from error


def sample_based_on_classes(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    selected_class_names: Set[str],
    probability: float,
) -> bool:
    if prediction_type not in ELIGIBLE_PREDICTION_TYPES:
        return False
    return sample_close_to_threshold(
        image=image,
        prediction=prediction,
        prediction_type=prediction_type,
        selected_class_names=selected_class_names,
        threshold=0.5,
        epsilon=1.0,
        only_top_classes=True,
        minimum_objects_close_to_threshold=1,
        probability=probability,
    )
