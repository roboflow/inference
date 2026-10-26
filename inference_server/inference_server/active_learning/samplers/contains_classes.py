"""Sampling of classification predictions of selected classes."""

from functools import partial
from typing import Any, Dict, Set

import numpy as np

from inference_server.active_learning.entities import (
    CLASSIFICATION_TASK,
    MULTI_LABEL_CLASSIFICATION_TASK,
    ActiveLearningConfigurationError,
    Prediction,
    PredictionType,
    SamplingMethod,
)
from inference_server.active_learning.samplers.close_to_threshold import (
    sample_close_to_threshold,
)

ELIGIBLE_PREDICTION_TYPES = {CLASSIFICATION_TASK, MULTI_LABEL_CLASSIFICATION_TASK}


def initialize_classes_based_sampling(
    strategy_config: Dict[str, Any],
) -> SamplingMethod:
    """Build the classes-based sampling method of a strategy.

    Args:
        strategy_config: Strategy configuration with ``name``,
            ``selected_class_names`` and ``probability``.

    Returns:
        The sampling method.

    Raises:
        ActiveLearningConfigurationError: If a required key is missing.
    """
    try:
        sample_function = partial(
            sample_based_on_classes,
            selected_class_names=set(strategy_config["selected_class_names"]),
            probability=strategy_config["probability"],
        )
        sampling_method = SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `classes_based_sampling` missing key detected: {error}."
        ) from error

    return sampling_method


def sample_based_on_classes(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    selected_class_names: Set[str],
    probability: float,
) -> bool:
    """Select a classification datapoint predicted as one of the classes.

    Args:
        image: Image the prediction was made for; not read.
        prediction: Prediction in the response format of the server.
        prediction_type: Task type of the model.
        selected_class_names: Classes that qualify a datapoint.
        probability: Probability of selecting a datapoint that qualifies.

    Returns:
        True when the datapoint is selected.
    """
    if prediction_type not in ELIGIBLE_PREDICTION_TYPES:
        return False

    sampling_result = sample_close_to_threshold(
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

    return sampling_result
