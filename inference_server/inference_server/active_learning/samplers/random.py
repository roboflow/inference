"""Sampling of a share of all datapoints."""

import random
from functools import partial
from typing import Any, Dict

import numpy as np

from inference_server.active_learning.entities import (
    ActiveLearningConfigurationError,
    Prediction,
    PredictionType,
    SamplingMethod,
)


def initialize_random_sampling(strategy_config: Dict[str, Any]) -> SamplingMethod:
    """Build the random sampling method of a strategy.

    Args:
        strategy_config: Strategy configuration with ``name`` and
            ``traffic_percentage``.

    Returns:
        The sampling method.

    Raises:
        ActiveLearningConfigurationError: If a required key is missing.
    """
    try:
        sample_function = partial(
            sample_randomly,
            traffic_percentage=strategy_config["traffic_percentage"],
        )
        sampling_method = SamplingMethod(
            name=strategy_config["name"],
            sample=sample_function,
        )
    except KeyError as error:
        raise ActiveLearningConfigurationError(
            f"In configuration of `random_sampling` missing key detected: {error}."
        ) from error

    return sampling_method


def sample_randomly(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    traffic_percentage: float,
) -> bool:
    """Select a datapoint with a fixed probability.

    Args:
        image: Image the prediction was made for; not read.
        prediction: Prediction; not read.
        prediction_type: Task type of the model; not read.
        traffic_percentage: Probability of selecting the datapoint.

    Returns:
        True when the datapoint is selected.
    """
    return random.random() < traffic_percentage
