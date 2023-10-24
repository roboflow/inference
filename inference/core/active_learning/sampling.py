import random
from functools import partial
from typing import Dict, Any

import numpy as np

from inference.core.active_learning.entities import (
    PredictionType,
    Prediction,
    SamplingResult,
    SamplingMethod,
)


def initialize_random_sampling(strategy_config: Dict[str, Any]) -> SamplingMethod:
    sample_function = partial(
        sample_randomly,
        traffic_percentage=strategy_config["traffic_percentage"],
        dataset_splits=strategy_config["dataset_splits"],
    )
    return SamplingMethod(
        name=strategy_config["name"],
        sample=sample_function,
    )


def sample_randomly(
    image: np.ndarray,
    prediction: Prediction,
    prediction_type: PredictionType,
    traffic_percentage: float,
    dataset_splits: Dict[str, float],
) -> SamplingResult:
    if random.random() >= traffic_percentage:
        return SamplingResult(datapoint_selected=False)
    sum_of_split_masses = sum(dataset_splits.values())
    split_names, splits_probabilities_unscaled = list(zip(*dataset_splits.items()))
    splits_probabilities = [
        p / sum_of_split_masses for p in splits_probabilities_unscaled
    ]
    target_split = np.random.choice(split_names, p=splits_probabilities)
    return SamplingResult(datapoint_selected=True, target_split=target_split)
