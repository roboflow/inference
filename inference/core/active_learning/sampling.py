import random
from functools import partial
from typing import Any, Dict

import numpy as np

from inference.core.active_learning.entities import (
    Prediction,
    PredictionType,
    SamplingMethod,
)


def initialize_random_sampling(strategy_config: Dict[str, Any]) -> SamplingMethod:
    sample_function = partial(
        sample_randomly,
        traffic_percentage=strategy_config["traffic_percentage"],
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
) -> bool:
    if random.random() >= traffic_percentage:
        return False
    return True
