from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.active_learning.samplers import random
from inference.core.active_learning.samplers.random import initialize_random_sampling
from inference.core.exceptions import ActiveLearningConfigurationError


@mock.patch.object(random.random, "random")
def test_initialize_random_sampling_when_config_is_valid(
    random_mock: MagicMock,
) -> None:
    # given
    random_mock.side_effect = [0.1, 0.2, 0.5, 0.7, 0.9]
    strategy_config = {
        "name": "default_strategy",
        "type": "random_sampling",
        "traffic_percentage": 0.5,
        "tags": ["c", "d"],
        "limits": [
            {"type": "minutely", "value": 10},
            {"type": "hourly", "value": 100},
            {"type": "daily", "value": 1000},
        ],
    }

    # when
    result = initialize_random_sampling(strategy_config=strategy_config)
    sampling_results = []
    for _ in range(5):
        sampling_results.append(
            result.sample(
                np.zeros((128, 128, 3), dtype=np.ndarray),
                {"some": "prediction"},
                "object-detection",
            )
        )

    # then
    assert result.name == "default_strategy"
    assert sampling_results == [True, True, False, False, False]


def test_initialize_random_sampling_when_strategy_name_is_not_present() -> None:
    # given
    strategy_config = {
        "type": "some_strategy",
        "traffic_percentage": 0.5,
        "tags": ["c", "d"],
        "limits": [
            {"type": "minutely", "value": 10},
            {"type": "hourly", "value": 100},
            {"type": "daily", "value": 1000},
        ],
    }

    # when
    with pytest.raises(ActiveLearningConfigurationError):
        _ = initialize_random_sampling(strategy_config=strategy_config)


def test_initialize_random_sampling_when_traffic_percentage_is_not_present() -> None:
    # given
    strategy_config = {
        "name": "default_strategy",
        "type": "some_strategy",
        "tags": ["c", "d"],
        "limits": [
            {"type": "minutely", "value": 10},
            {"type": "hourly", "value": 100},
            {"type": "daily", "value": 1000},
        ],
    }

    # when
    with pytest.raises(ActiveLearningConfigurationError):
        _ = initialize_random_sampling(strategy_config=strategy_config)
