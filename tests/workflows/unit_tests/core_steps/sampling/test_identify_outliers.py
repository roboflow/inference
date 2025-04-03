import numpy as np
import pytest

from inference.core.workflows.core_steps.sampling.identify_outliers.v1 import (
    IdentifyOutliersBlockV1,
)

default_inputs = {"threshold_percentile": 0.05, "warmup": 3, "window_size": 32}


def get_perturbed_value(initial_value: np.ndarray, perturbation: float) -> np.ndarray:
    # randomly fluctuate by +- rand in perturbation in dimensions
    return initial_value + np.random.uniform(
        -perturbation, perturbation, size=len(initial_value)
    )


@pytest.mark.skip(
    reason="Solve flakiness of the block: https://github.com/roboflow/inference/issues/901"
)
def test_identify_outliers() -> None:
    # given
    identify_changes_block = IdentifyOutliersBlockV1()

    # 5 random floats between -1 and 1
    initial_value = np.random.uniform(-1, 1, size=5)

    # warm up
    result = identify_changes_block.run(**default_inputs, embedding=initial_value)

    assert result is not None
    assert not result.get("is_outlier")
    assert result.get("warming_up")

    # add a bit of variance
    for i in range(32):
        result = identify_changes_block.run(
            **default_inputs, embedding=get_perturbed_value(initial_value, 1e-6)
        )

    assert not result.get("is_outlier")
    assert not result.get("warming_up")

    # make a large change
    result = identify_changes_block.run(
        **default_inputs, embedding=[0.5, 0.5, 0.5, 0.5, 0.5]
    )

    assert result.get("is_outlier")
