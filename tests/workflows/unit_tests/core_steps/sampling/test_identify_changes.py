import numpy as np

from inference.core.workflows.core_steps.sampling.identify_changes.v1 import (
    IdentifyChangesBlockV1,
)

default_inputs = {
    "strategy": "Exponential Moving Average (EMA)",
    "threshold_percentile": 0.2,
    "warmup": 3,
    "smoothing_factor": 0.1,
    "window_size": 10,
}


def get_perturbed_value(initial_value: np.ndarray, perturbation: float) -> np.ndarray:
    # randomly fluctuate by +- rand in perturbation in dimensions
    return initial_value + np.random.uniform(0, perturbation, size=len(initial_value))


def test_identify_changes() -> None:
    # given
    identify_changes_block = IdentifyChangesBlockV1()

    # 5 random floats between -1 and 1
    initial_value = np.random.uniform(-1, 1, size=5)
    initial_value_normalized = np.array(initial_value) / np.linalg.norm(initial_value)

    # warm up
    result = identify_changes_block.run(**default_inputs, embedding=initial_value)

    assert result is not None
    assert not result.get("is_outlier")
    assert result.get("warming_up")

    for i in range(10):
        result = identify_changes_block.run(**default_inputs, embedding=initial_value)

    assert np.allclose(result.get("average"), initial_value_normalized)
    assert np.allclose(result.get("std"), [0, 0, 0, 0, 0])
    assert not result.get("is_outlier")
    assert not result.get("warming_up")

    # add a bit of variance
    for i in range(10):
        result = identify_changes_block.run(
            **default_inputs, embedding=get_perturbed_value(initial_value, 1e-3)
        )

    result = identify_changes_block.run(**default_inputs, embedding=initial_value)
    assert not result.get("is_outlier")

    # ensure that the average and std have changed
    assert not np.allclose(result.get("average"), initial_value_normalized)
    assert not np.all(result.get("std") == [0, 0, 0, 0, 0])

    # make a large change
    result = identify_changes_block.run(
        **default_inputs, embedding=[0.5, 0.5, 0.5, 0.5, 0.5]
    )

    assert result.get("is_outlier")
    # average and std should not be zero anymore
    assert not np.allclose(result.get("average"), initial_value_normalized)
    assert not np.all(result.get("std") == [0, 0, 0, 0, 0])
