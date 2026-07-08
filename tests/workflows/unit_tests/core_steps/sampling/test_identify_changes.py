import numpy as np
import pytest

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


def test_identify_changes_zero_variance_is_not_outlier() -> None:
    # given
    identify_changes_block = IdentifyChangesBlockV1()
    embedding = np.array([0.1, -0.4, 0.3, 0.9, -0.2])

    # when - the second sample is scored against freshly initialised
    # cosine-similarity statistics (std is the literal 0), so the zero-std
    # guard must yield the deterministic non-outlier result instead of
    # dividing by zero and emitting NaN downstream; two runs keep the check
    # platform-exact (accumulating more steps re-introduces float rounding
    # noise into std, which made the assertions flake across architectures)
    inputs = {**default_inputs, "warmup": 1}
    for _ in range(2):
        result = identify_changes_block.run(**inputs, embedding=embedding)

    # then
    assert not result.get("warming_up")
    assert not result.get("is_outlier")
    assert result.get("percentile") == 0.5
    assert result.get("z_score") == 0


def test_identify_changes_tensor_sibling_zero_variance_is_not_outlier() -> None:
    # given
    torch = pytest.importorskip("torch")
    from inference.core.workflows.core_steps.sampling.identify_changes.v1_tensor import (
        IdentifyChangesBlockV1 as IdentifyChangesTensorBlockV1,
    )

    identify_changes_block = IdentifyChangesTensorBlockV1()
    embedding = torch.tensor([0.1, -0.4, 0.3, 0.9, -0.2])

    # when - same deterministic zero-std guard scenario as the numpy sibling
    inputs = {**default_inputs, "warmup": 1}
    for _ in range(2):
        result = identify_changes_block.run(**inputs, embedding=embedding)

    # then
    assert not result.get("warming_up")
    assert not result.get("is_outlier")
    assert result.get("percentile") == 0.5
    assert result.get("z_score") == 0
