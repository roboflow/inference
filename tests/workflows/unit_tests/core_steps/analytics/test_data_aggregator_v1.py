import pytest

from inference.core.workflows.core_steps.analytics.data_aggregator.v1 import (
    DataAggregatorBlockV1,
    ValuesDifferenceState,
)


def _feed(state: ValuesDifferenceState, values):
    for value in values:
        state.on_data(value=value)
    return state.get_result()


def test_values_difference_with_decreasing_then_min_pair():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [10, 1])

    # Assert
    assert result == 9


def test_values_difference_with_max_first_then_smaller_values():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [10, 1, 5])

    # Assert
    assert result == 9


def test_values_difference_with_monotonically_decreasing_values():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [5, 4, 3, 2, 1])

    # Assert
    assert result == 4


def test_values_difference_with_monotonically_increasing_values():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [1, 2, 3, 4, 5])

    # Assert
    assert result == 4


def test_values_difference_with_single_value_returns_zero():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [7])

    # Assert
    assert result == 0


def test_values_difference_with_no_values_returns_none():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = state.get_result()

    # Assert
    assert result is None


def test_values_difference_with_all_equal_values_returns_zero():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [5, 5, 5])

    # Assert
    assert result == 0


def test_values_difference_with_negative_values():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [-10, -3, -7])

    # Assert
    assert result == 7


def test_values_difference_with_float_values():
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [1.5, 3.7, 2.0])

    # Assert
    assert result == pytest.approx(2.2)


def test_values_difference_is_never_negative():
    # Regression test: previously the first observed value was locked into the
    # min slot and the second into the max slot without cross-comparison, so a
    # descending stream like [10, 1] produced -9 instead of 9.
    # Arrange
    state = ValuesDifferenceState()

    # Act
    result = _feed(state, [10, 1])

    # Assert
    assert result == 9
    assert result >= 0


def test_values_difference_matches_max_minus_min_end_to_end():
    # Regression test at the block level: the documented behaviour is
    # values_difference == max - min. Before the fix the block could return
    # values_difference=0 while reporting max=10 and min=1.
    # Arrange
    block = DataAggregatorBlockV1()
    aggregation_mode = {"speed": ["values_difference", "max", "min"]}
    observations = [10, 1, 5, 5]

    # Act
    result = None
    for value in observations:
        result = block.run(
            data={"speed": value},
            data_operations={},
            aggregation_mode=aggregation_mode,
            interval_unit="runs",
            interval=4,
        )

    # Assert
    assert result is not None
    assert result["speed_max"] == 10
    assert result["speed_min"] == 1
    assert (
        result["speed_values_difference"] == result["speed_max"] - result["speed_min"]
    )
    assert result["speed_values_difference"] == 9
