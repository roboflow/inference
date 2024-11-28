import pytest

from inference_cli.lib.benchmark_adapter import ensure_error_rate_is_below_threshold


def test_ensure_error_rate_is_below_threshold_when_threshold_not_given() -> None:
    # when
    ensure_error_rate_is_below_threshold(error_rate=30.0, threshold=None)

    # then - no error


def test_ensure_error_rate_is_below_threshold_when_value_below_threshold() -> None:
    # when
    ensure_error_rate_is_below_threshold(error_rate=30.0, threshold=30.5)

    # then - no error


def test_ensure_error_rate_is_below_threshold_when_value_equal_to_threshold() -> None:
    # when
    ensure_error_rate_is_below_threshold(error_rate=30.5, threshold=30.5)

    # then - no error


def test_ensure_error_rate_is_below_threshold_when_value_above_threshold() -> None:
    # when
    with pytest.raises(RuntimeError):
        ensure_error_rate_is_below_threshold(error_rate=30.51, threshold=30.5)
