from typing import Union

import pytest

from inference.core.exceptions import InvalidEnvironmentVariableError
from inference.core.utils.environment import safe_split_value, str2bool


def test_str2bool_when_non_boolean_value_given() -> None:
    # when
    with pytest.raises(InvalidEnvironmentVariableError):
        _ = str2bool(value=3)


@pytest.mark.parametrize("value", ["y", "t", "f", "", "F", "Y", "ok", "invalid"])
def test_str2bool_when_non_convertible_string_given(value: str) -> None:
    # when
    with pytest.raises(InvalidEnvironmentVariableError):
        _ = str2bool(value=value)


@pytest.mark.parametrize(
    "value,expected_result",
    [
        ("TRUE", True),
        ("true", True),
        ("TrUe", True),
        (True, True),
        ("FALSE", False),
        ("false", False),
        ("FaLsE", False),
        (False, False),
    ],
)
def test_str2bool_when_convertible_string_given(
    value: Union[str, bool], expected_result: bool
) -> None:
    # when
    result = str2bool(value=value)

    # then
    assert result is expected_result


def test_safe_split_value_when_none_value_given() -> None:
    # when
    result = safe_split_value(value=None)

    # then
    assert result is None


def test_safe_split_value_when_empty_value_given() -> None:
    # when
    result = safe_split_value(value="")

    # then
    assert result == [""]


def test_safe_split_value_when_splittable_value_given() -> None:
    # when
    result = safe_split_value(value="a,b,c,d")

    # then
    assert result == ["a", "b", "c", "d"]


def test_safe_split_value_when_non_splittable_value_given() -> None:
    # when
    result = safe_split_value(value="a,b,c,d", delimiter="|")

    # then
    assert result == ["a,b,c,d"]
