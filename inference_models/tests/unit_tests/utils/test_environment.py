from typing import Any

import pytest

from inference_models.errors import InvalidEnvVariable
from inference_models.utils.environment import parse_comma_separated_values, str2bool


def test_parse_comma_separated_values_when_empty_value_provided() -> None:
    # when
    result = parse_comma_separated_values(values="")

    # then
    assert result == [], "Expected empty output"


def test_parse_comma_separated_values_when_single_value_provided() -> None:
    # when
    result = parse_comma_separated_values(values="some")

    # then
    assert result == ["some"], "Expected single output"


def test_parse_comma_separated_values_when_single_not_stripped_value_provided() -> None:
    # when
    result = parse_comma_separated_values(values="    some     ")

    # then
    assert result == ["some"], "Expected single output"


def test_parse_comma_separated_values_when_empty_non_stripped_value_provided() -> None:
    # when
    result = parse_comma_separated_values(values="        ")

    # then
    assert result == [], "Expected empty output"


def test_parse_comma_separated_values_when_multiple_not_stripped_value_provided() -> (
    None
):
    # when
    result = parse_comma_separated_values(values="    some     , other      ")

    # then
    assert result == ["some", "other"], "Expected two stripped outputs"


def test_parse_comma_separated_values_when_multiple_stripped_value_provided() -> None:
    # when
    result = parse_comma_separated_values(values="some,other")

    # then
    assert result == ["some", "other"], "Expected two stripped outputs"


def test_str2bool_when_values_is_bool() -> None:
    # when
    result = str2bool(value=False, variable_name="some")

    # then
    assert result is False


def test_str2bool_when_value_type_is_invalid() -> None:
    # when
    with pytest.raises(InvalidEnvVariable):
        _ = str2bool(value=3.7, variable_name="some")


def test_str2bool_when_value_content_is_invalid() -> None:
    # when
    with pytest.raises(InvalidEnvVariable):
        _ = str2bool(value="invalid", variable_name="some")


@pytest.mark.parametrize("value", ["true", "TruE", "True"])
def test_str2bool_when_value_should_be_true(value: Any) -> None:
    # when
    result = str2bool(value=value, variable_name="some")

    # then
    assert result is True


@pytest.mark.parametrize("value", ["False", "false", "FalSe"])
def test_str2bool_when_value_should_be_false(value: Any) -> None:
    # when
    result = str2bool(value=value, variable_name="some")

    # then
    assert result is False
