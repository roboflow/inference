from typing import Union

import pytest

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)


@pytest.mark.parametrize(
    "value, cast_to, expected",
    [
        ("42", "int", 42),
        ("0", "int", 0),
        ("3.14", "int", 3),
        ("0.5", "float", 0.5),
        ("42", "float", 42.0),
        ("3.14", "float", 3.14),
    ],
)
def test_to_number_operation(
    value: str, cast_to: str, expected: Union[int, float]
) -> None:
    """ToNumber converts values to int or float according to cast_to."""
    operations = [{"type": "ToNumber", "cast_to": cast_to}]
    result = execute_operations(value=value, operations=operations)
    assert result == expected
    assert type(result) is type(expected)


def test_to_number_operation_invalid_input_raises() -> None:
    """ToNumber raises InvalidInputTypeError for non-numeric strings."""
    operations = [{"type": "ToNumber", "cast_to": "int"}]
    with pytest.raises(InvalidInputTypeError):
        execute_operations(value="not a number", operations=operations)
