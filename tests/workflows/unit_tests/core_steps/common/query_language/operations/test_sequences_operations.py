import pytest

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)


@pytest.mark.parametrize(
    "value, separator, expected",
    [
        (["a", "b", "c"], ", ", "a, b, c"),
        (["a"], ", ", "a"),
        ([], ", ", ""),
        ([1, 2, 3], "-", "1-2-3"),
        (["a", "b"], "\n", "a\nb"),
    ],
)
def test_sequence_join_operation(value: list, separator: str, expected: str) -> None:
    """SequenceJoin joins sequence elements into a single string using the separator."""
    operations = [{"type": "SequenceJoin", "separator": separator}]
    result = execute_operations(value=value, operations=operations)
    assert result == expected


def test_sequence_join_operation_default_separator() -> None:
    """SequenceJoin defaults to ', ' as separator."""
    operations = [{"type": "SequenceJoin"}]
    result = execute_operations(value=["a", "b"], operations=operations)
    assert result == "a, b"


def test_sequence_join_operation_invalid_input_raises() -> None:
    """SequenceJoin raises InvalidInputTypeError for non-iterable input."""
    operations = [{"type": "SequenceJoin", "separator": ", "}]
    with pytest.raises(InvalidInputTypeError):
        execute_operations(value=123, operations=operations)
