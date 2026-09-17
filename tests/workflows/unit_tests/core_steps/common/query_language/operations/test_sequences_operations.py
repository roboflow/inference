import pytest

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
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


def test_sequence_join_operation_when_output_exceeds_length_limit() -> None:
    """SequenceJoin raises when the joined string would exceed the 1,000,000 char limit."""
    operations = [{"type": "SequenceJoin", "separator": ", "}]
    with pytest.raises(OperationError) as error:
        execute_operations(value=["a" * 400_000] * 3, operations=operations)
    assert "1000000" in str(error.value)


def test_sequence_join_operation_when_output_is_within_length_limit() -> None:
    """SequenceJoin joins normally when the result stays under the length limit."""
    operations = [{"type": "SequenceJoin", "separator": ", "}]
    result = execute_operations(value=["a" * 10, "b" * 10], operations=operations)
    assert result == "a" * 10 + ", " + "b" * 10
