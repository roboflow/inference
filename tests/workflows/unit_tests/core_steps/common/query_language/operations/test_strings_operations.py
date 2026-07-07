import pytest

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)


@pytest.mark.parametrize(
    "value, regex, expected",
    [
        ("hello", "hello", True),
        ("hello world", "hello", True),
        ("say hello", "hello", True),
        ("hello", "^hello$", True),
        ("hello", "world", False),
        ("world", "hello", False),
        ("hell", "hello", False),
    ],
)
def test_string_matches_operation(value: str, regex: str, expected: bool) -> None:
    """StringMatches returns True when the string matches the regex and False otherwise."""
    operations = [{"type": "StringMatches", "regex": regex}]
    result = execute_operations(value=value, operations=operations)
    assert result is expected


def test_string_matches_operation_invalid_input_raises() -> None:
    """StringMatches raises InvalidInputTypeError for non-string input."""
    operations = [{"type": "StringMatches", "regex": "hello"}]
    with pytest.raises(InvalidInputTypeError):
        execute_operations(value=123, operations=operations)
