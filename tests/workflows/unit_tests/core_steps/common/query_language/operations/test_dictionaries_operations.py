import json

import pytest

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)


def test_dictionary_to_json_when_invalid_input_is_provided() -> None:
    # given
    operations = [{"type": "ConvertDictionaryToJSON"}]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value="some", operations=operations)


def test_dictionary_to_json_when_valid_input_is_provided() -> None:
    # given
    operations = [{"type": "ConvertDictionaryToJSON"}]
    data = {"some": "data"}

    # when
    result = execute_operations(value=data, operations=operations)

    # then
    assert json.loads(result) == data


def test_dictionary_to_json_when_non_serializable_input_is_provided() -> None:
    # given
    operations = [{"type": "ConvertDictionaryToJSON"}]
    data = {"some": {"a", "b"}}

    # when
    with pytest.raises(OperationError):
        _ = execute_operations(value=data, operations=operations)
