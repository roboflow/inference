from typing import Any

from inference.core.workflows.core_steps.common.query_language.errors import InvalidInputTypeError
from inference.core.workflows.core_steps.common.query_language.operations.utils import safe_stringify


def string_to_lower(value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidInputTypeError(
            public_message=f"Using operation string_to_lower(...) requires string as input data, got: {type(value)}",
            context="step_execution | roboflow_query_language_evaluation"
        )
    return value.lower()


def string_to_upper(value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidInputTypeError(
            public_message=f"Using operation string_to_upper(...) requires string as input data, got: {type(value)}",
            context="step_execution | roboflow_query_language_evaluation"
        )
    return value.upper()


def to_string(value: Any) -> str:
    try:
        return str(value)
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"Using operation to_string(...) requires value possible to be converted into string, "
                           f"got: {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )


def string_sub_sequence(value: Any, start: int, end: int) -> str:
    if not isinstance(value, str):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing string_sub_sequence(...), "
                           f"got value which of type {type(value)}: {value_as_str}",
            context="step_execution | roboflow_query_language_evaluation",
        )
    return value[start:end]
