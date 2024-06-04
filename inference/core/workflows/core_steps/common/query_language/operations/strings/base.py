import re
from typing import Any

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)


def string_to_lower(value: Any, execution_context: str, **kwargs) -> str:
    if not isinstance(value, str):
        raise InvalidInputTypeError(
            public_message=f"Using operation string_to_lower(...) in context {execution_context} which requires "
            f"string as input data, got: {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return value.lower()


def string_to_upper(value: Any, execution_context: str, **kwargs) -> str:
    if not isinstance(value, str):
        raise InvalidInputTypeError(
            public_message=f"Using operation string_to_upper(...) in context {execution_context} which requires "
            f"string as input data, got: {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return value.upper()


def to_string(value: Any, execution_context: str, **kwargs) -> str:
    try:
        return str(value)
    except (RuntimeError, RuntimeError) as e:
        raise InvalidInputTypeError(
            public_message=f"Using operation to_string(...) in context {execution_context} caused the following "
            f"error: {e} of type {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def string_sub_sequence(
    value: Any, start: int, end: int, execution_context: str, **kwargs
) -> str:
    if not isinstance(value, str):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing string_sub_sequence(...) in context {execution_context},"
            f"got value which of type {type(value)}: {value_as_str}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return value[start:end]


def string_matches(value: Any, regex: str, execution_context: str, **kwargs) -> bool:
    if not isinstance(value, str):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing string_matches(...) in context {execution_context}, "
            f"got value which of type {type(value)}: {value_as_str}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return not re.match(pattern=regex, string=value)
