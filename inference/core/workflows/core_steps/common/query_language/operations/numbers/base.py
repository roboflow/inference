from typing import Any, Union

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    NumberCastingMode,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)


def to_number(
    value: Any, cast_to: NumberCastingMode, execution_context: str, **kwargs
) -> Union[float, int]:
    try:
        value = float(value)
        if cast_to is cast_to.INT:
            return value
        return int(value)
    except (TypeError, ValueError) as e:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing operation to_number(...) in context {execution_context}, encountered "
            f"value `{value_as_str}` of type {type(value)} which cannot be casted into {cast_to.value}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def number_round(
    value: Union[float, int], decimal_digits: int, execution_context: str, **kwargs
) -> Union[float, int]:
    try:
        return round(value, decimal_digits)
    except (TypeError, ValueError) as e:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing number_round to_number(...) in context {execution_context}, encountered "
            f"value `{value_as_str}` of type {type(value)} which cannot be rounded",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def multiply(
    value: Union[float, int], other: Union[float, int], execution_context: str, **kwargs
) -> Union[float, int]:
    try:
        return value * other
    except (TypeError, ValueError) as e:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing multiply to_number(...) in context {execution_context}, encountered "
            f"value `{value_as_str}` of type {type(value)} which cannot be multiplied with {other}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def divide(
    value: Union[float, int], other: Union[float, int], execution_context: str, **kwargs
) -> Union[float, int]:
    try:
        return value / other
    except (TypeError, ValueError, ZeroDivisionError) as e:
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing multiply divide(...) in context {execution_context}, encountered "
            f"value `{value_as_str}` of type {type(value)} which cannot be multiplied with {other}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
