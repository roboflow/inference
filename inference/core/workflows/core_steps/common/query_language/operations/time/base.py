from datetime import datetime
from typing import Any, Literal

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)


def get_current_timestamp(**kwargs) -> datetime:
    return datetime.now()


SECONDS_TO_TIME_DIFFERENCE_BASE = {
    "days": 24 * 60 * 60,
    "hours": 60 * 60,
    "minutes": 60,
    "seconds": 1,
    "milliseconds": 1 / 1000,
}


def get_time_difference(
    value: Any,
    base: Literal["days", "hours", "minutes", "seconds", "milliseconds"],
    execution_context: str,
    **kwargs,
) -> float:
    if not isinstance(value, datetime):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"While executing get_time_difference(...) in context {execution_context},"
            f"got value which of type {type(value)}: {value_as_str}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    if base not in SECONDS_TO_TIME_DIFFERENCE_BASE:
        InvalidInputTypeError(
            public_message=f"Executing get_time_difference(...), expected base to be one of "
            f"{SECONDS_TO_TIME_DIFFERENCE_BASE.keys()}, got {base}.",
            context="step_execution | roboflow_query_language_evaluation",
        )
    reference_timestamp: datetime = value
    time_difference_seconds = (datetime.now() - reference_timestamp).total_seconds()
    modifier = SECONDS_TO_TIME_DIFFERENCE_BASE[base]
    return time_difference_seconds / modifier
