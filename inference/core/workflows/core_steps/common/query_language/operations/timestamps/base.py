from datetime import datetime
from typing import Any

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)


def timestamp_to_iso_format(value: Any, execution_context: str, **kwargs) -> str:
    if not isinstance(value, datetime):
        raise InvalidInputTypeError(
            public_message=f"Operation timestamp_to_iso_format(...) in context {execution_context} expects "
            f"input value to be timestamp (Python datetime object), got: {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    return value.isoformat()
