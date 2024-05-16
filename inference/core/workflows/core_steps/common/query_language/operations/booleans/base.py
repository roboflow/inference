from typing import Any

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)


def to_bool(value: Any, execution_context: str, **kwargs) -> bool:
    try:
        return bool(value)
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"Using operation to_bool(...) in context {execution_context} requires value "
            f"possible to be converted into boolean, got: {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
