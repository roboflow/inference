from typing import Any

from inference.core.workflows.core_steps.common.query_language.errors import InvalidInputTypeError


def to_bool(value: Any) -> bool:
    try:
        return bool(value)
    except (TypeError, ValueError) as e:
        raise InvalidInputTypeError(
            public_message=f"Using operation to_bool(...) requires value possible to be converted into boolean, "
                           f"got: {type(value)}",
            context="step_execution | roboflow_query_language_evaluation",
            inner_error=e,
        )
