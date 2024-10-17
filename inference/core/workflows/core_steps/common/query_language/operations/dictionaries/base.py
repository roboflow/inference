import json
from typing import Any

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)


def dictionary_to_json(value: Any, execution_context: str, **kwargs) -> str:
    if not isinstance(value, dict):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing dictionary_to_json(...) in context {execution_context}, "
            f"expected dict object as value, got {value_as_str} of type {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    try:
        return json.dumps(value)
    except Exception as e:
        raise OperationError(
            public_message=f"While Using operation dictionary_to_json(...) in context {execution_context} "
            f"encountered error: {e}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
