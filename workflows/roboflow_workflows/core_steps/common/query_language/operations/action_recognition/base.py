from typing import Any

from roboflow_workflows.core_steps.common.deserializers import (
    deserialize_action_recognition_result_kind,
)
from roboflow_workflows.core_steps.common.query_language.entities.enums import (
    ActionRecognitionProperty,
)
from roboflow_workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from roboflow_workflows.errors import RuntimeInputError


def extract_action_recognition_property(
    value: Any,
    *,
    property_name: ActionRecognitionProperty,
    execution_context: str,
    **kwargs,
) -> Any:
    """Read a public field from the model's per-frame result.

    Args:
        value: Model result or its serialized workflow parameter.
        property_name: Result field to return.
        execution_context: Operation location used for error reporting.
        **kwargs: Execution parameters supplied by the query language engine.

    Returns:
        The selected field, including a null bound during warm-up.

    Raises:
        InvalidInputTypeError: If the value is not a valid action recognition result.
    """
    try:
        result = deserialize_action_recognition_result_kind(
            parameter=execution_context, value=value
        )
    except RuntimeInputError as error:
        raise InvalidInputTypeError(
            public_message="Expected a valid action recognition result.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=error,
        ) from error

    result_field = getattr(result, property_name.value)
    return result_field
