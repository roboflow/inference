from typing import Any

from roboflow_workflows.core_steps.common.query_language.entities.enums import (
    ActionRecognitionProperty,
)
from roboflow_workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.entities import (
    ActionRecognitionResult,
)


def extract_action_recognition_property(
    value: Any,
    *,
    property_name: ActionRecognitionProperty,
    execution_context: str,
    **kwargs,
) -> Any:
    """Read a public field from the model's per-frame result.

    Args:
        value: Result produced by the Action Recognition Model.
        property_name: Result field to return.
        execution_context: Operation location used for error reporting.
        **kwargs: Execution parameters supplied by the query language engine.

    Returns:
        The selected field, including a null bound during warm-up.

    Raises:
        InvalidInputTypeError: If the input has a different kind.
    """
    if not isinstance(value, ActionRecognitionResult):
        raise InvalidInputTypeError(
            public_message=f"Expected an action recognition result, got {type(value).__name__}.",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )

    result_field = getattr(value, property_name.value)
    return result_field
