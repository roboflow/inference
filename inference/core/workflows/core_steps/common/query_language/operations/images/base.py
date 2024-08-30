from typing import Any

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    ImageProperty,
)
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.utils import (
    safe_stringify,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

PROPERTY_EXTRACTORS = {
    ImageProperty.SIZE: lambda image: image.numpy_image.shape[0]
    * image.numpy_image.shape[1],
    ImageProperty.HEIGHT: lambda image: image.numpy_image.shape[0],
    ImageProperty.WIDTH: lambda image: image.numpy_image.shape[1],
}


def extract_image_property(
    value: Any, property_name: ImageProperty, execution_context: str, **kwargs
) -> bool:
    if not isinstance(value, WorkflowImageData):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing extract_detections_property(...) in context {execution_context}, "
            f"expected WorkflowImageData object as value, got {value_as_str} of type {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    try:
        return PROPERTY_EXTRACTORS[property_name](value)
    except Exception as e:
        raise OperationError(
            public_message=f"While Using operation extract_detections_property(...) in context {execution_context} "
            f"encountered error: {e}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
