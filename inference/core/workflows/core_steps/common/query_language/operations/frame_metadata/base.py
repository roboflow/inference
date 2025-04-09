from typing import Any

from inference.core.workflows.core_steps.common.query_language.entities.enums import (
    FrameMetadataProperty,
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
    FrameMetadataProperty.FRAME_NUMBER: lambda image: image.video_metadata.frame_number,
    FrameMetadataProperty.FRAME_TIMESTAMP: lambda image: image.video_metadata.frame_timestamp,
    FrameMetadataProperty.SECONDS_SINCE_START: lambda image: (
        image.video_metadata.frame_number - 1
    )
    / image.video_metadata.fps,
}


def extract_frame_metadata(
    value: Any, property_name: FrameMetadataProperty, execution_context: str, **kwargs
):
    if not isinstance(value, WorkflowImageData):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing extract_frame_metadata(...) in context {execution_context}, "
            f"expected WorkflowImageData object as value, got {value_as_str} of type {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    try:
        return PROPERTY_EXTRACTORS[property_name](value)
    except Exception as e:
        raise OperationError(
            public_message=f"While Using operation extract_frame_metadata(...) in context {execution_context} "
            f"encountered error: {e}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )
