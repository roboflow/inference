import base64
from typing import Any

import cv2
import numpy as np

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
    ImageProperty.ASPECT_RATIO: lambda image: (
        image.numpy_image.shape[1] / image.numpy_image.shape[0]
        if image.numpy_image.shape[0] != 0
        else 0
    ),
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


def encode_image_to_jpeg(
    value: Any, compression_level: int, execution_context: str, **kwargs
) -> bytes:
    if not isinstance(value, WorkflowImageData):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing encode_image_to_jpeg(...) in context {execution_context}, "
            f"expected WorkflowImageData object as value, got {value_as_str} of type {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    try:
        workflow_image: WorkflowImageData = value
        return _image_to_jpeg_bytes(
            image=workflow_image.numpy_image, compression_level=compression_level
        )
    except Exception as e:
        raise OperationError(
            public_message=f"While Using operation encode_image_to_jpeg(...) in context {execution_context} "
            f"encountered error: {e}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def encode_image_to_base64(value: Any, execution_context: str, **kwargs) -> str:
    if not isinstance(value, WorkflowImageData):
        value_as_str = safe_stringify(value=value)
        raise InvalidInputTypeError(
            public_message=f"Executing encode_image_to_base64(...) in context {execution_context}, "
            f"expected WorkflowImageData object as value, got {value_as_str} of type {type(value)}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
        )
    try:
        workflow_image: WorkflowImageData = value
        return workflow_image.base64_image
    except Exception as e:
        raise OperationError(
            public_message=f"While Using operation encode_image_to_base64(...) in context {execution_context} "
            f"encountered error: {e}",
            context=f"step_execution | roboflow_query_language_evaluation | {execution_context}",
            inner_error=e,
        )


def _image_to_jpeg_bytes(image: np.ndarray, compression_level: int) -> bytes:
    encoding_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_level]
    success, img_encoded = cv2.imencode(".jpg", image, encoding_param)
    if not success:
        raise ValueError(f"Could not encode image into JPEG")
    return np.array(img_encoded).tobytes()
