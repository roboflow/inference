import base64
from typing import Any

import cv2
import numpy as np
import pytest

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def test_extract_image_property_operation_when_not_an_image_given() -> None:
    # given
    operations = [
        {
            "type": "ExtractImageProperty",
            "property_name": "size",
        }
    ]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value="some", operations=operations)


@pytest.mark.parametrize(
    "property_name, expected_value",
    [("size", 192 * 168), ("height", 192), ("width", 168)],
)
def test_extract_image_property_operation_when_invalid_property_given(
    property_name: str, expected_value: Any
) -> None:
    # given
    operations = [
        {
            "type": "ExtractImageProperty",
            "property_name": property_name,
        }
    ]
    image = _create_workflow_image()

    # when
    result = execute_operations(value=image, operations=operations)
    assert result == expected_value


def test_encode_image_to_jpeg_when_invalid_input_provided() -> None:
    # given
    operations = [
        {
            "type": "ConvertImageToJPEG",
            "compression_level": 75,
        }
    ]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value="invalid", operations=operations)


def test_encode_image_to_jpeg_when_valid_input_provided() -> None:
    # given
    operations = [
        {
            "type": "ConvertImageToJPEG",
            "compression_level": 75,
        }
    ]
    image = _create_workflow_image()

    # when
    result = execute_operations(value=image, operations=operations)

    # then
    try:
        recovered_image = cv2.imdecode(
            np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR
        )
    except Exception:
        recovered_image = cv2.imdecode(
            np.fromstring(result, np.uint8), cv2.IMREAD_COLOR
        )
    assert np.allclose(image.numpy_image, recovered_image)


def test_encode_image_to_jpeg_when_corrupted_np_array_provided() -> None:
    # given
    operations = [
        {
            "type": "ConvertImageToJPEG",
            "compression_level": 75,
        }
    ]
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some_image"),
        numpy_image=np.zeros((192, 168, 5), dtype=np.uint8),
    )

    # when
    with pytest.raises(OperationError):
        _ = execute_operations(value=image, operations=operations)


def test_encode_image_to_base64_when_invalid_input_provided() -> None:
    # given
    operations = [{"type": "ConvertImageToBase64"}]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value="invalid", operations=operations)


def test_encode_image_to_base64_when_valid_input_provided() -> None:
    # given
    operations = [{"type": "ConvertImageToBase64"}]
    image = _create_workflow_image()

    # when
    result = execute_operations(value=image, operations=operations)

    # then
    result_bytes = base64.b64decode(result)
    try:
        recovered_image = cv2.imdecode(
            np.frombuffer(result_bytes, np.uint8), cv2.IMREAD_COLOR
        )
    except Exception:
        recovered_image = cv2.imdecode(
            np.fromstring(result_bytes, np.uint8), cv2.IMREAD_COLOR
        )
    assert np.allclose(image.numpy_image, recovered_image)


def test_encode_image_to_base64_when_corrupted_np_array_provided() -> None:
    # given
    operations = [{"type": "ConvertImageToBase64"}]
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some_image"),
        numpy_image=np.zeros((192, 168, 5), dtype=np.uint8),
    )

    # when
    with pytest.raises(OperationError):
        _ = execute_operations(value=image, operations=operations)


def _create_workflow_image() -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some_image"),
        numpy_image=np.zeros((192, 168, 3), dtype=np.uint8),
    )
