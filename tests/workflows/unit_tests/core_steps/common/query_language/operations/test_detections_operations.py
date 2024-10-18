import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)


def test_detections_to_dictionary_when_invalid_input_is_provided() -> None:
    # given
    operations = [{"type": "DetectionsToDictionary"}]
    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value="invalid", operations=operations)


def test_detections_to_dictionary_when_valid_input_is_provided() -> None:
    # given
    operations = [{"type": "DetectionsToDictionary"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.3, 0.4]),
        data={
            "class_name": np.array(["cat", "dog"]),
            "detection_id": np.array(["one", "two"]),
            "image_dimensions": np.array([[192, 168], [192, 168]]),
        },
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result == {
        "image": {"width": 168, "height": 192},
        "predictions": [
            {
                "width": 2.0,
                "height": 2.0,
                "x": 1.0,
                "y": 2.0,
                "confidence": 0.3,
                "class_id": 0,
                "class": "cat",
                "detection_id": "one",
            },
            {
                "width": 2.0,
                "height": 2.0,
                "x": 5.0,
                "y": 6.0,
                "confidence": 0.4,
                "class_id": 1,
                "class": "dog",
                "detection_id": "two",
            },
        ],
    }


def test_detections_to_dictionary_when_malformed_input_is_provided() -> None:
    # given
    operations = [{"type": "DetectionsToDictionary"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.3, 0.4]),
    )

    # when
    with pytest.raises(OperationError):
        _ = execute_operations(value=detections, operations=operations)
