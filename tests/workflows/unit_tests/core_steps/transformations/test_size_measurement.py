import numpy as np
import supervision as sv
from inference.core.workflows.core_steps.transformations.size_measurement.v1 import (
    SizeMeasurementBlockV1,
    OUTPUT_KEY,
)
from inference.core.workflows.execution_engine.entities.base import Batch

def test_size_measurement_block():
    # given
    reference_predictions = Batch(
        content=[
            sv.Detections(
                xyxy=np.array([[10, 10, 50, 50]]),
                confidence=np.array([0.9]),
                mask=None,
                class_id=np.array([0])
            )
        ],
        indices=[(0,)]
    )
    object_predictions = Batch(
        content=[
            sv.Detections(
                xyxy=np.array([[20, 20, 60, 60], [30, 30, 70, 70]]),
                confidence=np.array([0.8, 0.85]),
                mask=None,
                class_id=np.array([0, 1])
            )
        ],
        indices=[(0,), (1,)]
    )
    reference_dimensions = "5.0,5.0"

    # when
    block = SizeMeasurementBlockV1()
    result = block.run(reference_predictions, object_predictions, reference_dimensions)

    # then
    expected_dimensions = [
        {"width": 5.0, "height": 5.0},
        {"width": 5.0, "height": 5.0}
    ]
    assert result == [{OUTPUT_KEY: expected_dimensions}], f"Expected {expected_dimensions}, but got {result}"

def test_size_measurement_block_with_mask():
    # given
    reference_predictions = Batch(
        content=[
            sv.Detections(
                xyxy=np.array([[10, 10, 50, 50]]),
                confidence=np.array([0.9]),
                mask=np.array([np.ones((100, 100), dtype=np.uint8)]),
                class_id=np.array([0])
            )
        ],
        indices=[(0,)]
    )
    object_predictions = Batch(
        content=[
            sv.Detections(
                xyxy=np.array([[20, 20, 60, 60], [30, 30, 70, 70]]),
                confidence=np.array([0.8, 0.85]),
                mask=np.array([np.ones((100, 100), dtype=np.uint8), np.ones((100, 100), dtype=np.uint8)]),
                class_id=np.array([0, 1])
            )
        ],
        indices=[(0,), (1,)]
    )
    reference_dimensions = "5.0,5.0"

    # when
    block = SizeMeasurementBlockV1()
    result = block.run(reference_predictions, object_predictions, reference_dimensions)

    # then
    expected_dimensions = [
        {"width": 5.0, "height": 5.0},
        {"width": 5.0, "height": 5.0}
    ]
    assert result == [{OUTPUT_KEY: expected_dimensions}], f"Expected {expected_dimensions}, but got {result}"

def test_size_measurement_block_with_invalid_reference_dimensions():
    # given
    reference_predictions = Batch(
        content=[
            sv.Detections(
                xyxy=np.array([[10, 10, 50, 50]]),
                confidence=np.array([0.9]),
                mask=None,
                class_id=np.array([0])
            )
        ],
        indices=[(0,)]
    )
    object_predictions = Batch(
        content=[
            sv.Detections(
                xyxy=np.array([[20, 20, 60, 60], [30, 30, 70, 70]]),
                confidence=np.array([0.8, 0.85]),
                mask=None,
                class_id=np.array([0, 1])
            )
        ],
        indices=[(0,), (1,)]
    )
    reference_dimensions = "invalid"

    # when
    block = SizeMeasurementBlockV1()
    try:
        block.run(reference_predictions, object_predictions, reference_dimensions)
        assert False, "Expected ValueError due to invalid reference_dimensions"
    except ValueError as e:
        # then
        assert str(e) == "reference_dimensions must be a string in the format 'width,height'", f"Unexpected error message: {str(e)}"
