import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.classical_cv.size_measurement.v1 import (
    OUTPUT_KEY,
    SizeMeasurementBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import Batch


def test_size_measurement_block():
    # given
    reference_predictions = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        mask=None,
        class_id=np.array([0]),
    )
    object_predictions = sv.Detections(
        xyxy=np.array([[20, 20, 60, 60], [30, 30, 70, 70]]),
        confidence=np.array([0.8, 0.85]),
        mask=None,
        class_id=np.array([0, 1]),
    )
    reference_dimensions = "5.0,5.0"

    # when
    block = SizeMeasurementBlockV1()
    result = block.run(reference_predictions, object_predictions, reference_dimensions)

    # then
    expected_dimensions = [
        {"width": 5.0, "height": 5.0, "longer": 5.0, "shorter": 5.0},
        {"width": 5.0, "height": 5.0, "longer": 5.0, "shorter": 5.0},
    ]
    assert result == {
        OUTPUT_KEY: expected_dimensions
    }, f"Expected {expected_dimensions}, but got {result}"


def test_size_measurement_block_with_mask():
    # given
    reference_predictions = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        mask=np.array([np.ones((100, 100), dtype=np.uint8)]),
        class_id=np.array([0]),
    )
    object_predictions = sv.Detections(
        xyxy=np.array([[20, 20, 60, 60], [30, 30, 70, 70]]),
        confidence=np.array([0.8, 0.85]),
        mask=np.array(
            [np.ones((100, 100), dtype=np.uint8), np.ones((100, 100), dtype=np.uint8)]
        ),
        class_id=np.array([0, 1]),
    )
    reference_dimensions = "5.0,5.0"

    # when
    block = SizeMeasurementBlockV1()
    result = block.run(reference_predictions, object_predictions, reference_dimensions)

    # then
    expected_dimensions = [
        {"width": 5.0, "height": 5.0, "longer": 5.0, "shorter": 5.0},
        {"width": 5.0, "height": 5.0, "longer": 5.0, "shorter": 5.0},
    ]
    assert result == {
        OUTPUT_KEY: expected_dimensions
    }, f"Expected {expected_dimensions}, but got {result}"


def test_size_measurement_block_with_invalid_reference_dimensions():
    # given
    reference_predictions = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        mask=None,
        class_id=np.array([0]),
    )
    object_predictions = sv.Detections(
        xyxy=np.array([[20, 20, 60, 60], [30, 30, 70, 70]]),
        confidence=np.array([0.8, 0.85]),
        mask=None,
        class_id=np.array([0, 1]),
    )
    reference_dimensions = "invalid"

    # when
    block = SizeMeasurementBlockV1()
    with pytest.raises(
        expected_exception=ValueError,
        match="reference_dimensions must be a string in the format 'width,height'",
    ):
        block.run(reference_predictions, object_predictions, reference_dimensions)


def test_size_measurement_tensor_block_skips_objects_with_empty_masks():
    # regression for the None-dimension guard ported from the numpy source
    # (c03c2514b): an InstanceDetections row whose mask decodes to no contour
    # yields (None, None) and must produce a None entry, not a TypeError.
    pytest.importorskip("inference_models")
    import torch

    from inference.core.workflows.core_steps.classical_cv.size_measurement.v1_tensor import (
        OUTPUT_KEY as TENSOR_OUTPUT_KEY,
    )
    from inference.core.workflows.core_steps.classical_cv.size_measurement.v1_tensor import (
        SizeMeasurementBlockV1 as TensorSizeMeasurementBlockV1,
    )
    from inference_models.models.base.instance_segmentation import InstanceDetections
    from inference_models.models.base.object_detection import Detections

    # given
    reference_predictions = Detections(
        xyxy=torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([0.9]),
    )
    valid_mask = torch.zeros((100, 100), dtype=torch.bool)
    valid_mask[10:60, 10:60] = True
    empty_mask = torch.zeros((100, 100), dtype=torch.bool)
    object_predictions = InstanceDetections(
        xyxy=torch.tensor([[10.0, 10.0, 60.0, 60.0], [30.0, 30.0, 70.0, 70.0]]),
        class_id=torch.tensor([0, 1]),
        confidence=torch.tensor([0.8, 0.85]),
        mask=torch.stack([valid_mask, empty_mask]),
    )

    # when
    block = TensorSizeMeasurementBlockV1()
    result = block.run(reference_predictions, object_predictions, "5.0,5.0")

    # then
    dimensions = result[TENSOR_OUTPUT_KEY]
    assert len(dimensions) == 2
    assert dimensions[0] is not None
    assert set(dimensions[0].keys()) == {"width", "height", "longer", "shorter"}
    assert dimensions[1] is None
