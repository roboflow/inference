import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.classical_cv.size_measurement.v1 import (
    OUTPUT_KEY,
    SizeMeasurementBlockV1,
    get_polygon_dimensions,
)


@pytest.mark.parametrize("backend", ["numpy", "tensor"])
@pytest.mark.parametrize(
    ("polygon", "expected"),
    [
        pytest.param(
            [[1.25, 2.25], [11.75, 2.25], [11.75, 7.75], [1.25, 7.75]],
            (10.5, 5.5),
            id="fractional",
        ),
        pytest.param(
            [[1, 2], [11, 2], [11, 7], [1, 7]],
            (10.0, 5.0),
            id="integer",
        ),
        pytest.param(
            [[1.25, 1.25], [2.25, 2.25], [3.25, 3.25]],
            (None, None),
            id="collinear",
        ),
    ],
)
def test_size_measurement_polygon_dimensions(
    backend: str,
    polygon: list,
    expected: tuple[float | None, float | None],
) -> None:
    """Measure polygon sides without losing fractional coordinates.

    Args:
        backend (str): The block implementation under test.
        polygon (list): Polygon vertices in pixel coordinates.
        expected (tuple): Expected width and height in pixels.
    """
    measure = get_polygon_dimensions
    if backend == "tensor":
        pytest.importorskip("inference_models")
        from roboflow_workflows.core_steps.classical_cv.size_measurement.v1_tensor import (
            get_polygon_dimensions as get_tensor_polygon_dimensions,
        )

        measure = get_tensor_polygon_dimensions

    assert measure(polygon) == expected


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
    from roboflow_workflows.core_steps.classical_cv.size_measurement.v1_tensor import (
        OUTPUT_KEY as TENSOR_OUTPUT_KEY,
    )
    from roboflow_workflows.core_steps.classical_cv.size_measurement.v1_tensor import (
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
