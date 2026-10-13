"""Rules the annotation blocks share, each declared once (readability R1-R3).

::

    color axis   colors.palette_indices    host arrays and device tensors
    tracker ids  rows.tracker_ids_of       DrawRows and the mask block
    palette      colors.resolve_palette    literal and selected values
    bounds       each block's _CHECKS      literal and selected values
"""

from typing import Any

import numpy as np
import pytest
import torch
from roboflow_workflows.v2.blocks.visualization.annotations import (
    BoundingBoxVisualizationBlock,
    MaskVisualizationBlock,
    PolygonVisualizationBlock,
    SemanticSegmentationVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.annotations.colors import (
    COLOR_AXES,
    palette_indices,
)
from roboflow_workflows.v2.blocks.visualization.annotations.rows import (
    read_rows,
    tracker_ids_of,
)
from roboflow_workflows.v2.engine.declaration import spec_of
from roboflow_workflows.v2.engine.errors import ParamsValidationError

from inference_models.models.base.object_detection import Detections
from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    image_of,
    instance_detections,
    scene,
    semantic,
)

CLASS_ID = np.array([4, -1, 2**53 + 1], dtype=np.int64)
TRACKER_ID = [7, 2**62 + 1, -3]


@pytest.mark.parametrize("axis", COLOR_AXES)
def test_device_indices_equal_host_indices_and_stay_on_the_device(axis: str) -> None:
    # when
    host = palette_indices(axis, class_id=CLASS_ID, tracker_id=TRACKER_ID)
    device = palette_indices(
        axis, class_id=torch.from_numpy(CLASS_ID), tracker_id=TRACKER_ID
    )

    # then
    assert isinstance(host, np.ndarray) and host.dtype == np.int64
    assert isinstance(device, torch.Tensor) and device.dtype == torch.int64
    np.testing.assert_array_equal(device.numpy(), host)


@pytest.mark.parametrize("axis", COLOR_AXES)
def test_device_indices_never_read_class_ids_on_the_host(axis: str) -> None:
    # given: a meta tensor has no values; any host read would fail
    class_id = torch.empty(3, dtype=torch.int64, device="meta")

    # when
    indices = palette_indices(axis, class_id=class_id, tracker_id=TRACKER_ID)

    # then
    assert indices.device.type == "meta" and indices.shape == (3,)


@pytest.mark.parametrize(
    "axis, tracker_id, message",
    [
        ("TRACK", None, "add a tracker step"),
        ("PER_ROW", TRACKER_ID, "color_axis must be one of"),
    ],
)
def test_axis_errors_are_the_same_on_host_and_device(axis, tracker_id, message) -> None:
    errors = []
    for class_id in (CLASS_ID, torch.from_numpy(CLASS_ID)):
        with pytest.raises(ValueError, match=message) as error:
            palette_indices(axis, class_id=class_id, tracker_id=tracker_id)
        errors.append(str(error.value))

    assert errors[0] == errors[1]


def test_tracker_ids_need_an_id_in_every_row() -> None:
    # when
    complete = tracker_ids_of([{"tracker_id": 2**62 + 1}, {"tracker_id": 3}])
    one_missing = tracker_ids_of([{"tracker_id": 1}, {}])
    none_row = tracker_ids_of([{"tracker_id": 1}, None])
    no_rows = tracker_ids_of([])

    # then
    np.testing.assert_array_equal(complete, [2**62 + 1, 3])
    assert complete.dtype == np.int64
    assert one_missing is None and none_row is None
    assert no_rows.shape == (0,) and no_rows.dtype == np.int64


def test_draw_rows_read_tracker_ids_with_the_same_reader() -> None:
    # given
    image = image_of(scene())
    rows = [{"tracker_id": 5}, {"tracker_id": 6}, {"tracker_id": 7}]

    # when
    tracker_ids = read_rows(detections(image, rows=rows)).tracker_ids()

    # then
    np.testing.assert_array_equal(tracker_ids, tracker_ids_of(rows))


@pytest.mark.parametrize(
    "dtype, host_dtype",
    [
        (torch.float16, np.float32),
        (torch.bfloat16, np.float32),
        (torch.float32, np.float32),
        (torch.float64, np.float64),
    ],
)
def test_host_boxes_are_float64_only_for_float64_device_boxes(
    dtype, host_dtype
) -> None:
    # given
    boxes = detections(image_of(scene()))
    typed = Detections(
        xyxy=boxes.xyxy.to(dtype),
        class_id=boxes.class_id,
        confidence=boxes.confidence,
        image_metadata=boxes.image_metadata,
    )

    # when
    rows = read_rows(typed)

    # then
    assert rows.xyxy.dtype == host_dtype


def predictions_for(block_class: type, image) -> Any:
    if block_class is BoundingBoxVisualizationBlock:
        return detections(image)
    if block_class is SemanticSegmentationVisualizationBlock:
        return semantic(image)
    return instance_detections(image)


@pytest.mark.parametrize(
    "block_class, name, value, message",
    [
        (BoundingBoxVisualizationBlock, "thickness", 0, "thickness must be an integer >= 1, got 0"),
        (BoundingBoxVisualizationBlock, "roundness", 1.5, "roundness must be a number in [0.0, 1.0], got 1.5"),
        (BoundingBoxVisualizationBlock, "palette_size", 0, "palette_size must be at least 1, got 0"),
        (PolygonVisualizationBlock, "thickness", 0, "thickness must be an integer >= 1, got 0"),
        (MaskVisualizationBlock, "opacity", 2, "opacity must be a number in [0.0, 1.0], got 2"),
        (MaskVisualizationBlock, "palette_size", 0, "palette_size must be at least 1, got 0"),
        (SemanticSegmentationVisualizationBlock, "opacity", -1, "opacity must be a number in [0.0, 1.0], got -1"),
        (SemanticSegmentationVisualizationBlock, "palette_size", 0, "palette_size must be at least 1, got 0"),
    ],
)  # fmt: skip
def test_a_selected_value_fails_with_the_literal_error_text(
    block_class: type, name: str, value: Any, message: str
) -> None:
    # given
    image = image_of(scene())
    refs = {"image": "$inputs.image", "predictions": "$inputs.predictions"}
    spec = spec_of(block_class)

    # when: the literal at compile time, then the same value selected at run
    with pytest.raises(ParamsValidationError) as compile_error:
        spec.validate_params({**refs, name: value})
    params = dict(spec.validate_params({**refs, name: f"$inputs.{name}"}))
    arguments = {
        **params,
        name: value,
        "image": image,
        "predictions": predictions_for(block_class, image),
    }
    with pytest.raises(ValueError) as run_error:
        block_class().run(**arguments)

    # then
    assert message in str(compile_error.value)
    assert message in str(run_error.value)
