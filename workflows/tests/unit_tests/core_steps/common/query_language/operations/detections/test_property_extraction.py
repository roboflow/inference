import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
)
from roboflow_workflows.core_steps.common.query_language.entities.operations import (
    DetectionsPropertyExtract,
)
from roboflow_workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from roboflow_workflows.core_steps.common.query_language.operations.core import (
    execute_operations,
)
from roboflow_workflows.core_steps.formatters.property_definition.v1 import (
    PropertyDefinitionBlockV1,
)
from roboflow_workflows.core_steps.transformations.bounding_rect.v1 import (
    BoundingRectBlockV1,
)
from roboflow_workflows.core_steps.transformations.bounding_rect.v1_tensor import (
    BoundingRectBlockV1 as TensorBoundingRectBlockV1,
)
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections


@pytest.fixture
def detections():
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        result = Detections(
            xyxy=torch.tensor([[10, 20, 110, 70], [30, 40, 90, 80]]),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.75, 0.5]),
        )
    else:
        result = sv.Detections(
            xyxy=np.array([[10, 20, 110, 70], [30, 40, 90, 80]]),
            class_id=np.array([0, 1]),
            confidence=np.array([0.75, 0.5]),
        )

    return result


def test_extract_tracker_ids(detections) -> None:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        detections.bboxes_metadata = [{"tracker_id": 42}, {"tracker_id": 0}]
    else:
        detections.tracker_id = np.array([42, 0])

    result = execute_operations(
        value=detections,
        operations=[
            {"type": "DetectionsPropertyExtract", "property_name": "tracker_id"}
        ],
    )

    assert result == [42, 0]


@pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="Supervision stores native tracker IDs separately from metadata",
)
def test_native_tracker_ids_take_precedence_over_metadata(detections) -> None:
    detections.tracker_id = np.array([42, 0])
    detections.data["tracker_id"] = np.array([99, 100])

    result = execute_operations(
        value=detections,
        operations=[
            {"type": "DetectionsPropertyExtract", "property_name": "tracker_id"}
        ],
    )

    assert result == [42, 0]


@pytest.mark.parametrize(
    "property_name, values",
    [("width", [12.5, 27.0]), ("tracker_id", [42, 0]), ("time_in_zone", [3.5, 0.0])],
)
def test_extract_existing_metadata(detections, property_name, values) -> None:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        detections.bboxes_metadata = [{property_name: value} for value in values]
    else:
        detections.data[property_name] = np.array(values)

    result = execute_operations(
        value=detections,
        operations=[
            {"type": "DetectionsPropertyExtract", "property_name": property_name}
        ],
    )

    # Measured widths must not be replaced with the axis-aligned box widths.
    assert result == values


@pytest.mark.parametrize("property_name", list(DetectionsProperty))
def test_extract_property_from_empty_detections(property_name) -> None:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        detections = Detections(
            xyxy=torch.empty((0, 4)),
            class_id=torch.empty(0, dtype=torch.long),
            confidence=torch.empty(0),
        )
    else:
        detections = sv.Detections.empty()

    result = execute_operations(
        value=detections,
        operations=[
            {"type": "DetectionsPropertyExtract", "property_name": property_name.value}
        ],
    )

    assert result == []


def test_extract_width_after_bounding_rect_on_empty_frame() -> None:
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        detections = InstanceDetections(
            xyxy=torch.empty((0, 4)),
            class_id=torch.empty(0, dtype=torch.long),
            confidence=torch.empty(0),
            mask=torch.empty((0, 32, 32), dtype=torch.bool),
        )
        bounding_rect = TensorBoundingRectBlockV1()
    else:
        detections = sv.Detections(
            xyxy=np.empty((0, 4)), mask=np.empty((0, 32, 32), dtype=bool)
        )
        bounding_rect = BoundingRectBlockV1()

    measured = bounding_rect.run(predictions=detections)["detections_with_rect"]
    result = PropertyDefinitionBlockV1().run(
        data=measured,
        operations=[
            DetectionsPropertyExtract(
                type="DetectionsPropertyExtract",
                property_name=DetectionsProperty.BOUNDING_RECT_WIDTH,
            )
        ],
    )

    assert result == {"output": []}


@pytest.mark.parametrize("property_name", ["tracker_id", "width", "time_in_zone"])
def test_extract_missing_property_reports_operation_context(
    detections, property_name
) -> None:
    with pytest.raises(OperationError, match=f"property `{property_name}`") as error:
        execute_operations(
            value=detections,
            operations=[
                {"type": "DetectionsPropertyExtract", "property_name": property_name}
            ],
        )

    assert error.value.context == (
        "step_execution | roboflow_query_language_evaluation | <root>[0]"
    )


@pytest.mark.parametrize(
    "property_name, expected",
    [("confidence", [0.75, 0.5]), ("class_id", [0, 1]), ("x_max", [110, 90])],
)
def test_extract_builtin_property(detections, property_name, expected) -> None:
    result = execute_operations(
        value=detections,
        operations=[
            {"type": "DetectionsPropertyExtract", "property_name": property_name}
        ],
    )

    assert result == expected


@pytest.mark.parametrize("value", [None, [], {}])
def test_extract_property_rejects_invalid_input(value) -> None:
    with pytest.raises(InvalidInputTypeError):
        execute_operations(
            value=value,
            operations=[
                {"type": "DetectionsPropertyExtract", "property_name": "tracker_id"}
            ],
        )
