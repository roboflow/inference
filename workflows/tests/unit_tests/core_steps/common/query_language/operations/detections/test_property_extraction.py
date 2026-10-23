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

from inference_models.models.base.instance_segmentation import (
    InstanceDetections as NativeInstanceDetections,
)
from inference_models.models.base.keypoints_detection import (
    KeyPoints as NativeKeyPoints,
)
from inference_models.models.base.object_detection import Detections as NativeDetections

# Under ENABLE_TENSOR_DATA_REPRESENTATION `extract_detections_property` dispatches to
# its tensor-native sibling, which rejects sv.Detections. Each sv-based test below is
# skipped when the flag is on and has a `*_tensor_native` parity test (skipped when
# the flag is off) driving the same scenario through `inference_models` objects, where
# per-detection properties live in `bboxes_metadata` instead of `sv.Detections.data`.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections input; extract_detections_property is native-only under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

_EXPECTED_MISSING_PROPERTY_CONTEXT = (
    "step_execution | roboflow_query_language_evaluation | <root>[0]"
)


def _extract(detections, property_name: str):
    return execute_operations(
        value=detections,
        operations=[
            {"type": "DetectionsPropertyExtract", "property_name": property_name}
        ],
    )


def _sv_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[10, 20, 110, 70], [30, 40, 90, 80]]),
        class_id=np.array([0, 1]),
        confidence=np.array([0.75, 0.5]),
    )


def _native_detections() -> NativeDetections:
    return NativeDetections(
        xyxy=torch.tensor([[10, 20, 110, 70], [30, 40, 90, 80]]),
        class_id=torch.tensor([0, 1]),
        confidence=torch.tensor([0.75, 0.5]),
    )


def _empty_native_detections() -> NativeDetections:
    return NativeDetections(
        xyxy=torch.empty((0, 4)),
        class_id=torch.empty(0, dtype=torch.long),
        confidence=torch.empty(0),
    )


def _empty_native_key_points() -> NativeKeyPoints:
    return NativeKeyPoints(
        xy=torch.empty((0, 17, 2)),
        class_id=torch.empty(0, dtype=torch.long),
        confidence=torch.empty((0, 17)),
    )


@_NUMPY_ONLY
def test_extract_tracker_ids() -> None:
    detections = _sv_detections()
    detections.tracker_id = np.array([42, 0])

    result = _extract(detections, "tracker_id")

    assert result == [42, 0]


@_TENSOR_ONLY
def test_extract_tracker_ids_tensor_native() -> None:
    detections = _native_detections()
    detections.bboxes_metadata = [{"tracker_id": 42}, {"tracker_id": 0}]

    result = _extract(detections, "tracker_id")

    assert result == [42, 0]


@_NUMPY_ONLY
def test_native_tracker_ids_take_precedence_over_data() -> None:
    # Supervision keeps tracker IDs in a native field; nothing in the repo writes
    # `data["tracker_id"]`, so the native field is the system of record.
    detections = _sv_detections()
    detections.tracker_id = np.array([42, 0])
    detections.data["tracker_id"] = np.array([99, 100])

    result = _extract(detections, "tracker_id")

    assert result == [42, 0]


@_NUMPY_ONLY
@pytest.mark.parametrize(
    "property_name, expected",
    [("confidence", [0.75, 0.5]), ("class_id", [0, 1]), ("width", [12.5, 27.0])],
)
def test_tracker_ids_do_not_leak_into_other_properties(property_name, expected) -> None:
    # The native tracker_id read must stay keyed on the requested property.
    detections = _sv_detections()
    detections.tracker_id = np.array([42, 0])
    detections.data["width"] = np.array([12.5, 27.0])

    result = _extract(detections, property_name)

    assert result == expected


@_TENSOR_ONLY
@pytest.mark.parametrize(
    "property_name, expected",
    [("confidence", [0.75, 0.5]), ("class_id", [0, 1]), ("width", [12.5, 27.0])],
)
def test_tracker_ids_do_not_leak_into_other_properties_tensor_native(
    property_name, expected
) -> None:
    detections = _native_detections()
    detections.bboxes_metadata = [
        {"tracker_id": 42, "width": 12.5},
        {"tracker_id": 0, "width": 27.0},
    ]

    result = _extract(detections, property_name)

    assert result == expected


@_NUMPY_ONLY
@pytest.mark.parametrize(
    "property_name, values",
    [("width", [12.5, 27.0]), ("tracker_id", [42, 0]), ("time_in_zone", [3.5, 0.0])],
)
def test_extract_property_stored_in_data(property_name, values) -> None:
    detections = _sv_detections()
    detections.data[property_name] = np.array(values)

    result = _extract(detections, property_name)

    # A value an upstream block stored under the property's key is returned as is.
    assert result == values


@_TENSOR_ONLY
@pytest.mark.parametrize(
    "property_name, values",
    [("width", [12.5, 27.0]), ("tracker_id", [42, 0]), ("time_in_zone", [3.5, 0.0])],
)
def test_extract_property_stored_in_bboxes_metadata_tensor_native(
    property_name, values
) -> None:
    detections = _native_detections()
    detections.bboxes_metadata = [{property_name: value} for value in values]

    result = _extract(detections, property_name)

    assert result == values


@_NUMPY_ONLY
def test_data_value_wins_over_builtin_extractor() -> None:
    # Long-standing numpy behavior: a `data` entry named like a built-in property
    # shadows the built-in extractor. The tensor sibling has no such override path
    # (built-ins always read the native tensors), so this has no parity test.
    detections = _sv_detections()
    detections.data["confidence"] = np.array([0.1, 0.2])

    result = _extract(detections, "confidence")

    assert result == [0.1, 0.2]


@_NUMPY_ONLY
@pytest.mark.parametrize("property_name", list(DetectionsProperty))
def test_extract_property_from_empty_detections(property_name) -> None:
    result = _extract(sv.Detections.empty(), property_name.value)

    assert result == []


@_TENSOR_ONLY
@pytest.mark.parametrize("property_name", list(DetectionsProperty))
def test_extract_property_from_empty_detections_tensor_native(property_name) -> None:
    result = _extract(_empty_native_detections(), property_name.value)

    assert result == []


@_TENSOR_ONLY
@pytest.mark.parametrize("property_name", list(DetectionsProperty))
def test_extract_property_from_empty_key_points_prediction_tensor_native(
    property_name,
) -> None:
    # The empty guard must run on the bounding-box component, after the keypoint
    # tuple is split; `len()` of the unsplit `(KeyPoints, Detections)` tuple is 2.
    prediction = (_empty_native_key_points(), _empty_native_detections())

    result = _extract(prediction, property_name.value)

    assert result == []


@_NUMPY_ONLY
def test_extract_width_after_bounding_rect_on_empty_frame() -> None:
    detections = sv.Detections(
        xyxy=np.empty((0, 4)), mask=np.empty((0, 32, 32), dtype=bool)
    )

    measured = BoundingRectBlockV1().run(predictions=detections)["detections_with_rect"]
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


@_TENSOR_ONLY
def test_extract_width_after_bounding_rect_on_empty_frame_tensor_native() -> None:
    detections = NativeInstanceDetections(
        xyxy=torch.empty((0, 4)),
        class_id=torch.empty(0, dtype=torch.long),
        confidence=torch.empty(0),
        mask=torch.empty((0, 32, 32), dtype=torch.bool),
    )

    measured = TensorBoundingRectBlockV1().run(predictions=detections)[
        "detections_with_rect"
    ]
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


@_NUMPY_ONLY
@pytest.mark.parametrize("property_name", ["tracker_id", "width", "time_in_zone"])
def test_extract_missing_property_reports_operation_context(property_name) -> None:
    with pytest.raises(OperationError, match=f"property `{property_name}`") as error:
        _extract(_sv_detections(), property_name)

    assert error.value.context == _EXPECTED_MISSING_PROPERTY_CONTEXT


@_TENSOR_ONLY
@pytest.mark.parametrize("property_name", ["tracker_id", "width", "time_in_zone"])
def test_extract_missing_property_reports_operation_context_tensor_native(
    property_name,
) -> None:
    with pytest.raises(OperationError, match=f"property `{property_name}`") as error:
        _extract(_native_detections(), property_name)

    assert error.value.context == _EXPECTED_MISSING_PROPERTY_CONTEXT


@_NUMPY_ONLY
def test_missing_property_message_matches_tensor_native() -> None:
    # Keep the public message identical across representation flags; the tensor
    # parity test asserts the same string.
    with pytest.raises(OperationError) as error:
        _extract(_sv_detections(), "time_in_zone")

    assert "property `time_in_zone` is not available on these detections" in str(
        error.value.public_message
    )


@_TENSOR_ONLY
def test_missing_property_message_matches_numpy_tensor_native() -> None:
    with pytest.raises(OperationError) as error:
        _extract(_native_detections(), "time_in_zone")

    assert "property `time_in_zone` is not available on these detections" in str(
        error.value.public_message
    )


@_NUMPY_ONLY
@pytest.mark.parametrize(
    "property_name, expected",
    [("confidence", [0.75, 0.5]), ("class_id", [0, 1]), ("x_max", [110, 90])],
)
def test_extract_builtin_property(property_name, expected) -> None:
    result = _extract(_sv_detections(), property_name)

    assert result == expected


@_TENSOR_ONLY
@pytest.mark.parametrize(
    "property_name, expected",
    [("confidence", [0.75, 0.5]), ("class_id", [0, 1]), ("x_max", [110, 90])],
)
def test_extract_builtin_property_tensor_native(property_name, expected) -> None:
    result = _extract(_native_detections(), property_name)

    assert result == expected


@pytest.mark.parametrize("value", [None, [], {}])
def test_extract_property_rejects_invalid_input(value) -> None:
    # Runs under both flags: the type guard precedes the empty-detections guard.
    with pytest.raises(InvalidInputTypeError):
        _extract(value, "tracker_id")
