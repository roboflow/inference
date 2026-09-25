import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.common.query_language.entities.enums import (
    DetectionsProperty,
)
from roboflow_workflows.core_steps.common.query_language.operations.detection.base import (
    extract_detection_property,
)
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
    CLASS_NAMES_KEY,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    POLYGON_KEY_IN_SV_DETECTIONS,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION `extract_detection_property` routes to the
# tensor-native extractor, which expects a 7-element tuple
# (xyxy, mask, class_id, confidence, tracker_id, data, metadata) instead of the sv
# single-detection 6-tuple (note class_id/confidence are swapped vs sv). The 6-tuple
# tests below are skipped when the flag is on; each has a `*_tensor_native` parity test
# (skipped when the flag is off) exercising the same scenario with the 7-tuple form.
# `extract_detection_property` is an ELEMENT-level op (one detection tuple, not a
# collection), so it has no Detections/InstanceDetections/KeyPoints code path; the
# representation only surfaces as which optional `data` keys are populated — hence the
# KEYPOINTS_XY (keypoint-origin) and POLYGON (instance-segmentation-origin) cases below.
_KEYPOINTS_XY = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
_POLYGON = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv 6-tuple input; extract_detection_property is native-only under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


@_NUMPY_ONLY
@pytest.mark.parametrize("dtype", [str, object], ids=["unicode-array", "object-array"])
@pytest.mark.parametrize("class_name", ["truck", "camión", ""])
def test_extract_class_name_from_supervision_detection(dtype, class_name: str) -> None:
    detections = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 50.0, 50.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([1]),
        data={"class_name": np.array([class_name], dtype=dtype)},
    )

    result = extract_detection_property(
        value=next(iter(detections)),
        property_name=DetectionsProperty.CLASS_NAME,
        execution_context="<test>",
    )

    assert result == class_name
    assert type(result) is str


@_NUMPY_ONLY
def test_extract_detection_property_with_area_px() -> None:
    # given
    detection = (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.float32(0.9),
        np.int64(1),
        None,
        {
            "class_name": np.array("leaf"),
            AREA_KEY_IN_SV_DETECTIONS: np.float32(400.0),
            AREA_CONVERTED_KEY_IN_SV_DETECTIONS: np.float32(4.0),
        },
    )

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.AREA,
        execution_context="<test>",
    )

    # then
    assert result == 400.0


@_NUMPY_ONLY
def test_extract_detection_property_with_area_converted() -> None:
    # given
    detection = (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.float32(0.9),
        np.int64(1),
        None,
        {
            "class_name": np.array("leaf"),
            AREA_KEY_IN_SV_DETECTIONS: np.float32(400.0),
            AREA_CONVERTED_KEY_IN_SV_DETECTIONS: np.float32(4.0),
        },
    )

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.AREA_CONVERTED,
        execution_context="<test>",
    )

    # then
    assert result == 4.0


@_NUMPY_ONLY
def test_extract_detection_property_with_keypoints_xy() -> None:
    # given - a detection from a keypoint model carries its keypoint coordinates in the
    # per-detection data dict (index 5 of the sv 6-tuple).
    detection = (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.float32(0.9),
        np.int64(1),
        None,
        {KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: _KEYPOINTS_XY},
    )

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.KEYPOINTS_XY,
        execution_context="<test>",
    )

    # then
    assert np.array_equal(result, _KEYPOINTS_XY)


@_NUMPY_ONLY
def test_extract_detection_property_with_polygon() -> None:
    # given - a detection from an instance-segmentation model carries its mask polygon
    # in the per-detection data dict (index 5 of the sv 6-tuple).
    detection = (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.float32(0.9),
        np.int64(1),
        None,
        {POLYGON_KEY_IN_SV_DETECTIONS: _POLYGON},
    )

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.POLYGON,
        execution_context="<test>",
    )

    # then
    assert np.array_equal(result, _POLYGON)


def _single_tensor_native_detection() -> tuple:
    # 7-tuple (xyxy, mask, class_id, confidence, tracker_id, data, metadata) as yielded
    # when iterating a native `inference_models.Detections`. AREA / AREA_CONVERTED are
    # read from `data` (index 5); `metadata` (index 6) carries the class_id -> name map
    # and is not consulted for the area properties.
    return (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.int64(1),
        np.float32(0.9),
        None,
        {
            AREA_KEY_IN_SV_DETECTIONS: np.float32(400.0),
            AREA_CONVERTED_KEY_IN_SV_DETECTIONS: np.float32(4.0),
        },
        {CLASS_NAMES_KEY: {1: "leaf"}},
    )


@_TENSOR_ONLY
@pytest.mark.parametrize("class_name", ["truck", "camión", ""])
def test_extract_class_name_tensor_native(class_name: str) -> None:
    detection = _single_tensor_native_detection()
    detection[6][CLASS_NAMES_KEY][1] = class_name

    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.CLASS_NAME,
        execution_context="<test>",
    )

    assert result == class_name
    assert type(result) is str


@_TENSOR_ONLY
def test_extract_detection_property_with_area_px_tensor_native() -> None:
    # given
    detection = _single_tensor_native_detection()

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.AREA,
        execution_context="<test>",
    )

    # then
    assert result == 400.0


@_TENSOR_ONLY
def test_extract_detection_property_with_area_converted_tensor_native() -> None:
    # given
    detection = _single_tensor_native_detection()

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.AREA_CONVERTED,
        execution_context="<test>",
    )

    # then
    assert result == 4.0


@_TENSOR_ONLY
def test_extract_detection_property_with_keypoints_xy_tensor_native() -> None:
    # given - keypoint-origin detection; keypoints live in `data` (index 5 of the
    # tensor-native 7-tuple).
    detection = (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.int64(1),
        np.float32(0.9),
        None,
        {KEYPOINTS_XY_KEY_IN_SV_DETECTIONS: _KEYPOINTS_XY},
        {CLASS_NAMES_KEY: {1: "person"}},
    )

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.KEYPOINTS_XY,
        execution_context="<test>",
    )

    # then
    assert np.array_equal(result, _KEYPOINTS_XY)


@_TENSOR_ONLY
def test_extract_detection_property_with_polygon_tensor_native() -> None:
    # given - instance-segmentation-origin detection; mask polygon lives in `data`
    # (index 5 of the tensor-native 7-tuple).
    detection = (
        np.array([0.0, 0.0, 50.0, 50.0], dtype=np.float32),
        None,
        np.int64(1),
        np.float32(0.9),
        None,
        {POLYGON_KEY_IN_SV_DETECTIONS: _POLYGON},
        {CLASS_NAMES_KEY: {1: "leaf"}},
    )

    # when
    result = extract_detection_property(
        value=detection,
        property_name=DetectionsProperty.POLYGON,
        execution_context="<test>",
    )

    # then
    assert np.array_equal(result, _POLYGON)
