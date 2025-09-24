import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
)
from inference.core.workflows.core_steps.common.query_language.operations.detections.base import (
    rename_detections,
)


def test_rename_detections_when_not_sv_detections_provided() -> None:
    # when
    with pytest.raises(InvalidInputTypeError):
        _ = rename_detections(
            detections="invalid",
            class_map={"a": "b"},
            strict=True,
            new_classes_id_offset=0,
            global_parameters={},
        )


def test_rename_detections_when_strict_mode_enabled_and_all_classes_present() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A", "b": "B"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    ), "Expected no to change"
    assert np.allclose(result.confidence, np.array([0.3, 0.4])), "Expected no to change"
    assert np.allclose(
        result.class_id, np.array([0, 1])
    ), "Expected to change with mapping"
    assert result.data["class_name"].tolist() == [
        "A",
        "B",
    ], "Expected to change with mapping"


def test_rename_detections_when_strict_mode_enabled_and_not_all_classes_present() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    with pytest.raises(OperationError):
        _ = rename_detections(
            detections=detections,
            class_map={"a": "A"},
            strict=True,
            new_classes_id_offset=1024,
            global_parameters={},
        )


def test_rename_detections_when_non_strict_mode_enabled_and_all_classes_present() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A", "b": "B"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    ), "Expected no to change"
    assert np.allclose(result.confidence, np.array([0.3, 0.4])), "Expected no to change"
    assert np.allclose(
        result.class_id, np.array([1024, 1025])
    ), "Expected to change with mapping"
    assert result.data["class_name"].tolist() == [
        "A",
        "B",
    ], "Expected to change with mapping"


def test_rename_detections_when_non_strict_mode_enabled_and_not_all_classes_present() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    ), "Expected no to change"
    assert np.allclose(result.confidence, np.array([0.3, 0.4])), "Expected no to change"
    assert np.allclose(
        result.class_id, np.array([1024, 11])
    ), "Expected to change with mapping"
    assert result.data["class_name"].tolist() == [
        "A",
        "b",
    ], "Expected to change with mapping"


def test_rename_detections_when_mapping_is_parametrised() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["a", "b"])},
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map="class_map_param",
        strict="strict_param",
        new_classes_id_offset=1024,
        global_parameters={
            "strict_param": True,
            "class_map_param": {"a": "A", "b": "B"},
        },
    )

    # then
    assert np.allclose(
        result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    ), "Expected no to change"
    assert np.allclose(result.confidence, np.array([0.3, 0.4])), "Expected no to change"
    assert np.allclose(
        result.class_id, np.array([0, 1])
    ), "Expected to change with mapping"
    assert result.data["class_name"].tolist() == [
        "A",
        "B",
    ], "Expected to change with mapping"
