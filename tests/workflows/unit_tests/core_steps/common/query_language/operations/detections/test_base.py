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


def test_rename_detections_when_detections_have_no_class_name_data_and_not_strict() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
    )

    # when
    result = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then - with no class names to rename, non-strict mode is a no-op: the
    # detections pass through length-consistent and unchanged, rather than having
    # class_id / class_name overwritten with zero-length arrays for the 2 real boxes.
    assert isinstance(result, sv.Detections)
    assert len(result) == 2, "Expected the 2 input boxes to be preserved"
    assert np.allclose(result.xyxy, np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))
    assert np.allclose(
        result.class_id, np.array([10, 11])
    ), "Expected class_id unchanged in the no-op path"
    assert np.allclose(result.confidence, np.array([0.3, 0.4]))
    assert (
        "class_name" not in result.data
    ), "Expected no fabricated class_name field on the no-op path"


def test_rename_detections_when_detections_have_no_class_name_data_and_strict() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        class_id=np.array([10, 11]),
        confidence=np.array([0.3, 0.4]),
    )

    # when / then - strict mode cannot guarantee class_map coverage without class
    # names, so it raises rather than emitting a length-mismatched result.
    with pytest.raises(OperationError):
        rename_detections(
            detections=detections,
            class_map={"a": "A"},
            strict=True,
            new_classes_id_offset=1024,
            global_parameters={},
        )


def test_rename_detections_when_detections_are_empty_is_a_noop() -> None:
    # given
    detections = sv.Detections.empty()

    # when - empty detections have nothing to rename; both modes return a
    # consistent (empty) result without raising.
    non_strict = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=False,
        new_classes_id_offset=1024,
        global_parameters={},
    )
    strict = rename_detections(
        detections=detections,
        class_map={"a": "A"},
        strict=True,
        new_classes_id_offset=1024,
        global_parameters={},
    )

    # then
    assert isinstance(non_strict, sv.Detections) and len(non_strict) == 0
    assert isinstance(strict, sv.Detections) and len(strict) == 0


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
