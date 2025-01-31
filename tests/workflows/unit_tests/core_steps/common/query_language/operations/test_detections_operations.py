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


def test_picking_detections_by_parent_class_when_invalid_input_provided() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execute_operations(value="FOR SURE NOT DETECTIONS", operations=operations)


def test_picking_detections_by_parent_class_when_empty_detections_provided() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections.empty()

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert result is not detections
    assert len(result) == 0


def test_picking_detections_by_parent_class_when_class_name_field_not_defined() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
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
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 0


def test_picking_detections_by_parent_class_when_parent_class_not_fond() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
            ]
        ),
        class_id=np.array([0, 1]),
        confidence=np.array([0.3, 0.4]),
        data={"class_name": np.array(["c", "d"])},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 0


def test_picking_detections_by_parent_class_when_no_child_detections_matching() -> None:
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 10, 10],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ]
        ),
        class_id=np.array([0, 0, 1]),
        confidence=np.array([0.3, 0.4, 0.5]),
        data={"class_name": np.array(["a", "a", "b"])},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 2
    assert np.allclose(result.xyxy, np.array([[0, 0, 10, 10], [20, 20, 30, 30]]))
    assert np.allclose(result.confidence, [0.3, 0.4])


def test_picking_detections_by_parent_class_when_there_are_child_detections_matching() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 50, 50],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
            ]
        ),
        class_id=np.array([0, 1, 1]),
        confidence=np.array([0.3, 0.4, 0.5]),
        data={"class_name": np.array(["a", "b", "b"])},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 3
    assert np.allclose(
        result.xyxy, np.array([[0, 0, 50, 50], [20, 20, 30, 30], [40, 40, 50, 50]])
    )
    assert np.allclose(result.confidence, [0.3, 0.4, 0.5])


def test_picking_detections_by_parent_class_when_there_are_child_detections_matching_different_parents() -> (
    None
):
    # given
    operations = [{"type": "PickDetectionsByParentClass", "parent_class": "a"}]
    detections = sv.Detections(
        xyxy=np.array(
            [
                [0, 0, 50, 50],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
                [100, 100, 200, 200],
                [150, 100, 250, 200],
                [400, 400, 600, 600],
            ]
        ),
        class_id=np.array([0, 1, 1, 0, 2, 3]),
        confidence=np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.9]),
        data={"class_name": np.array(["a", "b", "b", "a", "c", "d"])},
    )

    # when
    result = execute_operations(value=detections, operations=operations)

    # then
    assert len(result) == 5
    assert np.allclose(
        result.xyxy,
        np.array(
            [
                [0, 0, 50, 50],
                [100, 100, 200, 200],
                [20, 20, 30, 30],
                [40, 40, 50, 50],
                [150, 100, 250, 200],
            ]
        ),
    )
    assert np.allclose(result.confidence, [0.3, 0.6, 0.4, 0.5, 0.7])
