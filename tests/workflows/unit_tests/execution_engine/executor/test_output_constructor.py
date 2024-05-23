import numpy as np
import supervision as sv

from inference.core.workflows.entities.base import (
    CoordinatesSystem,
    JsonField,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)
from inference.core.workflows.execution_engine.executor.output_constructor import (
    construct_workflow_output,
)


def test_construct_response_when_coordinates_system_does_not_matter_for_specific_field_selector() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {"predictions": ["a", "b"]},
            {"predictions": ["c", "d"]},
        ],
    )
    workflow_outputs = [
        JsonField(
            type="JsonField",
            name="some_own",
            selector="$steps.a.predictions",
            coordinates_system=CoordinatesSystem.OWN,
        ),
        JsonField(
            type="JsonField",
            name="some_parent",
            selector="$steps.a.predictions",
            coordinates_system=CoordinatesSystem.PARENT,
        ),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs, execution_cache=execution_cache
    )

    # then
    assert len(result) == 2, "Expected exactly 2 outputs, as per outputs definitions"
    assert result["some_own"] == result["some_parent"] == [["a", "b"], ["c", "d"]], (
        "Expected both registered outputs to hold the same data, as coordinates system choice should not affect "
        "non-detections outputs"
    )


def test_construct_response_when_coordinates_system_does_not_matter_for_wildcard_selector() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[
            OutputDefinition(name="predictions"),
            OutputDefinition(name="foo"),
        ],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {"predictions": ["a", "b"], "foo": 1},
            {"predictions": ["c", "d"], "foo": 2},
        ],
    )
    workflow_outputs = [
        JsonField(
            type="JsonField",
            name="a_own",
            selector="$steps.a.*",
            coordinates_system=CoordinatesSystem.OWN,
        ),
        JsonField(
            type="JsonField",
            name="a_parent",
            selector="$steps.a.*",
            coordinates_system=CoordinatesSystem.PARENT,
        ),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs, execution_cache=execution_cache
    )

    # then
    assert len(result) == 2, "Expected exactly 2 outputs, as per outputs definitions"
    assert (
        result["a_own"]
        == result["a_parent"]
        == [
            {"predictions": ["a", "b"], "foo": 1},
            {"predictions": ["c", "d"], "foo": 2},
        ]
    ), (
        "Expected both registered outputs to hold the same data, as coordinates system choice should not affect "
        "non-detections outputs"
    )


def test_construct_response_when_detections_to_be_returned_in_own_coordinates() -> None:
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [20, 30, 40, 50]]),
        mask=np.ones((2, 100, 100)).astype(np.bool_),
        confidence=np.array([0.3, 0.4]),
        class_id=np.array([0, 1]),
        tracker_id=None,
        data={
            "class_name": np.array(["a", "b"]),
            "keypoints_class_id": np.array(
                [np.array([]), np.array([0, 0, 1])], dtype="object"
            ),
            "keypoints_class_name": np.array(
                [np.array([]), np.array(["0", "0", "1"])], dtype="object"
            ),
            "keypoints_confidence": np.array(
                [np.array([]), np.array([0.3, 0.4, 0.5])], dtype="object"
            ),
            "keypoints_xy": np.array(
                [np.array([]), np.array([[10, 20], [15, 25], [20, 30]])], dtype="object"
            ),
            "prediction_type": np.array(["test-prediction", "test-prediction"]),
            "parent_coordinates": np.array([[0, 0], [0, 0]]),
            "parent_dimensions": np.array([[100, 100], [100, 100]]),
            "root_parent_coordinates": np.array([[50, 50], [100, 100]]),
            "root_parent_dimensions": np.array([[200, 200], [200, 200]]),
        },
    )
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {"predictions": detections},
            {"predictions": detections},
        ],
    )
    workflow_outputs = [
        JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.predictions",
            coordinates_system=CoordinatesSystem.OWN,
        ),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs, execution_cache=execution_cache
    )

    # then
    assert result == {
        "some": [detections, detections]
    }, "Expected output to be passed without modifications"


def test_construct_response_when_detections_to_be_returned_in_own_coordinates_using_wildcard() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [20, 30, 40, 50]]),
        mask=np.ones((2, 100, 100)).astype(np.bool_),
        confidence=np.array([0.3, 0.4]),
        class_id=np.array([0, 1]),
        tracker_id=None,
        data={
            "class_name": np.array(["a", "b"]),
            "keypoints_class_id": np.array(
                [np.array([]), np.array([0, 0, 1])], dtype="object"
            ),
            "keypoints_class_name": np.array(
                [np.array([]), np.array(["0", "0", "1"])], dtype="object"
            ),
            "keypoints_confidence": np.array(
                [np.array([]), np.array([0.3, 0.4, 0.5])], dtype="object"
            ),
            "keypoints_xy": np.array(
                [np.array([]), np.array([[10, 20], [15, 25], [20, 30]])], dtype="object"
            ),
            "prediction_type": np.array(["test-prediction", "test-prediction"]),
            "parent_coordinates": np.array([[0, 0], [0, 0]]),
            "parent_dimensions": np.array([[100, 100], [100, 100]]),
            "root_parent_coordinates": np.array([[50, 50], [100, 100]]),
            "root_parent_dimensions": np.array([[200, 200], [200, 200]]),
        },
    )
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {"predictions": detections},
            {"predictions": detections},
        ],
    )
    workflow_outputs = [
        JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.*",
            coordinates_system=CoordinatesSystem.OWN,
        ),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs, execution_cache=execution_cache
    )

    # then
    assert result == {
        "some": [{"predictions": detections}, {"predictions": detections}]
    }, "Expected output to be passed without modifications"


def test_construct_response_when_detections_to_be_returned_in_parent_coordinates() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [20, 30, 40, 50]]),
        mask=np.ones((2, 100, 100)).astype(np.bool_),
        confidence=np.array([0.3, 0.4]),
        class_id=np.array([0, 1]),
        tracker_id=None,
        data={
            "class_name": np.array(["a", "b"]),
            "keypoints_class_id": np.array(
                [np.array([]), np.array([0, 0, 1])], dtype="object"
            ),
            "keypoints_class_name": np.array(
                [np.array([]), np.array(["0", "0", "1"])], dtype="object"
            ),
            "keypoints_confidence": np.array(
                [np.array([]), np.array([0.3, 0.4, 0.5])], dtype="object"
            ),
            "keypoints_xy": np.array(
                [np.array([]), np.array([[10, 20], [15, 25], [20, 30]])], dtype="object"
            ),
            "prediction_type": np.array(["test-prediction", "test-prediction"]),
            "parent_coordinates": np.array([[0, 0], [0, 0]]),
            "parent_id": np.array(["parent-1", "parent-1"]),
            "parent_dimensions": np.array([[100, 100], [100, 100]]),
            "root_parent_coordinates": np.array([[50, 100], [50, 100]]),
            "root_parent_id": np.array(["root-parent-1", "root-parent-1"]),
            "root_parent_dimensions": np.array([[200, 200], [200, 200]]),
        },
    )
    expected_masks = [
        np.zeros((200, 200)).astype(np.bool_),
        np.zeros((200, 200)).astype(np.bool_),
    ]
    expected_masks[0][
        100:200, 50:150
    ] = True  # previous mask was (100, 100) all True, we shift OX: 50, OY: 100
    expected_masks[1][
        100:200, 50:150
    ] = True  # previous mask was (100, 100) all True, we shift OX: 50, OY: 100
    expected_detections_in_parent_coordinates = sv.Detections(
        xyxy=np.array(
            [
                [50 + 10, 100 + 20, 50 + 30, 100 + 40],  # we shift OX: 50, OY: 100
                [50 + 20, 100 + 30, 50 + 40, 100 + 50],  # we shift OX: 50, OY: 100
            ]
        ),
        mask=np.array(expected_masks),
        confidence=np.array([0.3, 0.4]),
        class_id=np.array([0, 1]),
        tracker_id=None,
        data={
            "class_name": np.array(["a", "b"]),
            "keypoints_class_id": np.array(
                [np.array([]), np.array([0, 0, 1])], dtype="object"
            ),
            "keypoints_class_name": np.array(
                [np.array([]), np.array(["0", "0", "1"])], dtype="object"
            ),
            "keypoints_confidence": np.array(
                [np.array([]), np.array([0.3, 0.4, 0.5])], dtype="object"
            ),
            "keypoints_xy": np.array(
                [
                    np.array([]),
                    np.array(
                        [[50 + 10, 100 + 20], [50 + 15, 100 + 25], [50 + 20, 100 + 30]]
                    ),  # we shift OX: 50, OY: 100
                ],
                dtype="object",
            ),
            "prediction_type": np.array(["test-prediction", "test-prediction"]),
            "parent_id": np.array(
                ["root-parent-1", "root-parent-1"]
            ),  # substituting with old root values
            "parent_coordinates": np.array(
                [[0, 0], [0, 0]]
            ),  # substituting with old root values
            "parent_dimensions": np.array(
                [[200, 200], [200, 200]]
            ),  # substituting with old root values
            "root_parent_coordinates": np.array(
                [[0, 0], [0, 0]]
            ),  # substituting with old root values
            "root_parent_dimensions": np.array(
                [[200, 200], [200, 200]]
            ),  # substituting with old root values
            "root_parent_id": np.array(
                ["root-parent-1", "root-parent-1"]
            ),  # substituting with old root values
        },
    )
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {"predictions": detections},
        ],
    )
    workflow_outputs = [
        JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.predictions",
            coordinates_system=CoordinatesSystem.PARENT,
        ),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs, execution_cache=execution_cache
    )

    # then
    assert len(result) == 1, "Oly single output is expected"
    assert (
        len(result["some"]) == 1
    ), "Only single element in batch output of `some` expected"
    assert np.allclose(
        result["some"][0].xyxy, expected_detections_in_parent_coordinates.xyxy
    ), "Expected xyxy output to be shifted into parents coordinates"
    assert (
        len(result["some"][0]["keypoints_xy"][0]) == 0
    ), "Expected empty keypoints to remain empty"
    assert np.allclose(
        result["some"][0]["keypoints_xy"][1],
        expected_detections_in_parent_coordinates["keypoints_xy"][1],
    ), "Expected keypoints coordinates to be shifted into parents coordinates"
    assert np.allclose(
        result["some"][0].mask, expected_detections_in_parent_coordinates.mask
    ), "Expected masks to be transformed"
    assert np.allclose(
        result["some"][0]["parent_coordinates"],
        expected_detections_in_parent_coordinates["parent_coordinates"],
    ), "Expected parent_coordinates to be zeroed"
    assert np.allclose(
        result["some"][0]["parent_dimensions"],
        expected_detections_in_parent_coordinates["parent_dimensions"],
    ), "Expected parent_dimensions to be set into original root dimensions"
    assert np.allclose(
        result["some"][0]["root_parent_coordinates"],
        expected_detections_in_parent_coordinates["root_parent_coordinates"],
    ), "Expected root_parent_coordinates to be zeroed"
    assert np.allclose(
        result["some"][0]["root_parent_dimensions"],
        expected_detections_in_parent_coordinates["root_parent_dimensions"],
    ), "Expected root_parent_dimensions to be set into original root dimensions"
    assert (
        result["some"][0]["root_parent_id"].tolist()
        == result["some"][0]["parent_id"].tolist()
        == expected_detections_in_parent_coordinates["parent_id"].tolist()
    ), "Expected parent ids to point to root parent"


def test_construct_response_when_detections_to_be_returned_in_parent_coordinates_using_wildcard() -> (
    None
):
    # given
    detections = sv.Detections(
        xyxy=np.array([[10, 20, 30, 40], [20, 30, 40, 50]]),
        mask=np.ones((2, 100, 100)).astype(np.bool_),
        confidence=np.array([0.3, 0.4]),
        class_id=np.array([0, 1]),
        tracker_id=None,
        data={
            "class_name": np.array(["a", "b"]),
            "keypoints_class_id": np.array(
                [np.array([]), np.array([0, 0, 1])], dtype="object"
            ),
            "keypoints_class_name": np.array(
                [np.array([]), np.array(["0", "0", "1"])], dtype="object"
            ),
            "keypoints_confidence": np.array(
                [np.array([]), np.array([0.3, 0.4, 0.5])], dtype="object"
            ),
            "keypoints_xy": np.array(
                [np.array([]), np.array([[10, 20], [15, 25], [20, 30]])], dtype="object"
            ),
            "prediction_type": np.array(["test-prediction", "test-prediction"]),
            "parent_coordinates": np.array([[0, 0], [0, 0]]),
            "parent_id": np.array(["parent-1", "parent-1"]),
            "parent_dimensions": np.array([[100, 100], [100, 100]]),
            "root_parent_coordinates": np.array([[50, 100], [50, 100]]),
            "root_parent_id": np.array(["root-parent-1", "root-parent-1"]),
            "root_parent_dimensions": np.array([[200, 200], [200, 200]]),
        },
    )
    expected_masks = [
        np.zeros((200, 200)).astype(np.bool_),
        np.zeros((200, 200)).astype(np.bool_),
    ]
    expected_masks[0][
        100:200, 50:150
    ] = True  # previous mask was (100, 100) all True, we shift OX: 50, OY: 100
    expected_masks[1][
        100:200, 50:150
    ] = True  # previous mask was (100, 100) all True, we shift OX: 50, OY: 100
    expected_detections_in_parent_coordinates = sv.Detections(
        xyxy=np.array(
            [
                [50 + 10, 100 + 20, 50 + 30, 100 + 40],  # we shift OX: 50, OY: 100
                [50 + 20, 100 + 30, 50 + 40, 100 + 50],  # we shift OX: 50, OY: 100
            ]
        ),
        mask=np.array(expected_masks),
        confidence=np.array([0.3, 0.4]),
        class_id=np.array([0, 1]),
        tracker_id=None,
        data={
            "class_name": np.array(["a", "b"]),
            "keypoints_class_id": np.array(
                [np.array([]), np.array([0, 0, 1])], dtype="object"
            ),
            "keypoints_class_name": np.array(
                [np.array([]), np.array(["0", "0", "1"])], dtype="object"
            ),
            "keypoints_confidence": np.array(
                [np.array([]), np.array([0.3, 0.4, 0.5])], dtype="object"
            ),
            "keypoints_xy": np.array(
                [
                    np.array([]),
                    np.array(
                        [[50 + 10, 100 + 20], [50 + 15, 100 + 25], [50 + 20, 100 + 30]]
                    ),  # we shift OX: 50, OY: 100
                ],
                dtype="object",
            ),
            "prediction_type": np.array(["test-prediction", "test-prediction"]),
            "parent_id": np.array(
                ["root-parent-1", "root-parent-1"]
            ),  # substituting with old root values
            "parent_coordinates": np.array(
                [[0, 0], [0, 0]]
            ),  # substituting with old root values
            "parent_dimensions": np.array(
                [[200, 200], [200, 200]]
            ),  # substituting with old root values
            "root_parent_coordinates": np.array(
                [[0, 0], [0, 0]]
            ),  # substituting with old root values
            "root_parent_dimensions": np.array(
                [[200, 200], [200, 200]]
            ),  # substituting with old root values
            "root_parent_id": np.array(
                ["root-parent-1", "root-parent-1"]
            ),  # substituting with old root values
        },
    )
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {"predictions": detections},
        ],
    )
    workflow_outputs = [
        JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.*",
            coordinates_system=CoordinatesSystem.PARENT,
        ),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs, execution_cache=execution_cache
    )
    # then
    assert len(result) == 1, "Oly single output is expected"
    assert (
        len(result["some"]) == 1
    ), "Only single element in batch output of `some` expected"
    assert np.allclose(
        result["some"][0]["predictions"].xyxy,
        expected_detections_in_parent_coordinates.xyxy,
    ), "Expected xyxy output to be shifted into parents coordinates"
    assert (
        len(result["some"][0]["predictions"]["keypoints_xy"][0]) == 0
    ), "Expected empty keypoints to remain empty"
    assert np.allclose(
        result["some"][0]["predictions"]["keypoints_xy"][1],
        expected_detections_in_parent_coordinates["keypoints_xy"][1],
    ), "Expected keypoints coordinates to be shifted into parents coordinates"
    assert np.allclose(
        result["some"][0]["predictions"].mask,
        expected_detections_in_parent_coordinates.mask,
    ), "Expected masks to be transformed"
    assert np.allclose(
        result["some"][0]["predictions"]["parent_coordinates"],
        expected_detections_in_parent_coordinates["parent_coordinates"],
    ), "Expected parent_coordinates to be zeroed"
    assert np.allclose(
        result["some"][0]["predictions"]["parent_dimensions"],
        expected_detections_in_parent_coordinates["parent_dimensions"],
    ), "Expected parent_dimensions to be set into original root dimensions"
    assert np.allclose(
        result["some"][0]["predictions"]["root_parent_coordinates"],
        expected_detections_in_parent_coordinates["root_parent_coordinates"],
    ), "Expected root_parent_coordinates to be zeroed"
    assert np.allclose(
        result["some"][0]["predictions"]["root_parent_dimensions"],
        expected_detections_in_parent_coordinates["root_parent_dimensions"],
    ), "Expected root_parent_dimensions to be set into original root dimensions"
    assert (
        result["some"][0]["predictions"]["root_parent_id"].tolist()
        == result["some"][0]["predictions"]["parent_id"].tolist()
        == expected_detections_in_parent_coordinates["parent_id"].tolist()
    ), "Expected parent ids to point to root parent"


def test_construct_response_when_step_output_is_missing_due_to_conditional_execution() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    workflow_outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.predictions"),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs, execution_cache=execution_cache
    )

    # then
    assert result == {"some": []}


def test_construct_response_when_expected_step_property_is_missing() -> None:
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    workflow_outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.non_existing"),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs, execution_cache=execution_cache
    )

    # then
    assert result == {"some": []}
