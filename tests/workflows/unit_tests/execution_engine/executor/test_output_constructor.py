from typing import Any, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv
from networkx import DiGraph

from inference.core.workflows.core_steps.loader import KINDS_SERIALIZERS
from inference.core.workflows.errors import AssumptionError, ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.entities.base import JsonField
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    INTEGER_KIND,
    STRING_KIND,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    NodeCategory,
    OutputNode,
)
from inference.core.workflows.execution_engine.v1.executor.output_constructor import (
    construct_workflow_output,
    convert_sv_detections_coordinates,
    create_array,
    data_contains_sv_detections,
    place_data_in_array,
    serialize_data_piece,
)


def test_data_contains_sv_detections_when_no_sv_detections_provided() -> None:
    # given
    data = {
        "some": 1,
        "other": [1, 3, 4],
        "yet_another": "a",
    }

    # when
    result = data_contains_sv_detections(data=data)

    # then
    assert result is False


def test_data_contains_sv_detections_when_sv_detections_provided_directly() -> None:
    # given
    data = sv.Detections.empty()

    # when
    result = data_contains_sv_detections(data=data)

    # then
    assert result is True


def test_data_contains_sv_detections_when_sv_detections_provided_in_dict() -> None:
    # given
    data = {"some": "a", "other": {"some": sv.Detections.empty()}}

    # when
    result = data_contains_sv_detections(data=data)

    # then
    assert result is True


def test_convert_sv_detections_coordinates_when_sv_detections_provided_directly() -> (
    None
):
    # given
    data = assembly_sv_detections()

    # when
    result = convert_sv_detections_coordinates(data=data)

    # then
    assert_transformed_detections_matches_expectation(result=result)


def test_convert_sv_detections_coordinates_when_sv_detections_provided_in_list() -> (
    None
):
    # given
    data = [assembly_sv_detections(), 1, 2, 3]

    # when
    result = convert_sv_detections_coordinates(data=data)

    # then
    assert len(result) == 4
    assert_transformed_detections_matches_expectation(result=result[0])
    assert result[1:] == [1, 2, 3]


def test_convert_sv_detections_coordinates_when_sv_detections_provided_in_nested_list() -> (
    None
):
    # given
    data = [[assembly_sv_detections()], 1, 2, 3]

    # when
    result = convert_sv_detections_coordinates(data=data)

    # then
    assert len(result) == 4
    assert_transformed_detections_matches_expectation(result=result[0][0])
    assert result[1:] == [1, 2, 3]


def test_convert_sv_detections_coordinates_when_sv_detections_provided_in_dict() -> (
    None
):
    # given
    data = {"a": assembly_sv_detections(), "b": "some"}

    # when
    result = convert_sv_detections_coordinates(data=data)

    # then
    assert len(result) == 2
    assert result["b"] == "some"
    assert_transformed_detections_matches_expectation(result=result["a"])


def test_convert_sv_detections_coordinates_when_sv_detections_provided_in_nested_dict() -> (
    None
):
    # given
    data = {"a": {"nested": assembly_sv_detections()}, "b": "some"}

    # when
    result = convert_sv_detections_coordinates(data=data)

    # then
    assert len(result) == 2
    assert result["b"] == "some"
    assert_transformed_detections_matches_expectation(result=result["a"]["nested"])


def assembly_sv_detections() -> sv.Detections:
    return sv.Detections(
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


def assert_transformed_detections_matches_expectation(
    result: sv.Detections,
) -> None:
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
    assert np.allclose(
        result.xyxy, expected_detections_in_parent_coordinates.xyxy
    ), "Expected xyxy output to be shifted into parents coordinates"
    assert (
        len(result["keypoints_xy"][0]) == 0
    ), "Expected empty keypoints to remain empty"
    assert np.allclose(
        result["keypoints_xy"][1],
        expected_detections_in_parent_coordinates["keypoints_xy"][1],
    ), "Expected keypoints coordinates to be shifted into parents coordinates"
    assert np.allclose(
        result.mask, expected_detections_in_parent_coordinates.mask
    ), "Expected masks to be transformed"
    assert np.allclose(
        result["parent_coordinates"],
        expected_detections_in_parent_coordinates["parent_coordinates"],
    ), "Expected parent_coordinates to be zeroed"
    assert np.allclose(
        result["parent_dimensions"],
        expected_detections_in_parent_coordinates["parent_dimensions"],
    ), "Expected parent_dimensions to be set into original root dimensions"
    assert np.allclose(
        result["root_parent_coordinates"],
        expected_detections_in_parent_coordinates["root_parent_coordinates"],
    ), "Expected root_parent_coordinates to be zeroed"
    assert np.allclose(
        result["root_parent_dimensions"],
        expected_detections_in_parent_coordinates["root_parent_dimensions"],
    ), "Expected root_parent_dimensions to be set into original root dimensions"
    assert (
        result["root_parent_id"].tolist()
        == result["parent_id"].tolist()
        == expected_detections_in_parent_coordinates["parent_id"].tolist()
    ), "Expected parent ids to point to root parent"


def test_place_data_in_array() -> None:
    # given
    array = [
        [None, None, None],
        [None, None],
        [],
    ]

    # when
    place_data_in_array(array=array, index=(1, 1), data=3)

    # then
    assert array == [
        [None, None, None],
        [None, 3],
        [],
    ]


def test_place_data_in_array_under_non_existing_index() -> None:
    # given
    array = [
        [None, None, None],
        [None, None],
        [],
    ]

    # when
    with pytest.raises(IndexError):
        place_data_in_array(array=array, index=(1, 2), data=3)


def test_place_data_in_array_when_dimensionality_missmatch_happens() -> None:
    # given
    array = [
        [None, None, None],
        [None, None],
        [],
    ]

    # when
    with pytest.raises(TypeError):
        place_data_in_array(array=array, index=(1, 1, 1), data=3)


def test_create_array_for_dimension_one() -> None:
    # when
    result = create_array(indices=np.array([(0,), (1,), (4,), (6,)]))

    # then
    assert result == [None] * 7


def test_create_array_for_dimension_two() -> None:
    # when
    result = create_array(
        indices=np.array(
            [(0, 0), (0, 2), (1, 0), (4, 0), (4, 1), (4, 2), (6, 0), (6, 1)]
        )
    )

    # then
    assert result == [
        [None, None, None],  # 3 elements for (0, )
        [None],  # 1 element for (1, )
        [],  # no elements for (2, )
        [],  # no elements for (3, )
        [None, None, None],  # 3 elements for (4, )
        [],  # no elements for (5, )
        [None, None],  # 2 elements for (6, )
    ]


def test_create_array_for_dimension_three() -> None:
    # when
    result = create_array(
        indices=np.array(
            [
                (0, 0, 0),
                (0, 0, 1),
                (0, 2, 0),
                (0, 2, 3),
                (1, 0, 0),
                (4, 0, 0),
                (4, 0, 2),
                (4, 1, 0),
                (4, 2, 0),
                (4, 2, 1),
                (6, 0, 0),
                (6, 1, 3),
            ]
        )
    )

    # then
    assert result == [
        [[None, None], [], [None, None, None, None]],
        [[None]],
        [[]],  # no elements for (2, )
        [[]],  # no elements for (3, )
        [[None, None, None], [None], [None, None]],
        [[]],  # no elements for (5, )
        [[None], [None, None, None, None]],
    ]


def test_construct_workflow_output_when_no_batch_outputs_present() -> None:
    # given
    execution_data_manager = MagicMock()
    execution_data_manager.get_selector_indices.return_value = None
    workflow_outputs = [
        JsonField(type="JsonField", name="a", selector="$steps.some.a"),
        JsonField(type="JsonField", name="b", selector="$inputs.b"),
    ]
    execution_graph = DiGraph()
    execution_graph.add_node(
        "$outputs.a",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name=workflow_outputs[0].name,
            selector=workflow_outputs[0].selector,
            data_lineage=[],
            output_manifest=workflow_outputs[0],
        ),
    )
    execution_graph.add_node(
        "$outputs.b",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name=workflow_outputs[1].name,
            selector=workflow_outputs[1].selector,
            data_lineage=[],
            output_manifest=workflow_outputs[1],
        ),
    )
    data_lookup = {
        "$steps.some.a": "a_value",
        "$inputs.b": "b_value",
    }

    def get_non_batch_data(selector: str) -> Any:
        return data_lookup[selector]

    execution_data_manager.get_non_batch_data = get_non_batch_data

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs,
        execution_graph=execution_graph,
        execution_data_manager=execution_data_manager,
        serialize_results=True,
        kinds_serializers=KINDS_SERIALIZERS,
    )

    # then
    assert result == [{"a": "a_value", "b": "b_value"}]


def test_construct_workflow_output_when_batch_outputs_present() -> None:
    # given
    execution_data_manager = MagicMock()
    workflow_outputs = [
        JsonField(type="JsonField", name="a", selector="$steps.some.a"),
        JsonField(type="JsonField", name="b", selector="$steps.some.b"),
        JsonField(type="JsonField", name="b_empty", selector="$steps.some.b_empty"),
        JsonField(
            type="JsonField",
            name="b_empty_nested",
            selector="$steps.some.b_empty_nested",
        ),
        JsonField(type="JsonField", name="c", selector="$steps.other.c"),
    ]
    execution_graph = DiGraph()
    execution_graph.add_node(
        "$outputs.a",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name=workflow_outputs[0].name,
            selector=workflow_outputs[0].selector,
            data_lineage=["<workflow_input>"],
            output_manifest=workflow_outputs[0],
        ),
    )
    execution_graph.add_node(
        "$outputs.b",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name=workflow_outputs[1].name,
            selector=workflow_outputs[1].selector,
            data_lineage=["<workflow_input>", "some"],
            output_manifest=workflow_outputs[1],
        ),
    )
    execution_graph.add_node(
        "$outputs.b_empty",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name=workflow_outputs[2].name,
            selector=workflow_outputs[2].selector,
            data_lineage=["<workflow_input>"],
            output_manifest=workflow_outputs[2],
        ),
    )
    execution_graph.add_node(
        "$outputs.b_empty_nested",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name=workflow_outputs[3].name,
            selector=workflow_outputs[3].selector,
            data_lineage=["<workflow_input>", "other", "yet_another"],
            output_manifest=workflow_outputs[3],
        ),
    )
    execution_graph.add_node(
        "$outputs.c",
        node_compilation_output=OutputNode(
            node_category=NodeCategory.OUTPUT_NODE,
            name=workflow_outputs[4].name,
            selector=workflow_outputs[4].selector,
            data_lineage=[],
            output_manifest=workflow_outputs[4],
        ),
    )
    data_lookup = {
        "$steps.other.c": "c_value",
    }

    def get_non_batch_data(selector: str) -> Any:
        return data_lookup[selector]

    execution_data_manager.get_non_batch_data = get_non_batch_data

    indices_lookup = {
        "$steps.some.a": [(0,), (2,)],
        "$steps.some.b": [(0, 0), (0, 1), (1, 0), (2, 3)],
        "$steps.some.b_empty": [],
        "$steps.some.b_empty_nested": [],
        "$steps.other.c": None,
    }

    def get_selector_indices(selector: str) -> Optional[List[tuple]]:
        return indices_lookup[selector]

    execution_data_manager.get_selector_indices = get_selector_indices

    batch_data_lookup = {
        "$steps.some.a": {(0,): "a1", (2,): "a2"},
        "$steps.some.b": {
            (0, 0): "b1",
            (0, 1): "b2",
            (1, 0): "b3",
            (2, 3): "b4",
        },
        "$steps.some.b_empty": {},
        "$steps.some.b_empty_nested": {},
    }

    def get_batch_data(selector: str, indices: List[tuple]) -> List[Any]:
        return [batch_data_lookup[selector][index] for index in indices]

    execution_data_manager.get_batch_data = get_batch_data
    execution_data_manager.get_lineage_indices.return_value = [(0,), (1,), (2,)]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs,
        execution_graph=execution_graph,
        execution_data_manager=execution_data_manager,
        serialize_results=True,
        kinds_serializers=KINDS_SERIALIZERS,
    )

    # then
    assert (
        len(result) == 3
    ), "Expected 3 results, as that the highest size of 1-dim batch"
    assert result[0] == {
        "a": "a1",
        "b": ["b1", "b2"],
        "c": "c_value",
        "b_empty": None,
        "b_empty_nested": [[]],
    }
    assert result[1] == {
        "a": None,
        "b": ["b3"],
        "c": "c_value",
        "b_empty": None,
        "b_empty_nested": [[]],
    }
    assert result[2] == {
        "a": "a2",
        "b": [None, None, None, "b4"],
        "c": "c_value",
        "b_empty": None,
        "b_empty_nested": [[]],
    }


def test_serialize_data_piece_for_wildcard_output_when_serializer_not_found() -> None:
    # when
    result = serialize_data_piece(
        output_name="my_output",
        data_piece={"some": "data", "other": "another"},
        kind={"some": [STRING_KIND], "other": [STRING_KIND]},
        kinds_serializers={},
    )

    # then
    assert result == {"some": "data", "other": "another"}, "Expected data not t0 change"


def test_serialize_data_piece_for_wildcard_output_when_missmatch_in_input_detected() -> (
    None
):
    # when
    with pytest.raises(AssumptionError):
        _ = serialize_data_piece(
            output_name="my_output",
            data_piece="not a dict",
            kind={"some": [STRING_KIND], "other": [STRING_KIND]},
            kinds_serializers={},
        )


def test_serialize_data_piece_for_wildcard_output_when_serializers_found_but_all_failing() -> (
    None
):
    # given
    def _faulty_serializer(value: Any) -> Any:
        raise Exception()

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = serialize_data_piece(
            output_name="my_output",
            data_piece={"some": "data", "other": "another"},
            kind={"some": [STRING_KIND, INTEGER_KIND], "other": STRING_KIND},
            kinds_serializers={
                STRING_KIND.name: _faulty_serializer,
                INTEGER_KIND.name: _faulty_serializer,
            },
        )


def test_serialize_data_piece_for_wildcard_output_when_serializers_found_with_one_failing_and_one_successful() -> (
    None
):
    # given
    faulty_calls = []

    def _faulty_serializer(value: Any) -> Any:
        faulty_calls.append(1)
        raise Exception()

    def _valid_serializer(value: Any) -> Any:
        return "serialized", value

    # when
    result = serialize_data_piece(
        output_name="my_output",
        data_piece={"some": "data", "other": "another"},
        kind={"some": [INTEGER_KIND, STRING_KIND], "other": [STRING_KIND]},
        kinds_serializers={
            STRING_KIND.name: _valid_serializer,
            INTEGER_KIND.name: _faulty_serializer,
        },
    )

    # then
    assert len(faulty_calls) == 1, "Expected faulty serializer attempted"
    assert result == {
        "some": ("serialized", "data"),
        "other": ("serialized", "another"),
    }


def test_serialize_data_piece_for_wildcard_output_when_serializers_found_and_successful() -> (
    None
):
    # given
    def _valid_serializer(value: Any) -> Any:
        return "serialized", value

    # when
    result = serialize_data_piece(
        output_name="my_output",
        data_piece={"some": "data", "other": "another"},
        kind={"some": [INTEGER_KIND, STRING_KIND], "other": [STRING_KIND]},
        kinds_serializers={
            STRING_KIND.name: _valid_serializer,
            INTEGER_KIND.name: _valid_serializer,
        },
    )

    # then
    assert result == {
        "some": ("serialized", "data"),
        "other": ("serialized", "another"),
    }


def test_serialize_data_piece_for_specific_output_when_serializer_not_found() -> None:
    # when
    result = serialize_data_piece(
        output_name="my_output",
        data_piece="data",
        kind=[STRING_KIND],
        kinds_serializers={},
    )

    # then
    assert result == "data", "Expected data not to change"


def test_serialize_data_piece_for_specific_output_when_serializers_found_but_all_failing() -> (
    None
):
    # given
    def _faulty_serializer(value: Any) -> Any:
        raise Exception()

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = serialize_data_piece(
            output_name="my_output",
            data_piece="data",
            kind=[STRING_KIND, INTEGER_KIND],
            kinds_serializers={
                STRING_KIND.name: _faulty_serializer,
                INTEGER_KIND.name: _faulty_serializer,
            },
        )


def test_serialize_data_piece_for_specific_output_when_serializers_found_with_one_failing_and_one_successful() -> (
    None
):
    # given
    faulty_calls = []

    def _faulty_serializer(value: Any) -> Any:
        faulty_calls.append(1)
        raise Exception()

    def _valid_serializer(value: Any) -> Any:
        return "serialized", value

    # when
    result = serialize_data_piece(
        output_name="my_output",
        data_piece="data",
        kind=[INTEGER_KIND, STRING_KIND],
        kinds_serializers={
            STRING_KIND.name: _valid_serializer,
            INTEGER_KIND.name: _faulty_serializer,
        },
    )

    # then
    assert len(faulty_calls) == 1, "Expected faulty serializer attempted"
    assert result == ("serialized", "data")


def test_serialize_data_piece_for_specific_output_when_serializers_found_and_successful() -> (
    None
):
    # given
    def _valid_serializer(value: Any) -> Any:
        return "serialized", value

    # when
    result = serialize_data_piece(
        output_name="my_output",
        data_piece="data",
        kind=[INTEGER_KIND, STRING_KIND],
        kinds_serializers={
            STRING_KIND.name: _valid_serializer,
            INTEGER_KIND.name: _valid_serializer,
        },
    )

    # then
    assert result == ("serialized", "data")
