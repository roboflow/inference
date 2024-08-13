from typing import Any

import pytest

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.step_input_assembler import (
    GuardForIndicesWrapping,
    ensure_compound_input_indices_match,
    get_empty_batch_elements_indices,
    reduce_batch_dimensionality,
    remove_indices,
    unfold_parameters,
)


def test_unfold_parameters_when_only_non_batches_parameters_given() -> None:
    # given
    parameters = {
        "some": "a",
        "other": 1,
        "my_list": [1, 2, 3],
        "my_dict": {"a": -1, "b": -2},
    }

    # when
    result = list(unfold_parameters(parameters=parameters))

    # then
    assert len(result) == 1, "Expected single element to be yield"
    assert result[0] == {
        "some": "a",
        "other": 1,
        "my_list": [1, 2, 3],
        "my_dict": {"a": -1, "b": -2},
    }, "Expected to see the same values as in input"


def test_unfold_parameters_when_batch_parameter_given_in_non_compound_form() -> None:
    # given
    parameters = {
        "some": "a",
        "other": Batch(content=[10, 20, 30], indices=[(0, 0), (0, 1), (0, 2)]),
        "my_list": [1, 2, 3],
        "my_dict": {"a": -1, "b": -2},
    }

    # when
    result = list(unfold_parameters(parameters=parameters))

    # then
    assert len(result) == 3, "Expected three elements as this is the size of batch"
    assert result[0] == {
        "some": "a",
        "other": 10,
        "my_list": [1, 2, 3],
        "my_dict": {"a": -1, "b": -2},
    }, "Expected to first batch element and other elements broadcast"
    assert result[1] == {
        "some": "a",
        "other": 20,
        "my_list": [1, 2, 3],
        "my_dict": {"a": -1, "b": -2},
    }, "Expected to second batch element and other elements broadcast"
    assert result[2] == {
        "some": "a",
        "other": 30,
        "my_list": [1, 2, 3],
        "my_dict": {"a": -1, "b": -2},
    }, "Expected to third batch element and other elements broadcast"


def test_unfold_parameters_when_batch_parameter_given_in_list() -> None:
    # given
    parameters = {
        "some": "a",
        "my_list": [
            Batch(content=[10, 20, 30], indices=[(0, 0), (0, 1), (0, 2)]),
            Batch(content=[11, 21, 31], indices=[(0, 0), (0, 1), (0, 2)]),
            Batch(content=[12, 22, 32], indices=[(0, 0), (0, 1), (0, 2)]),
        ],
        "my_dict": {"a": -1, "b": -2},
    }

    # when
    result = list(unfold_parameters(parameters=parameters))

    # then
    assert (
        len(result) == 3
    ), "Expected three elements as this is the size of batch each batch"
    assert result[0] == {
        "some": "a",
        "my_list": [10, 11, 12],
        "my_dict": {"a": -1, "b": -2},
    }, "Expected to first batches element and other elements broadcast"
    assert result[1] == {
        "some": "a",
        "my_list": [20, 21, 22],
        "my_dict": {"a": -1, "b": -2},
    }, "Expected to second batch elements and other elements broadcast"
    assert result[2] == {
        "some": "a",
        "my_list": [30, 31, 32],
        "my_dict": {"a": -1, "b": -2},
    }, "Expected to third batch elements and other elements broadcast"


def test_unfold_parameters_when_batch_parameter_given_in_dict() -> None:
    # given
    parameters = {
        "some": "a",
        "my_list": [1, 2, 3],
        "my_dict": {
            "a": Batch(content=[10, 20, 30], indices=[(0, 0), (0, 1), (0, 2)]),
            "b": Batch(content=[11, 21, 31], indices=[(0, 0), (0, 1), (0, 2)]),
        },
    }

    # when
    result = list(unfold_parameters(parameters=parameters))

    # then
    assert (
        len(result) == 3
    ), "Expected three elements as this is the size of batch each batch"
    assert result[0] == {
        "some": "a",
        "my_list": [1, 2, 3],
        "my_dict": {"a": 10, "b": 11},
    }, "Expected to first batches element and other elements broadcast"
    assert result[1] == {
        "some": "a",
        "my_list": [1, 2, 3],
        "my_dict": {"a": 20, "b": 21},
    }, "Expected to second batch elements and other elements broadcast"
    assert result[2] == {
        "some": "a",
        "my_list": [1, 2, 3],
        "my_dict": {"a": 30, "b": 31},
    }, "Expected to third batch elements and other elements broadcast"


def test_unfold_parameters_when_batch_parameter_given_in_primitive_parameter_list_and_dict() -> (
    None
):
    # given
    parameters = {
        "some": "static_value",
        "my_batch": Batch(content=["0", "1", "2"], indices=[(0, 0), (0, 1), (0, 2)]),
        "my_list": [
            Batch(content=["a", "c", "e"], indices=[(0, 0), (0, 1), (0, 2)]),
            Batch(content=["b", "d", "f"], indices=[(0, 0), (0, 1), (0, 2)]),
        ],
        "my_dict": {
            "a": Batch(content=[10, 20, 30], indices=[(0, 0), (0, 1), (0, 2)]),
            "b": Batch(content=[11, 21, 31], indices=[(0, 0), (0, 1), (0, 2)]),
        },
    }

    # when
    result = list(unfold_parameters(parameters=parameters))

    # then
    assert (
        len(result) == 3
    ), "Expected three elements as this is the size of batch each batch"
    assert result[0] == {
        "some": "static_value",
        "my_batch": "0",
        "my_list": ["a", "b"],
        "my_dict": {"a": 10, "b": 11},
    }, "Expected to first batches element and other elements broadcast"
    assert result[1] == {
        "some": "static_value",
        "my_batch": "1",
        "my_list": ["c", "d"],
        "my_dict": {"a": 20, "b": 21},
    }, "Expected to second batch elements and other elements broadcast"
    assert result[2] == {
        "some": "static_value",
        "my_batch": "2",
        "my_list": ["e", "f"],
        "my_dict": {"a": 30, "b": 31},
    }, "Expected to third batch elements and other elements broadcast"


def test_unfold_parameters_when_batch_parameter_given_in_primitive_parameter_list_and_dict_and_there_is_length_missmatch_in_batches() -> (
    None
):
    # given
    parameters = {
        "some": "static_value",
        "my_batch": Batch(content=["0", "1", "2"], indices=[(0, 0), (0, 1), (0, 2)]),
        "my_list": [
            Batch(content=["a", "c", "e"], indices=[(0, 0), (0, 1), (0, 2)]),
            Batch(content=["b", "d"], indices=[(0, 0), (0, 1)]),
        ],
        "my_dict": {
            "a": Batch(content=[10, 20, 30], indices=[(0, 0), (0, 1), (0, 2)]),
            "b": Batch(content=[11, 21, 31], indices=[(0, 0), (0, 1), (0, 2)]),
        },
    }

    # when
    result = list(unfold_parameters(parameters=parameters))

    # then
    assert (
        len(result) == 2
    ), "Expected two elements as this is the size of smallest batch"
    assert result[0] == {
        "some": "static_value",
        "my_batch": "0",
        "my_list": ["a", "b"],
        "my_dict": {"a": 10, "b": 11},
    }, "Expected to first batches element and other elements broadcast"
    assert result[1] == {
        "some": "static_value",
        "my_batch": "1",
        "my_list": ["c", "d"],
        "my_dict": {"a": 20, "b": 21},
    }, "Expected to second batch elements and other elements broadcast"


def test_unfold_parameters_when_batch_parameter_given_in_primitive_parameter_list_and_dict_and_nested_batches() -> (
    None
):
    # given
    parameters = {
        "some": "static_value",
        "my_batch": Batch(content=["0", "1", "2"], indices=[(0, 0), (0, 1), (0, 2)]),
        "my_list": [
            Batch(content=["a", "c", "e"], indices=[(0, 0), (0, 1), (0, 2)]),
            Batch(content=["b", "d", "f"], indices=[(0, 0), (0, 1), (0, 2)]),
        ],
        "my_dict": {
            "a": Batch(
                content=[
                    Batch(content=["a1", "a2"], indices=[(0, 0, 0), (0, 0, 1)]),
                    Batch(content=["b1", "b2"], indices=[(0, 1, 0), (0, 1, 1)]),
                    Batch(content=["c1", "c2"], indices=[(0, 2, 0), (0, 2, 1)]),
                ],
                indices=[(0, 0), (0, 1), (0, 2)],
            ),
            "b": Batch(content=[11, 21, 31], indices=[(0, 0), (0, 1), (0, 2)]),
        },
    }

    # when
    result = list(unfold_parameters(parameters=parameters))

    # then
    assert (
        len(result) == 3
    ), "Expected three elements as this is the size of batch each batch"
    my_dict_0 = result[0]["my_dict"]
    assert isinstance(my_dict_0["a"], Batch), "Expected nested batch not to be unfolded"
    assert list(my_dict_0["a"]) == [
        "a1",
        "a2",
    ], "Expected nested batch content not to be touched"
    assert my_dict_0["b"] == 11, "Expected other dict-batch to be unfolded"
    del result[0]["my_dict"]
    assert result[0] == {
        "some": "static_value",
        "my_batch": "0",
        "my_list": ["a", "b"],
    }, "Expected to first batches element and other elements broadcast"
    my_dict_1 = result[1]["my_dict"]
    assert isinstance(my_dict_1["a"], Batch), "Expected nested batch not to be unfolded"
    assert list(my_dict_1["a"]) == [
        "b1",
        "b2",
    ], "Expected nested batch content not to be touched"
    assert my_dict_1["b"] == 21, "Expected other dict-batch to be unfolded"
    del result[1]["my_dict"]
    assert result[1] == {
        "some": "static_value",
        "my_batch": "1",
        "my_list": ["c", "d"],
    }, "Expected to second batch elements and other elements broadcast"
    my_dict_2 = result[2]["my_dict"]
    assert isinstance(my_dict_2["a"], Batch), "Expected nested batch not to be unfolded"
    assert list(my_dict_2["a"]) == [
        "c1",
        "c2",
    ], "Expected nested batch content not to be touched"
    assert my_dict_2["b"] == 31, "Expected other dict-batch to be unfolded"
    del result[2]["my_dict"]
    assert result[2] == {
        "some": "static_value",
        "my_batch": "2",
        "my_list": ["e", "f"],
    }, "Expected to third batch elements and other elements broadcast"


def test_unfold_parameters_when_batch_parameter_given_in_primitive_parameter_list_and_dict_and_primitives_are_in_lists() -> (
    None
):
    # given
    parameters = {
        "some": "static_value",
        "my_batch": Batch(content=["0", "1", "2"], indices=[(0, 0), (0, 1), (0, 2)]),
        "my_list": [
            Batch(content=["a", "c", "e"], indices=[(0, 0), (0, 1), (0, 2)]),
            Batch(content=["b", "d", "f"], indices=[(0, 0), (0, 1), (0, 2)]),
            "some",
            "other",
        ],
        "my_dict": {
            "a": Batch(content=[10, 20, 30], indices=[(0, 0), (0, 1), (0, 2)]),
            "b": Batch(content=[11, 21, 31], indices=[(0, 0), (0, 1), (0, 2)]),
            "primitive": "value",
            "nested_list": [0, 1],
        },
    }

    # when
    result = list(unfold_parameters(parameters=parameters))

    # then
    assert (
        len(result) == 3
    ), "Expected three elements as this is the size of batch each batch"
    assert result[0] == {
        "some": "static_value",
        "my_batch": "0",
        "my_list": ["a", "b", "some", "other"],
        "my_dict": {"a": 10, "b": 11, "primitive": "value", "nested_list": [0, 1]},
    }, "Expected to first batches element and other elements broadcast"
    assert result[1] == {
        "some": "static_value",
        "my_batch": "1",
        "my_list": ["c", "d", "some", "other"],
        "my_dict": {"a": 20, "b": 21, "primitive": "value", "nested_list": [0, 1]},
    }, "Expected to second batch elements and other elements broadcast"
    assert result[2] == {
        "some": "static_value",
        "my_batch": "2",
        "my_list": ["e", "f", "some", "other"],
        "my_dict": {"a": 30, "b": 31, "primitive": "value", "nested_list": [0, 1]},
    }, "Expected to third batch elements and other elements broadcast"


def test_remove_empty_indices() -> None:
    # given
    value = {
        "some": "a",
        "other": [1, 2, 3, 4],
        "simple_batch": Batch(
            content=["a", "c", "e"], indices=[(0, 0), (0, 1), (0, 2)]
        ),
        "list_of_batches": [
            Batch(content=["1", "2", "3"], indices=[(0, 0), (0, 1), (0, 2)]),
            Batch(content=["4", "5", "6"], indices=[(0, 0), (0, 1), (0, 2)]),
            "not_to_be_touched",
        ],
        "dict_of_batches": {
            "not": ["to", "be", "touched"],
            "nested_batch": Batch(
                content=["a1", "a2", "a3"], indices=[(0, 0), (0, 1), (0, 2)]
            ),
            "not_impacted_batch": Batch(
                content=["b1", "b2", "b3"], indices=[(1, 0), (1, 1), (1, 2)]
            ),
        },
    }

    # when
    result = remove_indices(value=value, indices={(0, 1), (0, 2)})

    # then
    assert result["some"] == "a", "Expected not to be touched"
    assert result["other"] == [1, 2, 3, 4], "Expected not to be touched"
    assert result["simple_batch"].indices == [(0, 0)], "Expected to be filtered"
    assert list(result["simple_batch"]) == ["a"], "Expected to be filtered"
    assert result["list_of_batches"][0].indices == [(0, 0)], "Expected to be filtered"
    assert list(result["list_of_batches"][0]) == ["1"], "Expected to be filtered"
    assert result["list_of_batches"][1].indices == [(0, 0)], "Expected to be filtered"
    assert list(result["list_of_batches"][1]) == ["4"], "Expected to be filtered"
    assert (
        result["list_of_batches"][2] == "not_to_be_touched"
    ), "Expected not to be touched"
    assert result["dict_of_batches"]["not"] == [
        "to",
        "be",
        "touched",
    ], "Expected not to be touched"
    assert result["dict_of_batches"]["nested_batch"].indices == [
        (0, 0)
    ], "Expected to be filtered"
    assert list(result["dict_of_batches"]["nested_batch"]) == [
        "a1"
    ], "Expected to be filtered"
    assert list(result["dict_of_batches"]["not_impacted_batch"]) == [
        "b1",
        "b2",
        "b3",
    ], "Expected not to be touched"


@pytest.mark.parametrize("value", ["some", 1, 2, {"some": "value"}, [1, 2, 3]])
def test_get_empty_batch_elements_indices_from_non_batch_elements(value: Any) -> None:
    # when
    result = get_empty_batch_elements_indices(value=value)

    # then
    assert result == set(), "Expected to see empty result, as no batches provided"


def test_get_empty_batch_elements_indices_from_single_batch() -> None:
    # given
    batch = Batch(
        content=["1", None, "3", None], indices=[(0, 0), (0, 1), (0, 2), (0, 3)]
    )

    # when
    result = get_empty_batch_elements_indices(value=batch)

    # then
    assert result == {
        (0, 1),
        (0, 3),
    }, "Expected to see two indices, the ones for None elements in content"


def test_get_empty_batch_elements_indices_from_list_of_batches() -> None:
    # given
    batch = [
        Batch(content=["1", None, "3", "4"], indices=[(0, 0), (0, 1), (0, 2), (0, 3)]),
        Batch(content=["1", "2", "3", None], indices=[(0, 0), (0, 1), (0, 2), (0, 3)]),
    ]

    # when
    result = get_empty_batch_elements_indices(value=batch)

    # then
    assert result == {
        (0, 1),
        (0, 3),
    }, "Expected to see two indices, the ones for None elements in content"


def test_get_empty_batch_elements_indices_from_dict_of_batches() -> None:
    # given
    value = {
        "a": Batch(
            content=["1", None, "3", "4"], indices=[(0, 0), (0, 1), (0, 2), (0, 3)]
        ),
        "b": Batch(
            content=["1", "2", "3", None], indices=[(0, 0), (0, 1), (0, 2), (0, 3)]
        ),
    }

    # when
    result = get_empty_batch_elements_indices(value=value)

    # then
    assert result == {
        (0, 1),
        (0, 3),
    }, "Expected to see two indices, the ones for None elements in content"


def test_ensure_compound_input_indices_match_when_there_is_no_indices() -> None:
    # given
    indices = []

    # when
    ensure_compound_input_indices_match(indices=indices)

    # then - no error


def test_ensure_compound_input_indices_match_when_there_is_only_a_single_set_of_indices() -> (
    None
):
    # given
    indices = [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
    ]

    # when
    ensure_compound_input_indices_match(indices=indices)

    # then - no error


def test_ensure_compound_input_indices_match_when_there_is_a_match() -> None:
    # given
    indices = [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (0, 1), (0, 2), (0, 3)]]

    # when
    ensure_compound_input_indices_match(indices=indices)

    # then - no error


def test_ensure_compound_input_indices_match_when_there_is_no_match() -> None:
    # given
    indices = [[(0, 0), (0, 1), (0, 2), (0, 3)], [(0, 0), (0, 2), (0, 3)]]

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        ensure_compound_input_indices_match(indices=indices)


def test_reduce_batch_dimensionality_when_reduction_is_legal() -> None:
    # given
    guard_of_indices_wrapping = GuardForIndicesWrapping()

    # when
    result = reduce_batch_dimensionality(
        indices=[(0, 0), (0, 1), (1, 2), (1, 3)],
        upper_level_index=[(0,), (1,)],
        data=["a", "b", "c", "d"],
        guard_of_indices_wrapping=guard_of_indices_wrapping,
    )

    # then
    assert len(result) == 2, "Expected 2 output batches"
    assert result.indices == [
        (0,),
        (1,),
    ], "Expected index to be reduced 1 lvl of dimensionality"
    assert list(result[0]) == [
        "a",
        "b",
    ], "Expected 1st batch to contain elements with prefix (0, ) in index"
    assert result[0].indices == [
        (0, 0),
        (0, 1),
    ], "Part of original index must be preserved"
    assert list(result[1]) == [
        "c",
        "d",
    ], "Expected 2nd batch to contain elements with prefix (1, ) in index"
    assert result[1].indices == [
        (1, 2),
        (1, 3),
    ], "Part of original index must be preserved"


def test_reduce_batch_dimensionality_when_reduction_is_legal_but_nested_result_skip_high_level_index_element() -> (
    None
):
    # given
    guard_of_indices_wrapping = GuardForIndicesWrapping()

    # when
    result = reduce_batch_dimensionality(
        indices=[(0, 0), (0, 1), (2, 2), (2, 3)],
        upper_level_index=[(0,), (1,), (2,)],
        data=["a", "b", "c", "d"],
        guard_of_indices_wrapping=guard_of_indices_wrapping,
    )

    # then
    assert len(result) == 3, "Expected 3 output batches"
    assert result.indices == [
        (0,),
        (1,),
        (2,),
    ], "Expected index to be reduced 1 lvl of dimensionality"
    assert list(result[0]) == [
        "a",
        "b",
    ], "Expected 1st batch to contain elements with prefix (0, ) in index"
    assert result[0].indices == [
        (0, 0),
        (0, 1),
    ], "Part of original index must be preserved"
    assert result[1] is None, "Expected missing element to be None"
    assert list(result[2]) == [
        "c",
        "d",
    ], "Expected 3rd batch to contain elements with prefix (2, ) in index"
    assert result[2].indices == [
        (2, 2),
        (2, 3),
    ], "Part of original index must be preserved"


def test_reduce_batch_dimensionality_when_reduction_is_illegal() -> None:
    # given
    guard_of_indices_wrapping = GuardForIndicesWrapping()
    guard_of_indices_wrapping.register_wrapping(
        indices_before_wrapping=[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)],
    )

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = reduce_batch_dimensionality(
            indices=[(0, 0), (0, 1), (1, 2), (1, 3)],
            upper_level_index=[(0,), (1,)],
            data=["a", "b", "c", "d"],
            guard_of_indices_wrapping=guard_of_indices_wrapping,
        )
