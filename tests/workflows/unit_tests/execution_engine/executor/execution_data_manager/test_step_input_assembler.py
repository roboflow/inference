from typing import Any

import pytest

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.step_input_assembler import (
    GuardForIndicesWrapping,
    ensure_compound_input_indices_match,
    filter_to_valid_prefix_chains,
    get_empty_batch_elements_indices,
    get_masks_intersection_for_dimensions,
    intersect_masks_per_dimension,
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


def test_get_masks_intersection_for_dimensions_two_masks_same_dimension_returns_intersection() -> (
    None
):
    # Two masks at the same dimension level (dimension 1): intersection should contain
    # only indices that appear in both masks.
    # given
    mask_a = {(0,), (3,)}
    mask_b = {(1,), (2,), (3,)}

    # when
    result = get_masks_intersection_for_dimensions(
        batch_masks=[mask_a, mask_b],
        dimensions={1},
    )

    # then
    assert result == {1: {(3,)}}, "Expected intersection of two masks at dimension 1"


# --- intersect_masks_per_dimension (intra-dimensional intersection) ---


def test_intersect_masks_per_dimension_two_masks_same_dimension_returns_intersection() -> (
    None
):
    # At dimension 1: only indices present in every mask are kept.
    mask_a = {(0,), (3,)}
    mask_b = {(1,), (2,), (3,)}

    result = intersect_masks_per_dimension(
        batch_masks=[mask_a, mask_b],
        dimensions={1},
    )

    assert result == {1: {(3,)}}


def test_intersect_masks_per_dimension_multi_dimension_separate_masks_at_dim2_empty() -> (
    None
):
    # At dimension 2 we have two *separate* masks: one with {(1,0)}, one with {(3,1)}.
    # Intra-dim intersection: only indices that appear in *every* contributing mask.
    # So result[2] is empty (no index is in both). Contrast with scenario 1 where one mask
    # has {(1,0), (3,1)} at dim 2, so after intersection dim 2 = {(1,0), (3,1)}.
    mask_d1_a = {(0,), (1,), (3,)}
    mask_d1_b = {(1,), (2,), (3,)}
    mask_d2_a = {(1, 0)}
    mask_d2_b = {(3, 1)}
    mask_d3 = {(1, 0, 0)}

    result = intersect_masks_per_dimension(
        batch_masks=[mask_d1_a, mask_d1_b, mask_d2_a, mask_d2_b, mask_d3],
        dimensions={1, 2, 3},
    )

    assert result[1] == {(1,), (3,)}
    assert result[2] == set()
    assert result[3] == {(1, 0, 0)}


def test_intersect_masks_per_dimension_three_masks_partial_overlap_at_dim1() -> None:
    # Three masks at dim 1: intersection is only the index present in all three.
    mask_a = {(0,), (1,)}
    mask_b = {(1,), (2,)}
    mask_c = {(1,)}

    result = intersect_masks_per_dimension(
        batch_masks=[mask_a, mask_b, mask_c],
        dimensions={1},
    )

    assert result == {1: {(1,)}}


def test_intersect_masks_per_dimension_single_mask_multi_dim_returns_that_mask() -> (
    None
):
    # Single mask: "intersection" over one set is the set itself.
    mask = {(0,), (1,), (0, 0), (1, 0)}

    result = intersect_masks_per_dimension(
        batch_masks=[mask],
        dimensions={1, 2},
    )

    assert result[1] == {(0,), (1,)}
    assert result[2] == {(0, 0), (1, 0)}


def test_intersect_masks_per_dimension_two_masks_agree_at_dim2_one_has_no_dim2() -> (
    None
):
    # At dim 2 only one mask has indices; we intersect over non-empty sets only, so result is that set.
    mask_d1_both = {(0,), (1,)}  # both masks have dim-1
    mask_d2_one = {(0, 0), (1, 0)}  # only this mask has dim-2

    result = intersect_masks_per_dimension(
        batch_masks=[mask_d1_both, mask_d2_one],
        dimensions={1, 2},
    )

    assert result[1] == {(0,), (1,)}
    assert result[2] == {(0, 0), (1, 0)}


# --- filter_to_valid_prefix_chains (inter-level intersection) ---


def test_filter_to_valid_prefix_chains_keeps_only_full_chain() -> None:
    # Input per_dim is what we get *after* intra-dim intersection when there is a single
    # mask at dim 2 (e.g. scenario 1: one mask has {(1,0), (3,1)}). So dim 2 = {(1,0), (3,1)}.
    # This test checks inter-level filtering only: keep indices that form a full chain.
    # (1,) -> (1,0) -> (1,0,0) is complete; (3,) -> (3,1) has no dim3 child, so (3,) and (3,1) are dropped.
    per_dim = {
        1: {(1,), (3,)},
        2: {(1, 0), (3, 1)},
        3: {(1, 0, 0)},
    }

    result = filter_to_valid_prefix_chains(per_dim, dimensions={1, 2, 3})

    assert result == {1: {(1,)}, 2: {(1, 0)}, 3: {(1, 0, 0)}}


def test_filter_to_valid_prefix_chains_single_dimension_unchanged() -> None:
    per_dim = {1: {(0,), (1,)}}

    result = filter_to_valid_prefix_chains(per_dim, dimensions={1})

    assert result == {1: {(0,), (1,)}}


def test_filter_to_valid_prefix_chains_multiple_complete_chains_all_kept() -> None:
    # Two full chains: (0,) -> (0,0) -> (0,0,0) and (1,) -> (1,0) -> (1,0,0). Both kept.
    per_dim = {
        1: {(0,), (1,)},
        2: {(0, 0), (1, 0)},
        3: {(0, 0, 0), (1, 0, 0)},
    }

    result = filter_to_valid_prefix_chains(per_dim, dimensions={1, 2, 3})

    assert result[1] == {(0,), (1,)}
    assert result[2] == {(0, 0), (1, 0)}
    assert result[3] == {(0, 0, 0), (1, 0, 0)}


def test_filter_to_valid_prefix_chains_orphan_at_dim1_dropped() -> None:
    # (2,) at dim 1 has no descendant at dim 2; (0,) -> (0,0) is a complete chain.
    per_dim = {
        1: {(0,), (2,)},
        2: {(0, 0)},
    }

    result = filter_to_valid_prefix_chains(per_dim, dimensions={1, 2})

    assert result == {1: {(0,)}, 2: {(0, 0)}}


def test_filter_to_valid_prefix_chains_empty_at_one_dimension_all_empty() -> None:
    # Dim 2 is empty; no chain can cross all dimensions.
    per_dim = {
        1: {(0,), (1,)},
        2: set(),
        3: {(0, 0, 0)},
    }

    result = filter_to_valid_prefix_chains(per_dim, dimensions={1, 2, 3})

    assert result == {1: set(), 2: set(), 3: set()}


# --- get_masks_intersection_for_dimensions (full pipeline) ---


def test_get_masks_intersection_for_dimensions_scenario1_hierarchical_chain() -> None:
    # Intra-dim intersection then prefix-chain filter yields (1,) -> (1,0) -> (1,0,0).
    batch_masks = [
        {(0,), (1,), (3,)},
        {(1,), (2,), (3,)},
        {(1, 0), (3, 1)},
        {(1, 0, 0)},
    ]

    result = get_masks_intersection_for_dimensions(
        batch_masks=batch_masks,
        dimensions={1, 2, 3},
    )

    assert result == {1: {(1,)}, 2: {(1, 0)}, 3: {(1, 0, 0)}}


def test_get_masks_intersection_for_dimensions_scenario2_intra_dim_empty_all_empty() -> (
    None
):
    # dim2 has two separate masks {(1,0)} and {(3,1)} -> intersection empty.
    # So no valid chain; all dimensions end up empty.
    batch_masks = [
        {(0,), (1,), (3,)},
        {(1,), (2,), (3,)},
        {(1, 0)},
        {(3, 1)},
        {(1, 0, 0)},
    ]

    result = get_masks_intersection_for_dimensions(
        batch_masks=batch_masks,
        dimensions={1, 2, 3},
    )

    assert result == {1: set(), 2: set(), 3: set()}


def test_get_masks_intersection_for_dimensions_empty_batch_masks_returns_none_per_dim() -> (
    None
):
    result = get_masks_intersection_for_dimensions(
        batch_masks=[],
        dimensions={1, 2},
    )

    assert result == {1: None, 2: None}


def test_get_masks_intersection_for_dimensions_single_mask_multi_dim_identity() -> None:
    # One mask: intersection is that mask; chain filter keeps full chain(s).
    batch_masks = [
        {(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0)},
    ]

    result = get_masks_intersection_for_dimensions(
        batch_masks=batch_masks,
        dimensions={1, 2, 3},
    )

    assert result[1] == {(0,), (1,)}
    assert result[2] == {(0, 0), (1, 0)}
    assert result[3] == {(0, 0, 0), (1, 0, 0)}


def test_get_masks_intersection_for_dimensions_three_masks_dim1_agree_one_index() -> (
    None
):
    # All three masks at dim 1 contain only (1,); no higher dims.
    batch_masks = [{(0,), (1,)}, {(1,), (2,)}, {(1,)}]

    result = get_masks_intersection_for_dimensions(
        batch_masks=batch_masks,
        dimensions={1},
    )

    assert result == {1: {(1,)}}
