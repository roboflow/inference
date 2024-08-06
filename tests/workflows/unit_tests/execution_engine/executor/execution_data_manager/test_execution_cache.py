import pytest

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.execution_cache import (
    ExecutionCache,
)
from tests.workflows.unit_tests.execution_engine.executor.execution_data_manager.common import (
    prepare_execution_graph_for_tests,
)


def test_execution_cache_contains_steps_after_declaration_of_workflow_steps() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert (
        cache.contains_step(step_name="simd_step") is True
    ), "simd_step must be registered"
    assert (
        cache.contains_step(step_name="non_simd_step") is True
    ), "non_simd_step must be registered"


def test_execution_cache_does_not_contain_not_registered_step() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert (
        cache.contains_step(step_name="not_registered") is False
    ), "not_registered must not be marked as registered"


def test_execution_cache_does_not_contain_non_step_graph_nodes() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert (
        cache.contains_step(step_name="image") is False
    ), "image must not be marked as registered"


def test_execution_cache_correctly_recognised_data_modes_after_declaration_of_workflows_steps() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert (
        cache.step_outputs_batches(step_name="simd_step") is True
    ), "simd_step must be recognised as batch oriented"
    assert (
        cache.step_outputs_batches(step_name="non_simd_step") is False
    ), "non_simd_step must be recognised as not batch oriented"


def test_execution_cache_correctly_recognised_steps_outputs_after_declaration_of_workflows_steps() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert (
        cache.is_step_output_declared(selector="$steps.non_simd_step.a") is True
    ), "non_simd_step output a must be recognised"
    assert (
        cache.is_step_output_declared(selector="$steps.non_simd_step.b") is True
    ), "non_simd_step output b must be recognised"
    assert (
        cache.is_step_output_declared(selector="$steps.non_simd_step.c") is False
    ), "non_simd_step output c must not be recognised, as there was no such output"
    assert (
        cache.is_step_output_declared(selector="$steps.simd_step.c") is True
    ), "simd_step output b must be recognised"


def test_execution_cache_correctly_returns_data_registration_status_for_declared_steps() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert (
        cache.is_step_output_data_registered(step_name="non_simd_step") is False
    ), "Outputs was not registered, hence False expected"
    assert (
        cache.is_step_output_data_registered(step_name="simd_step") is False
    ), "Outputs was not registered, hence False expected"


def test_is_step_output_data_registered_method_for_not_declared_step() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=[], are_batch_oriented=[], steps_outputs=[]
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.is_step_output_data_registered(step_name="non_existing")


def test_is_step_output_declared_method_for_invalid_selector() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=[], are_batch_oriented=[], steps_outputs=[]
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert cache.is_step_output_declared(selector="invalid") is False


def test_is_step_output_declared_method_for_input_selector() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=[], are_batch_oriented=[], steps_outputs=[]
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert cache.is_step_output_declared(selector="$inputs.a") is False


def test_is_step_output_declared_method_for_not_registered_step() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=[], are_batch_oriented=[], steps_outputs=[]
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    assert cache.is_step_output_declared(selector="$steps.a") is False


def test_execution_cache_correctly_returns_all_data_for_declared_steps_when_outputs_were_not_registered() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)
    non_simd_step_outputs = cache.get_all_non_batch_step_outputs(
        step_name="non_simd_step"
    )
    simd_step_outputs = cache.get_all_batch_step_outputs(
        step_name="simd_step", batch_elements_indices=[(0,), (1,), (2,)]
    )

    # then
    assert non_simd_step_outputs == {
        "a": None,
        "b": None,
    }, "Expected empty output to be returned in non-batch fashion"
    assert simd_step_outputs == [
        {"c": None},
        {"c": None},
        {"c": None},
    ], "Expected empty output to be returned in batch fashion"


def test_execution_cache_raises_error_when_trying_to_get_non_batch_outputs_in_batch_oriented_fashion() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.get_all_batch_step_outputs(
            step_name="non_simd_step", batch_elements_indices=[(0,), (1,), (2,)]
        )


def test_execution_cache_raises_error_when_trying_to_get_batch_outputs_in_non_batch_oriented_fashion() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.get_all_non_batch_step_outputs(
            step_name="simd_step",
        )


def test_execution_cache_correctly_returns_selected_data_for_declared_steps_when_outputs_were_not_registered() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)
    non_simd_step_outputs = cache.get_non_batch_output(
        selector="$steps.non_simd_step.a"
    )
    simd_step_outputs = cache.get_batch_output(
        selector="$steps.simd_step.c", batch_elements_indices=[(0,), (1,), (2,)]
    )

    # then
    assert (
        non_simd_step_outputs is None
    ), "Expected empty output to be returned in non-batch fashion"
    assert simd_step_outputs == [
        None,
        None,
        None,
    ], "Expected empty output to be returned in batch fashion"


def test_execution_cache_correctly_raises_error_when_attempting_to_get_non_declared_step_output_of_simd_step() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.get_batch_output(
            selector="$steps.simd_step.not_existing",
            batch_elements_indices=[(0,), (1,), (2,)],
        )


def test_execution_cache_correctly_raises_error_when_attempting_to_get_non_declared_step_output_of_non_simd_step() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.get_non_batch_output(
            selector="$steps.non_simd_step.not_existing",
        )


def test_execution_cache_correctly_raises_error_when_attempting_to_get_simd_step_output_in_non_simd_mode() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.get_non_batch_output(
            selector="$steps.simd_step.c",
        )


def test_execution_cache_correctly_raises_error_when_attempting_to_get_non_simd_step_output_in_simd_mode() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.get_batch_output(
            selector="$steps.non_simd_step.c", batch_elements_indices=[(0,), (1,), (2,)]
        )


def test_execution_cache_correctly_raises_error_when_attempting_to_get_data_of_non_declared_step_in_simd_mode() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.get_batch_output(
            selector="$steps.not_existing.a",
            batch_elements_indices=[(0,), (1,), (2,)],
        )


def test_execution_cache_correctly_raises_error_when_attempting_to_get_data_of_non_declared_step_in_non_simd_mode() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )

    # when
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # then
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = cache.get_non_batch_output(
            selector="$steps.not_existing.a",
        )


def test_registration_of_non_batch_output_for_non_declared_step() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=[],
        are_batch_oriented=[],
        steps_outputs=[],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.register_non_batch_step_outputs(
            step_name="non_existing",
            outputs={"some": "a"},
        )


def test_registration_of_batch_output_for_non_declared_step() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=[],
        are_batch_oriented=[],
        steps_outputs=[],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.register_batch_of_step_outputs(
            step_name="non_existing",
            indices=[(0,), (1,)],
            outputs=[{"some": "a"}, {"some": "b"}],
        )


def test_registration_of_non_batch_output_for_batch_oriented_step() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.register_non_batch_step_outputs(
            step_name="simd_step",
            outputs={"c": 1},
        )


def test_registration_of_batch_output_for_non_batch_oriented_step() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.register_batch_of_step_outputs(
            step_name="non_simd_step",
            indices=[(0,)],
            outputs=[{"a": 1, "b": 2}],
        )


def test_registration_of_non_batch_output_when_not_all_data_fields_provided() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.register_non_batch_step_outputs(
            step_name="non_simd_step",
            outputs={"a": 1},
        )


def test_registration_of_non_batch_output_when_registration_should_succeed() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    cache.register_non_batch_step_outputs(
        step_name="non_simd_step",
        outputs={"a": 1, "b": 2},
    )

    # then
    assert (
        cache.is_step_output_data_registered(step_name="non_simd_step") is True
    ), "Expected to denote the fact of data registration"
    assert cache.get_all_non_batch_step_outputs(step_name="non_simd_step") == {
        "a": 1,
        "b": 2,
    }, "Expected to be able to retrieve all the data"
    assert (
        cache.get_non_batch_output(selector="$steps.non_simd_step.a") == 1
    ), "Expected to be able to retrieve selected data element"


def test_registration_of_batch_output_when_registration_should_fail_due_to_keys_missmatch() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.register_batch_of_step_outputs(
            step_name="simd_step",
            indices=[(0,), (1,)],
            outputs=[{"c": 0}, {"c": 1, "d": "invalid"}],
        )


def test_registration_of_batch_output_when_registration_should_fail_due_to_indices_missmatch() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.register_batch_of_step_outputs(
            step_name="simd_step",
            indices=[(0,), (1,), (2,)],  # to long list of indices
            outputs=[{"c": 0}, {"c": 1}],
        )


def test_registration_of_batch_output_when_registration_should_succeed() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    cache.register_batch_of_step_outputs(
        step_name="simd_step",
        indices=[(0,), (1,)],
        outputs=[{"c": 0}, {"c": 1}],
    )

    # then
    assert (
        cache.is_step_output_data_registered(step_name="simd_step") is True
    ), "Expected to denote the fact of data registration"
    assert cache.get_all_batch_step_outputs(
        step_name="simd_step", batch_elements_indices=[(0,), (1,)]
    ) == [{"c": 0}, {"c": 1}], "Expected to be able to retrieve all the data"
    assert cache.get_batch_output(
        selector="$steps.simd_step.c", batch_elements_indices=[(0,), (1,)]
    ) == [0, 1], "Expected to be able to retrieve selected data element"


def test_registration_of_batch_output_when_registration_should_succeed_and_retrieval_is_performed_with_modified_indices_set() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    cache.register_batch_of_step_outputs(
        step_name="simd_step",
        indices=[(0,), (1,)],
        outputs=[{"c": 0}, {"c": 1}],
    )

    # then
    assert cache.get_batch_output(
        selector="$steps.simd_step.c", batch_elements_indices=[(1,), (0,), (0,), (3,)]
    ) == [
        1,
        0,
        0,
        None,
    ], "Expected to be able to retrieve selected data elements: [second, first, first, non existing]"


def test_registration_of_batch_output_when_registration_should_succeed_and_retrieval_is_performed_with_masking() -> (
    None
):
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["non_simd_step", "simd_step"],
        are_batch_oriented=[False, True],
        steps_outputs=[
            [OutputDefinition(name="a"), OutputDefinition(name="b")],
            [OutputDefinition(name="c")],
        ],
    )
    cache = ExecutionCache.init(execution_graph=execution_graph)

    # when
    cache.register_batch_of_step_outputs(
        step_name="simd_step",
        indices=[(0,), (1,)],
        outputs=[{"c": 0}, {"c": 1}],
    )

    # then
    assert cache.get_batch_output(
        selector="$steps.simd_step.c",
        batch_elements_indices=[(1,), (0,), (0,), (3,)],
        mask={(1,)},
    ) == [
        1,
        None,
        None,
        None,
    ], "Expected to be able to retrieve selected data elements: [second, masked, masked, non existing]"
