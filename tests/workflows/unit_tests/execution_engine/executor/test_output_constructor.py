import numpy as np

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


def test_construct_response_when_field_needs_to_be_grabbed_from_nested_output_in_own_coordinates() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step(
        step_name="b",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {"predictions": ["a", "b"], "predictions_parent_coordinates": ["a`", "b`"]},
            {"predictions": ["c", "d"], "predictions_parent_coordinates": ["c`", "d`"]},
        ],
    )
    execution_cache.register_step_outputs(
        step_name="b",
        outputs=[
            {"predictions": ["g", "h", "i"]},
        ],
    )
    workflow_outputs = [
        JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.predictions",
            coordinates_system=CoordinatesSystem.OWN,
        ),
        JsonField(
            type="JsonField",
            name="other",
            selector="$steps.b.predictions",
            coordinates_system=CoordinatesSystem.OWN,
        ),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs,
        execution_cache=execution_cache,
        runtime_parameters={},
    )

    # then
    assert result == {"some": [["a", "b"], ["c", "d"]], "other": [["g", "h", "i"]]}


def test_construct_response_when_field_needs_to_be_grabbed_from_nested_output_in_parent_coordinates() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step(
        step_name="b",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {"predictions": ["a", "b"], "predictions_parent_coordinates": ["a`", "b`"]},
            {"predictions": ["c", "d"], "predictions_parent_coordinates": ["c`", "d`"]},
        ],
    )
    execution_cache.register_step_outputs(
        step_name="b",
        outputs=[
            {"predictions": ["g", "h", "i"]},
        ],
    )
    workflow_outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.predictions"),
        JsonField(type="JsonField", name="other", selector="$steps.b.predictions"),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs,
        execution_cache=execution_cache,
        runtime_parameters={},
    )

    # then
    assert result == {"some": [["a`", "b`"], ["c`", "d`"]], "other": [["g", "h", "i"]]}


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
        workflow_outputs=workflow_outputs,
        execution_cache=execution_cache,
        runtime_parameters={},
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
        workflow_outputs=workflow_outputs,
        execution_cache=execution_cache,
        runtime_parameters={},
    )

    # then
    assert result == {"some": []}


def test_construct_response_when_wildcard_selector_used_and_parent_coordinates_system_selected() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[
            OutputDefinition(name="predictions"),
            OutputDefinition(name="data"),
        ],
        compatible_with_batches=True,
    )
    execution_cache.register_step(
        step_name="b",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {
                "predictions": ["a", "b"],
                "predictions_parent_coordinates": ["a`", "b`"],
                "data": ":)",
            },
            {
                "predictions": ["c", "d"],
                "predictions_parent_coordinates": ["c`", "d`"],
                "data": ":(",
            },
        ],
    )
    execution_cache.register_step_outputs(
        step_name="b",
        outputs=[
            {"predictions": ["g", "h", "i"]},
        ],
    )
    workflow_outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.*"),
        JsonField(type="JsonField", name="other", selector="$steps.b.*"),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs,
        execution_cache=execution_cache,
        runtime_parameters={},
    )

    # then
    assert result == {
        "some": [
            {"predictions": ["a`", "b`"], "data": ":)"},
            {"predictions": ["c`", "d`"], "data": ":("},
        ],
        "other": [{"predictions": ["g", "h", "i"]}],
    }


def test_construct_response_when_wildcard_selector_used_and_own_coordinates_system_selected() -> (
    None
):
    # given
    execution_cache = ExecutionCache.init()
    execution_cache.register_step(
        step_name="a",
        output_definitions=[
            OutputDefinition(name="predictions"),
            OutputDefinition(name="data"),
        ],
        compatible_with_batches=True,
    )
    execution_cache.register_step(
        step_name="b",
        output_definitions=[OutputDefinition(name="predictions")],
        compatible_with_batches=True,
    )
    execution_cache.register_step_outputs(
        step_name="a",
        outputs=[
            {
                "predictions": ["a", "b"],
                "predictions_parent_coordinates": ["a`", "b`"],
                "data": ":)",
            },
            {
                "predictions": ["c", "d"],
                "predictions_parent_coordinates": ["c`", "d`"],
                "data": ":(",
            },
        ],
    )
    execution_cache.register_step_outputs(
        step_name="b",
        outputs=[
            {"predictions": ["g", "h", "i"]},
        ],
    )
    workflow_outputs = [
        JsonField(
            type="JsonField",
            name="some",
            selector="$steps.a.*",
            coordinates_system=CoordinatesSystem.OWN,
        ),
        JsonField(
            type="JsonField",
            name="other",
            selector="$steps.b.*",
            coordinates_system=CoordinatesSystem.OWN,
        ),
    ]

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs,
        execution_cache=execution_cache,
        runtime_parameters={},
    )

    # then
    assert result == {
        "some": [
            {"predictions": ["a", "b"], "data": ":)"},
            {"predictions": ["c", "d"], "data": ":("},
        ],
        "other": [{"predictions": ["g", "h", "i"]}],
    }


def test_construct_response_when_results_to_be_grabbed_from_inputs() -> None:
    # given
    execution_cache = ExecutionCache.init()
    workflow_outputs = [
        JsonField(type="JsonField", name="some", selector="$inputs.image"),
        JsonField(type="JsonField", name="other", selector="$inputs.confidence"),
    ]
    runtime_parameters = {
        "image": [np.zeros((192, 168, 3), dtype=np.uint8)],
        "confidence": 0.3,
    }

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs,
        execution_cache=execution_cache,
        runtime_parameters=runtime_parameters,
    )

    # then
    assert len(result) == 2, "Expected 2 elements to be registered in the output"
    assert (
        result["other"] == 0.3
    ), "Confidence value (without list wrapping) to be fetched from inputs and named `other`"
    assert np.allclose(
        result["some"], [np.zeros((192, 168, 3), dtype=np.uint8)]
    ), "`some` output expected to carry one-element list of input image"


def test_construct_response_when_results_to_be_grabbed_from_inputs_and_step_outputs() -> (
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
            {"predictions": ["a", "b", "c"]},
            {"predictions": ["d", "e", "f"]},
        ],
    )
    workflow_outputs = [
        JsonField(type="JsonField", name="a", selector="$inputs.image"),
        JsonField(type="JsonField", name="b", selector="$inputs.confidence"),
        JsonField(type="JsonField", name="c", selector="$steps.a.predictions"),
    ]
    runtime_parameters = {
        "image": [np.zeros((192, 168, 3), dtype=np.uint8)],
        "confidence": 0.3,
    }

    # when
    result = construct_workflow_output(
        workflow_outputs=workflow_outputs,
        execution_cache=execution_cache,
        runtime_parameters=runtime_parameters,
    )

    # then
    assert len(result) == 3, "Expected 3 elements to be registered in the output"
    assert (
        result["b"] == 0.3
    ), "Confidence value (without list wrapping) to be fetched from inputs and named `b`"
    assert np.allclose(
        result["a"], [np.zeros((192, 168, 3), dtype=np.uint8)]
    ), "`a` output expected to carry one-element list of input image"
    assert result["c"] == [
        ["a", "b", "c"],
        ["d", "e", "f"],
    ], "All predictions from step `c` expected to be registered under `c` output"
