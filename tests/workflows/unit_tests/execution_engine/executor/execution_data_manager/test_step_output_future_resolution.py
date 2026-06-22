from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    DynamicStepInputDefinition,
    NodeInputCategory,
    ParameterSpecification,
)
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.step_input_assembler import (
    GuardForIndicesWrapping,
    construct_non_simd_step_non_compound_input,
    get_non_compound_parameter_value,
)


def _completed_future(value):
    future = Future()
    future.set_result(value)
    return future


def _failed_future(error):
    future = Future()
    future.set_exception(error)
    return future


def test_construct_non_simd_step_non_compound_input_resolves_nested_step_output_futures() -> (
    None
):
    execution_cache = MagicMock()
    execution_cache.get_non_batch_output.return_value = _completed_future(
        {"predictions": _completed_future(["resolved"])}
    )
    parameter_spec = DynamicStepInputDefinition(
        parameter_specification=ParameterSpecification(parameter_name="predictions"),
        category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
        data_lineage=[],
        selector="$steps.segmentation.predictions",
    )

    value, is_empty = construct_non_simd_step_non_compound_input(
        step_node=SimpleNamespace(name="consumer"),
        parameter_spec=parameter_spec,
        runtime_parameters={},
        execution_cache=execution_cache,
    )

    assert value == {"predictions": ["resolved"]}
    assert is_empty is False


def test_construct_non_simd_step_non_compound_input_preserves_resolved_step_output() -> (
    None
):
    resolved_output = {"predictions": ["resolved"]}
    execution_cache = MagicMock()
    execution_cache.get_non_batch_output.return_value = resolved_output
    parameter_spec = DynamicStepInputDefinition(
        parameter_specification=ParameterSpecification(parameter_name="predictions"),
        category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
        data_lineage=[],
        selector="$steps.segmentation.predictions",
    )

    value, is_empty = construct_non_simd_step_non_compound_input(
        step_node=SimpleNamespace(name="consumer"),
        parameter_spec=parameter_spec,
        runtime_parameters={},
        execution_cache=execution_cache,
    )

    assert value is resolved_output
    assert is_empty is False


def test_construct_non_simd_step_non_compound_input_propagates_future_resolution_error() -> (
    None
):
    resolution_error = RuntimeError("future resolution failed")
    execution_cache = MagicMock()
    execution_cache.get_non_batch_output.return_value = _completed_future(
        {"predictions": _failed_future(resolution_error)}
    )
    parameter_spec = DynamicStepInputDefinition(
        parameter_specification=ParameterSpecification(parameter_name="predictions"),
        category=NodeInputCategory.NON_BATCH_STEP_OUTPUT,
        data_lineage=[],
        selector="$steps.segmentation.predictions",
    )

    with pytest.raises(RuntimeError) as error_info:
        construct_non_simd_step_non_compound_input(
            step_node=SimpleNamespace(name="consumer"),
            parameter_spec=parameter_spec,
            runtime_parameters={},
            execution_cache=execution_cache,
        )

    assert error_info.value is resolution_error


def test_get_non_compound_parameter_value_resolves_batched_step_output_futures() -> (
    None
):
    execution_cache = MagicMock()
    execution_cache.get_batch_output.return_value = [
        _completed_future({"frame": 0}),
        _completed_future({"frame": 1}),
    ]
    dynamic_batches_manager = MagicMock()
    dynamic_batches_manager.get_indices_for_data_lineage.return_value = [(0,), (1,)]
    parameter = DynamicStepInputDefinition(
        parameter_specification=ParameterSpecification(parameter_name="predictions"),
        category=NodeInputCategory.BATCH_STEP_OUTPUT,
        data_lineage=["<workflow_input>"],
        selector="$steps.segmentation.predictions",
    )

    value, indices, contains_empty = get_non_compound_parameter_value(
        parameter=parameter,
        step_execution_dimensionality=1,
        masks={1: None},
        scalars_discarded=False,
        dynamic_batches_manager=dynamic_batches_manager,
        runtime_parameters={},
        execution_cache=execution_cache,
        guard_of_indices_wrapping=GuardForIndicesWrapping(),
        auto_batch_casting_lineage_supports={},
        step_requests_batch_input=True,
    )

    assert isinstance(value, Batch)
    assert list(value) == [{"frame": 0}, {"frame": 1}]
    assert value.indices == [(0,), (1,)]
    assert indices == [(0,), (1,)]
    assert contains_empty is False


def test_get_non_compound_parameter_value_propagates_batched_future_resolution_error() -> (
    None
):
    resolution_error = RuntimeError("batched future resolution failed")
    execution_cache = MagicMock()
    execution_cache.get_batch_output.return_value = [
        _completed_future({"frame": 0}),
        _failed_future(resolution_error),
    ]
    dynamic_batches_manager = MagicMock()
    dynamic_batches_manager.get_indices_for_data_lineage.return_value = [(0,), (1,)]
    parameter = DynamicStepInputDefinition(
        parameter_specification=ParameterSpecification(parameter_name="predictions"),
        category=NodeInputCategory.BATCH_STEP_OUTPUT,
        data_lineage=["<workflow_input>"],
        selector="$steps.segmentation.predictions",
    )

    with pytest.raises(RuntimeError) as error_info:
        get_non_compound_parameter_value(
            parameter=parameter,
            step_execution_dimensionality=1,
            masks={1: None},
            scalars_discarded=False,
            dynamic_batches_manager=dynamic_batches_manager,
            runtime_parameters={},
            execution_cache=execution_cache,
            guard_of_indices_wrapping=GuardForIndicesWrapping(),
            auto_batch_casting_lineage_supports={},
            step_requests_batch_input=True,
        )

    assert error_info.value is resolution_error
