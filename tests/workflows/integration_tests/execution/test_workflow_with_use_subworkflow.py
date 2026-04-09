"""
End-to-end tests for roboflow_core/use_subworkflow@v1 (nested workflow execution).
"""

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine


def _echo_child_workflow() -> dict:
    """Minimal child: one WorkflowParameter, one formatter step, one output."""
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "child_msg",
                "default_value": "default-child",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/first_non_empty_or_default@v1",
                "name": "pick",
                "data": ["$inputs.child_msg"],
                "default": "fallback-inner",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "echo",
                "selector": "$steps.pick.output",
            },
        ],
    }


def test_workflow_with_use_subworkflow_maps_parent_input_to_child_output(
    model_manager: ModelManager,
) -> None:
    embedded = _echo_child_workflow()
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "parent_msg",
                "default_value": "unused-default",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/use_subworkflow@v1",
                "name": "nested",
                "embedded_workflow": embedded,
                "parameter_bindings": {
                    "child_msg": "$inputs.parent_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested.echo",
            },
        ],
    }
    init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = engine.run(
        runtime_parameters={"parent_msg": "hello-from-parent"},
    )

    assert len(result) == 1
    assert result[0] == {"from_child": "hello-from-parent"}


def test_workflow_with_stacked_use_subworkflow_runs_at_depth_two(
    model_manager: ModelManager,
) -> None:
    """Parent use_subworkflow wraps a child workflow that itself contains use_subworkflow."""
    inner = _echo_child_workflow()
    middle = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "wrapper_msg",
                "default_value": "unused-middle",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/use_subworkflow@v1",
                "name": "inner_nested",
                "embedded_workflow": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.wrapper_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "wrapped",
                "selector": "$steps.inner_nested.echo",
            },
        ],
    }
    workflow_definition = {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "root_msg",
                "default_value": "unused-root",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/use_subworkflow@v1",
                "name": "outer_nested",
                "embedded_workflow": middle,
                "parameter_bindings": {
                    "wrapper_msg": "$inputs.root_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "final",
                "selector": "$steps.outer_nested.wrapped",
            },
        ],
    }
    init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = engine.run(runtime_parameters={"root_msg": "depth-two-value"})

    assert len(result) == 1
    assert result[0] == {"final": "depth-two-value"}
