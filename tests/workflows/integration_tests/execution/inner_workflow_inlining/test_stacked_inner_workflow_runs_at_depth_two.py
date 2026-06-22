"""
Equivalence for ``test_stacked_inner_workflow_runs_at_depth_two``:
two-level ``inner_workflow`` nesting vs a single inlined ``pick`` step.
"""

from typing import Any, Dict

from inference.core.managers.base import ModelManager
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    echo_child_workflow,
    execution_engine,
)


def _stacked_nested_workflow() -> Dict[str, Any]:
    inner = echo_child_workflow()
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
                "type": "roboflow_core/inner_workflow@v1",
                "name": "inner_nested",
                "workflow_definition": inner,
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
    return {
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
                "type": "roboflow_core/inner_workflow@v1",
                "name": "outer_nested",
                "workflow_definition": middle,
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


def _flat_equivalent_stacked() -> Dict[str, Any]:
    return {
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
                "type": "scalar_only_echo",
                "name": "pick",
                "value": "$inputs.root_msg",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "final",
                "selector": "$steps.pick.output",
            },
        ],
    }


def test_inlined_echo_matches_stacked_inner_workflow(
    model_manager: ModelManager,
) -> None:
    nested_engine = execution_engine(model_manager, _stacked_nested_workflow())
    flat_engine = execution_engine(model_manager, _flat_equivalent_stacked())

    runtime_parameters = {"root_msg": "depth-two-value"}
    nested_result = nested_engine.run(runtime_parameters=runtime_parameters)
    flat_result = flat_engine.run(runtime_parameters=runtime_parameters)

    assert nested_result == flat_result
    assert len(nested_result) == 1
    assert nested_result[0] == {"final": "depth-two-value"}
