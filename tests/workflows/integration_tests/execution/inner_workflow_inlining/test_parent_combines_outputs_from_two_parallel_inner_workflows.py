"""
Equivalence for ``test_parent_combines_outputs_from_two_parallel_inner_workflows``.
"""

from typing import Any, Dict

from inference.core.managers.base import ModelManager
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    echo_child_workflow,
    execution_engine,
)


def _nested_workflow(inner: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "msg_a",
                "default_value": "",
            },
            {
                "type": "WorkflowParameter",
                "name": "msg_b",
                "default_value": "",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "branch_a",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.msg_a",
                },
            },
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "branch_b",
                "workflow_definition": inner,
                "parameter_bindings": {
                    "child_msg": "$inputs.msg_b",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "first",
                "selector": "$steps.branch_a.echo",
            },
            {
                "type": "JsonField",
                "name": "second",
                "selector": "$steps.branch_b.echo",
            },
        ],
    }


def _flat_workflow() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "msg_a",
                "default_value": "",
            },
            {
                "type": "WorkflowParameter",
                "name": "msg_b",
                "default_value": "",
            },
        ],
        "steps": [
            {
                "type": "scalar_only_echo",
                "name": "branch_a",
                "value": "$inputs.msg_a",
            },
            {
                "type": "scalar_only_echo",
                "name": "branch_b",
                "value": "$inputs.msg_b",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "first",
                "selector": "$steps.branch_a.output",
            },
            {
                "type": "JsonField",
                "name": "second",
                "selector": "$steps.branch_b.output",
            },
        ],
    }


def test_inlined_parallel_echo_matches_two_inner_workflows(
    model_manager: ModelManager,
) -> None:
    inner = echo_child_workflow()
    nested_engine = execution_engine(model_manager, _nested_workflow(inner))
    flat_engine = execution_engine(model_manager, _flat_workflow())

    runtime_parameters = {"msg_a": "left-branch", "msg_b": "right-branch"}
    nested_result = nested_engine.run(runtime_parameters=runtime_parameters)
    flat_result = flat_engine.run(runtime_parameters=runtime_parameters)

    assert nested_result == flat_result
    assert len(flat_result) == 1
    assert flat_result[0] == {"first": "left-branch", "second": "right-branch"}
