"""
Equivalence for ``test_inner_workflow_with_list_valued_workflow_parameter``.
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
                "name": "parent_msg",
                "default_value": "",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/inner_workflow@v1",
                "name": "nested",
                "workflow_definition": inner,
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


def _flat_workflow() -> Dict[str, Any]:
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowParameter",
                "name": "parent_msg",
                "default_value": "",
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/first_non_empty_or_default@v1",
                "name": "pick",
                "data": ["$inputs.parent_msg"],
                "default": "fallback-inner",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.pick.output",
            },
        ],
    }


def test_inlined_list_workflow_parameter_matches_inner_workflow(
    model_manager: ModelManager,
) -> None:
    inner = echo_child_workflow()
    nested_engine = execution_engine(model_manager, _nested_workflow(inner))
    flat_engine = execution_engine(model_manager, _flat_workflow())

    runtime_parameters = {"parent_msg": ["alpha", "beta", "gamma"]}
    nested_result = nested_engine.run(runtime_parameters=runtime_parameters)
    flat_result = flat_engine.run(runtime_parameters=runtime_parameters)

    assert nested_result == flat_result
    assert len(nested_result) == 1
    assert nested_result[0]["from_child"] == ["alpha", "beta", "gamma"]
