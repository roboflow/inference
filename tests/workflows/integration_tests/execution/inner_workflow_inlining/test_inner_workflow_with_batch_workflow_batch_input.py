"""
Equivalence for ``test_inner_workflow_with_batch_workflow_batch_input``.
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
                "type": "WorkflowBatchInput",
                "name": "parent_msg",
                "kind": ["string"],
                "dimensionality": 1,
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
                "type": "WorkflowBatchInput",
                "name": "parent_msg",
                "kind": ["string"],
                "dimensionality": 1,
            },
        ],
        "steps": [
            {
                "type": "scalar_only_echo",
                "name": "pick",
                "value": "$inputs.parent_msg",
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


def test_inlined_batch_input_matches_inner_workflow(
    model_manager: ModelManager,
) -> None:
    inner = echo_child_workflow()
    nested_engine = execution_engine(model_manager, _nested_workflow(inner))
    flat_engine = execution_engine(model_manager, _flat_workflow())

    runtime_parameters = {"parent_msg": ["alpha", "beta", "gamma"]}
    nested_result = nested_engine.run(runtime_parameters=runtime_parameters)
    flat_result = flat_engine.run(runtime_parameters=runtime_parameters)

    assert nested_result == flat_result
    assert len(nested_result) == 3
    assert [row["from_child"] for row in nested_result] == ["alpha", "beta", "gamma"]
