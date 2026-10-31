"""
Equivalence for ``test_inner_workflow_maps_parent_input_to_child_output``:
nested ``inner_workflow`` vs child steps inlined at the parent level.
"""

from typing import Any, Dict

from inference.core.managers.base import ModelManager
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    echo_child_workflow,
    execution_engine,
)


def _nested_parent_workflow(inner: Dict[str, Any]) -> Dict[str, Any]:
    return {
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


def _flat_parent_workflow_equivalent_to_binding() -> Dict[str, Any]:
    return {
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


def test_inlined_echo_steps_match_inner_workflow_result(
    model_manager: ModelManager,
) -> None:
    inner = echo_child_workflow()
    nested_engine = execution_engine(model_manager, _nested_parent_workflow(inner))
    flat_engine = execution_engine(
        model_manager, _flat_parent_workflow_equivalent_to_binding()
    )

    runtime_parameters = {"parent_msg": "hello-from-parent"}
    nested_result = nested_engine.run(runtime_parameters=runtime_parameters)
    flat_result = flat_engine.run(runtime_parameters=runtime_parameters)

    assert nested_result == flat_result
    assert len(nested_result) == 1
    assert nested_result[0] == {"from_child": "hello-from-parent"}
