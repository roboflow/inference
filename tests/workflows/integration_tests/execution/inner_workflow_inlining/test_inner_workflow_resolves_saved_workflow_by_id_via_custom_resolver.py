"""
Equivalence for ``test_inner_workflow_resolves_saved_workflow_by_id_via_custom_resolver``:
nested workflow resolved by reference at compile time vs inlined child steps.
"""

from typing import Any, Dict, Optional

from inference.core.managers.base import ModelManager
from inference.core.workflows.execution_engine.v1.inner_workflow.reference_resolution import (
    WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER,
)
from tests.workflows.integration_tests.execution.inner_workflow_inlining._common import (
    echo_child_workflow,
    execution_engine,
)


def _nested_by_ref_workflow() -> Dict[str, Any]:
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
                "name": "nested_by_ref",
                "workflow_workspace_id": "stub-ws",
                "workflow_id": "stub-id",
                "workflow_version_id": "stub-version-id",
                "parameter_bindings": {
                    "child_msg": "$inputs.parent_msg",
                },
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "from_child",
                "selector": "$steps.nested_by_ref.echo",
            },
        ],
    }


def _flat_parent_workflow() -> Dict[str, Any]:
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


def test_inlined_echo_matches_inner_workflow_resolved_by_ref(
    model_manager: ModelManager,
) -> None:
    def resolver(
        workspace_id: str,
        workflow_id: str,
        workflow_version_id: Optional[str],
        init_parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        assert workspace_id == "stub-ws"
        assert workflow_id == "stub-id"
        assert workflow_version_id == "stub-version-id"
        return echo_child_workflow()

    nested_engine = execution_engine(
        model_manager,
        _nested_by_ref_workflow(),
        extra_init_parameters={WORKFLOWS_CORE_INNER_WORKFLOW_SPEC_RESOLVER: resolver},
    )
    flat_engine = execution_engine(model_manager, _flat_parent_workflow())

    runtime_parameters = {"parent_msg": "hello-from-ref"}
    nested_result = nested_engine.run(runtime_parameters=runtime_parameters)
    flat_result = flat_engine.run(runtime_parameters=runtime_parameters)

    assert nested_result == flat_result
    assert len(flat_result) == 1
    assert flat_result[0] == {"from_child": "hello-from-ref"}
