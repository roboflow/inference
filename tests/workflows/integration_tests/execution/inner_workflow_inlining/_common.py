"""Shared helpers for inner_workflow vs inlined-steps equivalence tests."""

from typing import Any, Dict, Optional

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine


def execution_engine(
    model_manager: ModelManager,
    workflow_definition: Dict[str, Any],
    *,
    extra_init_parameters: Optional[Dict[str, Any]] = None,
) -> ExecutionEngine:
    init_parameters: Dict[str, Any] = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    if extra_init_parameters:
        init_parameters.update(extra_init_parameters)
    return ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )


def echo_child_workflow() -> Dict[str, Any]:
    """Aligned with test_workflow_with_inner_workflow._echo_child_workflow."""
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


def child_dynamic_crop_from_parent_detections() -> Dict[str, Any]:
    """Aligned with test_workflow_with_inner_workflow._child_dynamic_crop_from_parent_detections."""
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "cropping",
                "image": "$inputs.image",
                "predictions": "$inputs.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "crop_predictions",
                "selector": "$steps.cropping.predictions",
            },
        ],
    }
