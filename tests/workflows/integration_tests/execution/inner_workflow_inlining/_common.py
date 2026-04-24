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
    """Child workflow that echoes ``child_msg`` via ``ScalarOnlyEchoBlock`` (``scalar_only_echo``).

    Requires the stub plugin from ``stub_plugins.scalar_only_block_plugin`` to be registered
    (see ``conftest.py`` in this package, or patch ``blocks_loader.get_plugin_modules``).

    For list-valued ``WorkflowParameter`` substitutions into ``child_msg``, use a non-scalar
    child step instead (see ``test_inner_workflow_with_list_valued_workflow_parameter``).
    """
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
                "type": "scalar_only_echo",
                "name": "pick",
                "value": "$inputs.child_msg",
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


def child_dimension_collapse_from_parent_detections() -> Dict[str, Any]:
    """Inner workflow: ``dimension_collapse`` only; parent supplies OD ``predictions``."""
    return {
        "version": "1.0",
        "inputs": [
            {
                "type": "WorkflowBatchInput",
                "name": "predictions",
                "kind": ["object_detection_prediction"],
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/dimension_collapse@v1",
                "name": "collapse",
                "data": "$inputs.predictions",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "collapsed",
                "selector": "$steps.collapse.output",
            },
        ],
    }


def child_detection_only_for_parent_dynamic_crop() -> Dict[str, Any]:
    """Inner workflow: OD only; parent runs ``dynamic_crop`` on inner detections."""
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "general_detection",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "detection_predictions",
                "selector": "$steps.general_detection.predictions",
            },
        ],
    }
