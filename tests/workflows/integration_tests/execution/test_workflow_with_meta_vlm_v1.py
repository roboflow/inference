from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/meta_vlm@v1",
            "name": "muse",
            "images": "$inputs.image",
            "model_version": "Muse Spark 1.2",
            "task_type": "object-detection",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v2",
            "name": "parser",
            "vlm_output": "$steps.muse.output",
            "image": "$inputs.image",
            "classes": "$steps.muse.classes",
            "model_type": "muse",
            "task_type": "object-detection",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "muse_result",
            "selector": "$steps.muse.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.predictions",
        },
    ],
}


def test_object_detection_workflow_compiles(model_manager: ModelManager) -> None:
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    assert execution_engine is not None
