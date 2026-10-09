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
            "type": "roboflow_core/mistral_vlm@v1",
            "name": "mistral",
            "images": "$inputs.image",
            "model_version": "Mistral Large 4",
            "task_type": "object-detection",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "visualization",
            "image": "$inputs.image",
            "predictions": "$steps.mistral.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "mistral_result",
            "selector": "$steps.mistral.output",
        },
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.mistral.predictions",
        },
        {
            "type": "JsonField",
            "name": "visualization",
            "selector": "$steps.visualization.image",
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
