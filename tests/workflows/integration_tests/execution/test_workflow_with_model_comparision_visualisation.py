import numpy as np
from matplotlib import pyplot as plt

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_1",
            "default_value": "yolov8n-640",
        },
        {
            "type": "WorkflowParameter",
            "name": "model_2",
            "default_value": "yolov8n-1280",
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "$inputs.model_1",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model_1",
            "images": "$inputs.image",
            "model_id": "$inputs.model_2",
        },
        {
            "type": "roboflow_core/model_comparison_visualization@v1",
            "name": "model_comparison_visualization",
            "image": "$inputs.image",
            "predictions_a": "$steps.model_1.predictions",
            "predictions_b": "$steps.model.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_1_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.predictions",
        },
        {
            "type": "JsonField",
            "name": "model_2_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model_1.predictions",
        },
        {
            "type": "JsonField",
            "name": "visualization",
            "coordinates_system": "own",
            "selector": "$steps.model_comparison_visualization.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with multiple models",
    use_case_title="Comparison of detection models predictions",
    use_case_description="""
This example showcases how to compare predictions from two different models using Workflows and 
Model Comparison Visualization block.
    """,
    workflow_definition=WORKFLOW_DEFINITION,
    workflow_name_in_app="two-detection-models-comparison",
)
def test_detection_plus_classification_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_1_predictions",
        "model_2_predictions",
        "visualization",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["model_1_predictions"]) == 2
    ), "Expected 2 dogs crops on input image"
    assert (
        len(result[0]["model_2_predictions"]) == 2
    ), "Expected 2 dogs crops on input image"
    assert isinstance(
        result[0]["visualization"].numpy_image, np.ndarray
    ), "Expected visualization result"
