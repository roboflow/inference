import numpy as np

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
            "name": "model_id",
            "default_value": "deepfashion2-1000-items/1",
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_multi_label_classification_model@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "$inputs.model_id",
        },
        {
            "type": "roboflow_core/classification_label_visualization@v1",
            "name": "classification_label_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.model.predictions",
            "text": "Class and Confidence",
            "text_position": "CENTER",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        },
        {
            "type": "JsonField",
            "name": "classification_label_visualization",
            "selector": "$steps.classification_label_visualization.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with visualization blocks",
    use_case_title="Workflow with multi-label classification label visualization",
    workflow_name_in_app="multi-label-classification-visualization",
    use_case_description="""
This workflow demonstrates how to visualize the predictions of a multi-label classification model. 
It is compatable with single-label and multi-label classification tasks. It is also 
compatible with supervision visualization fields like text position, color, scale, etc.
    """,
    workflow_definition=WORKFLOW_DEFINITION,
)
def test_classification_multi_label_visualization_workflow_when_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": dogs_image})

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
        "classification_label_visualization",
    }, "Expected all declared outputs to be delivered"
