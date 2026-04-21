import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_EDGE_REFINEMENT = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/convert_grayscale@v1",
            "name": "gray_image",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/image_blur@v1",
            "name": "blurred_image",
            "image": "$steps.gray_image.image",
        },
        {
            "type": "roboflow_core/threshold@v1",
            "name": "thresholded_image",
            "image": "$steps.blurred_image.image",
            "thresh_value": 127,
            "threshold_type": "binary",
        },
        {
            "type": "roboflow_core/contours_detection@v1",
            "name": "detect_objects",
            "image": "$steps.thresholded_image.image",
            "raw_image": "$inputs.image",
            "line_thickness": 2,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "number_objects",
            "coordinates_system": "own",
            "selector": "$steps.detect_objects.number_contours",
        },
        {
            "type": "JsonField",
            "name": "image_with_contours",
            "coordinates_system": "own",
            "selector": "$steps.detect_objects.image",
        },
    ],
}


WORKFLOW_WITH_EDGE_REFINEMENT_SIMPLE = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/convert_grayscale@v1",
            "name": "gray_image",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "image",
            "coordinates_system": "own",
            "selector": "$steps.gray_image.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow with edge-based contour detection",
    use_case_description="""
This workflow demonstrates edge detection combined with contour detection,
using classical computer vision techniques for object boundary refinement.
    """,
    workflow_definition=WORKFLOW_WITH_EDGE_REFINEMENT,
    workflow_name_in_app="edge-contour-detection",
)
def test_workflow_with_edge_detection_preprocessing(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_EDGE_REFINEMENT,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        image=dogs_image,
    )

    # then
    assert "number_objects" in result
    assert "image_with_contours" in result
    assert result["number_objects"] > 0


def test_workflow_with_simple_preprocessing(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_EDGE_REFINEMENT_SIMPLE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        image=dogs_image,
    )

    # then
    assert "image" in result
