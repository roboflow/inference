import matplotlib.pyplot as plt
import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_PIXELS_COUNTING = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/dominant_color@v1",
            "name": "dominant_color",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/pixel_color_count@v1",  # Correct type
            "name": "pixelation",
            "image": "$inputs.image",
            "target_color": "$steps.dominant_color.rgb_color",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "matching_pixels_count",
            "coordinates_system": "own",
            "selector": "$steps.pixelation.matching_pixels_count",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow calculating pixels with dominant color",
    use_case_description="""
This example shows how Dominant Color block and Pixel Color Count block can be used together.

First, dominant color gets detected and then number of pixels with that color is calculated.
    """,
    workflow_definition=WORKFLOW_WITH_PIXELS_COUNTING,
    workflow_name_in_app="pixels-counting",
)
def test_workflow_with_color_of_pixels_counting(
    model_manager: ModelManager,
) -> None:
    """
    In this test set we check how classical pattern matching block integrates with
    other blocks that accept sv.Detections.

    Please point out that a single image is passed as template, and batch of images
    are passed as images to look for template. This workflow does also validate
    Execution Engine capabilities to broadcast batch-oriented inputs properly.
    """
    # given
    red_image = np.zeros((200, 200, 3), dtype=np.uint8)
    red_image[0:190, 0:190] = (0, 0, 255)
    green_image = np.zeros((200, 200, 3), dtype=np.uint8)
    green_image[0:190, 0:180] = (0, 255, 0)
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PIXELS_COUNTING,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [red_image, green_image],
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided, so two outputs expected"
    assert set(result[0].keys()) == {
        "matching_pixels_count",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "matching_pixels_count",
    }, "Expected all declared outputs to be delivered"
    assert (
        result[0]["matching_pixels_count"] == 190 * 190
    ), "Expected 190*190 red pixels in first image, as that is how image was prepared"
    assert (
        result[1]["matching_pixels_count"] == 190 * 180
    ), "Expected 190*180 green pixels in second image, as that is how image was prepared"
