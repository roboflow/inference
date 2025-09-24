import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_CONTOUR_DETECTION = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/convert_grayscale@v1",
            "name": "image_convert_grayscale",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/image_blur@v1",
            "name": "image_blur",
            "image": "$steps.image_convert_grayscale.image",
        },
        {
            "type": "roboflow_core/threshold@v1",
            "name": "image_threshold",
            "image": "$steps.image_blur.image",
            "thresh_value": 200,
            "threshold_type": "binary_inv",
        },
        {
            "type": "roboflow_core/contours_detection@v1",
            "name": "image_contours",
            "image": "$steps.image_threshold.image",
            "raw_image": "$inputs.image",
            "line_thickness": 5,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "number_contours",
            "coordinates_system": "own",
            "selector": "$steps.image_contours.number_contours",
        },
        {
            "type": "JsonField",
            "name": "contour_image",
            "coordinates_system": "own",
            "selector": "$steps.image_contours.image",
        },
        {
            "type": "JsonField",
            "name": "contours",
            "coordinates_system": "own",
            "selector": "$steps.image_contours.contours",
        },
        {
            "type": "JsonField",
            "name": "grayscale_image",
            "coordinates_system": "own",
            "selector": "$steps.image_convert_grayscale.image",
        },
        {
            "type": "JsonField",
            "name": "blurred_image",
            "coordinates_system": "own",
            "selector": "$steps.image_blur.image",
        },
        {
            "type": "JsonField",
            "name": "thresholded_image",
            "coordinates_system": "own",
            "selector": "$steps.image_threshold.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow detecting contours",
    use_case_description="""
In this example we show how classical contour detection works in cooperation
with blocks performing its pre-processing (conversion to gray and blur).
    """,
    workflow_definition=WORKFLOW_WITH_CONTOUR_DETECTION,
    workflow_name_in_app="contours-detection",
)
def test_workflow_with_classical_contour_detection(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CONTOUR_DETECTION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One image provided, so one output expected"
    assert set(result[0].keys()) == {
        "number_contours",
        "contour_image",
        "contours",
        "grayscale_image",
        "blurred_image",
        "thresholded_image",
    }, "Expected all declared outputs to be delivered"
    assert not np.array_equal(
        result[0]["grayscale_image"].numpy_image, dogs_image
    ), "Expected grayscale image to be different from input image"
    assert not np.array_equal(
        result[0]["blurred_image"].numpy_image, dogs_image
    ), "Expected blur image to be different from input image"
    assert not np.array_equal(
        result[0]["thresholded_image"].numpy_image, dogs_image
    ), "Expected threshold image to be different from input image"
    assert not np.array_equal(
        result[0]["contour_image"].numpy_image, dogs_image
    ), "Expected contour outline image to be different from input image"
    assert (
        result[0]["number_contours"] == 27
    ), "Expected number of contours to be 27 for dogs image"
    assert (
        result[0]["number_contours"] == 27
    ), "Expected number of contours to be 27 for dogs image"
    assert (
        len(result[0]["contours"]) == 27
    ), "Expected length of contours to be 27 for dogs image"
