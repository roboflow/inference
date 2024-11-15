import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.transformations.stitch_images.v1 import (
    OUTPUT_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_STITCH_IMAGES = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image1"},
        {"type": "InferenceImage", "name": "image2"},
        {
            "type": "InferenceParameter",
            "name": "count_of_best_matches_per_query_descriptor",
        },
        {"type": "InferenceParameter", "name": "max_allowed_reprojection_error"},
    ],
    "steps": [
        {
            "type": "roboflow_core/stitch_images@v1",
            "name": "stitch_images",
            "image1": "$inputs.image1",
            "image2": "$inputs.image2",
            "count_of_best_matches_per_query_descriptor": "$inputs.count_of_best_matches_per_query_descriptor",
            "max_allowed_reprojection_error": "$inputs.max_allowed_reprojection_error",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "stitched_image",
            "selector": f"$steps.stitch_images.{OUTPUT_KEY}",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow stitching images",
    use_case_description="""
In this example two images of the same scene are stitched together.
Given enough shared details order of the images does not influence final result.

Please note that images need to have enough common details for the algorithm to stitch them properly.
    """,
    workflow_definition=WORKFLOW_STITCH_IMAGES,
    workflow_name_in_app="stitch_images",
)
def test_workflow_with_classical_pattern_matching(
    model_manager: ModelManager,
    stitch_left_image: np.ndarray,
    stitch_right_image: np.ndarray,
) -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_STITCH_IMAGES,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image1": stitch_left_image,
            "image2": stitch_right_image,
            "count_of_best_matches_per_query_descriptor": 2,
            "max_allowed_reprojection_error": 3,
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "One set of images provided, so one output expected"
    assert set(result[0].keys()) == {
        "stitched_image",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["stitched_image"].numpy_image.shape == (
        2918,
        2034,
        3,
    ), "Expected result image shape must match (2918, 2034, 3)"
