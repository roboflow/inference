import cv2
import numpy as np
import pytest

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Run the real estimator and use its homography to check the canvas without
    # pinning the dimensions to one platform's SIFT/RANSAC result.
    homographies = []
    find_homography = cv2.findHomography

    def record_homography(**kwargs):
        homography, inliers = find_homography(**kwargs)
        homographies.append(homography)
        return homography, inliers

    monkeypatch.setattr(cv2, "findHomography", record_homography)
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_STITCH_IMAGES,
        init_parameters={},
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    try:
        result = execution_engine.run(
            runtime_parameters={
                "image1": stitch_left_image,
                "image2": stitch_right_image,
                "count_of_best_matches_per_query_descriptor": 2,
                "max_allowed_reprojection_error": 3,
            }
        )
    finally:
        execution_engine.close()

    assert isinstance(result, list)
    assert len(result) == 1
    assert set(result[0]) == {"stitched_image"}
    panorama = result[0]["stitched_image"].numpy_image
    assert stitch_left_image.shape == stitch_right_image.shape
    height, width, channels = stitch_left_image.shape
    (homography,) = homographies
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]])
    projected = corners @ homography.T
    projected_xy = (projected[:, :2] / projected[:, 2:]).astype(np.float32)
    all_corners = np.vstack((corners[:, :2], projected_xy.astype(int)))
    lower_x, lower_y = all_corners.min(axis=0)
    upper_x, upper_y = all_corners.max(axis=0)

    assert panorama.shape == (upper_y - lower_y, upper_x - lower_x, channels)
    anchor = panorama[-lower_y : height - lower_y, -lower_x : width - lower_x]
    assert np.array_equal(anchor, stitch_left_image) or np.array_equal(
        anchor, stitch_right_image
    )


@pytest.mark.parametrize("swap_inputs", [False, True])
def test_workflow_stitch_preserves_nonoverlapping_edges(swap_inputs: bool) -> None:
    """The panorama retains a marker in each source image's nonoverlapping region."""
    scene = np.random.default_rng(4).integers(
        12, 240, size=(192, 340, 3), dtype=np.uint8
    )
    left_marker = (17, 250, 77)
    right_marker = (233, 31, 205)
    scene[60:78, 10:28] = left_marker
    scene[110:128, 306:324] = right_marker
    first, second = scene[:, :240].copy(), scene[:, 100:].copy()
    if swap_inputs:
        first, second = second, first

    engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_STITCH_IMAGES, init_parameters={}
    )
    try:
        results = engine.run(
            runtime_parameters={
                "image1": first,
                "image2": second,
                "count_of_best_matches_per_query_descriptor": 2,
                "max_allowed_reprojection_error": 3,
            },
        )
    finally:
        engine.close()

    panorama = results[0]["stitched_image"].numpy_image
    assert panorama.shape[1] >= 330
    for marker in (left_marker, right_marker):
        assert np.count_nonzero(np.all(panorama == marker, axis=-1)) >= 200
