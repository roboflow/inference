"""End-to-end test of `roboflow_core/detections_nearest_neighbor@v1` plugged
into a real workflow with two upstream `RoboflowObjectDetectionModel` steps.

Two detection runs at different confidence thresholds on the same crowd image
give a query set and a target set of person detections; the
DetectionsNearestNeighbor block then matches each query detection to its
nearest target detection(s). Assertions are shape/type based rather than
numeric to keep the test stable across model-weight nudges.
"""

import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

DETECTIONS_NEAREST_NEIGHBOR_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "yolov8n-640",
        },
        {
            "type": "WorkflowParameter",
            "name": "target_confidence",
            "default_value": 0.5,
        },
    ],
    "steps": [
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "query_detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": 0.3,
        },
        {
            "type": "RoboflowObjectDetectionModel",
            "name": "target_detection",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.target_confidence",
        },
        {
            "type": "roboflow_core/detections_nearest_neighbor@v1",
            "name": "nearest_neighbor",
            "query_predictions": "$steps.query_detection.predictions",
            "target_predictions": "$steps.target_detection.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "query_predictions",
            "selector": "$steps.nearest_neighbor.query_predictions",
        },
        {
            "type": "JsonField",
            "name": "matched_query_detections",
            "selector": "$steps.nearest_neighbor.matched_query_detections",
        },
        {
            "type": "JsonField",
            "name": "matched_target_detections",
            "selector": "$steps.nearest_neighbor.matched_target_detections",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with multiple models",
    use_case_title="Workflow presenting nearest-neighbor detection matching",
    use_case_description="""
This workflow runs the same object-detection model against one input image at
two different confidence thresholds, then feeds the two detection sets into
the Nearest Neighbor Detection Match block, which reports the nearest target
detection (or detections, on a tie) for every query detection, along with two
flat, aligned detection sets covering every matched query-target pair.

This pattern is useful whenever you need to relate detections from two
separate sources by spatial proximity - a broad detector vs. a stricter one, a
primary detection set vs. a reference/landmark set, etc.
    """,
    workflow_definition=DETECTIONS_NEAREST_NEIGHBOR_WORKFLOW,
    workflow_name_in_app="detections-nearest-neighbor",
)
def test_detections_nearest_neighbor_workflow(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_NEAREST_NEIGHBOR_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "target_confidence": 0.5,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected one output element for one input image"
    assert set(result[0].keys()) == {
        "query_predictions",
        "matched_query_detections",
        "matched_target_detections",
    }

    query_predictions = result[0]["query_predictions"]["predictions"]
    matched_query_detections = result[0]["matched_query_detections"]["predictions"]
    matched_target_detections = result[0]["matched_target_detections"]["predictions"]

    assert (
        len(query_predictions) > 0
    ), "Expected at least one person detection at confidence=0.3 on the crowd image."
    assert len(matched_query_detections) == len(matched_target_detections), (
        "matched_query_detections and matched_target_detections must stay the "
        "same length and index-aligned."
    )
    assert len(matched_query_detections) > 0, (
        "Expected at least one query detection to find a nearest target "
        "detection on the crowd image."
    )

    for detection in query_predictions:
        assert "nearest_target_distance" in detection
        distance = detection["nearest_target_distance"]
        assert distance is None or isinstance(distance, float)

    matched_distances = [
        detection["nearest_target_distance"] for detection in matched_query_detections
    ]
    assert all(distance is not None for distance in matched_distances), (
        "Every detection appearing in matched_query_detections must carry a "
        "real (non-None) nearest_target_distance."
    )


def test_detections_nearest_neighbor_workflow_when_target_set_is_empty(
    model_manager: ModelManager,
    crowd_image: np.ndarray,
) -> None:
    # given: target_confidence=0.999 is unreachable on this model/image, so
    # target_predictions is empty and no query detection can find a match.
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTIONS_NEAREST_NEIGHBOR_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": crowd_image,
            "model_id": "yolov8n-640",
            "target_confidence": 0.999,
        }
    )

    # then
    assert isinstance(result, list)
    assert len(result) == 1
    query_predictions = result[0]["query_predictions"]["predictions"]
    matched_query_detections = result[0]["matched_query_detections"]["predictions"]
    matched_target_detections = result[0]["matched_target_detections"]["predictions"]

    assert len(query_predictions) > 0, (
        "Expected at least one person detection at confidence=0.3 on the crowd "
        "image, even though the target set is empty."
    )
    assert len(matched_query_detections) == 0
    assert len(matched_target_detections) == 0
    for detection in query_predictions:
        assert detection["nearest_target_distance"] is None, (
            "With an empty target set every query detection must be unmatched "
            "(nearest_target_distance=None)."
        )
