import numpy as np

from inference.core.env import USE_INFERENCE_MODELS, WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

# check both overlap types
OVERLAP_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/overlap@v1",
            "name": "any_overlap",
            "predictions": "$steps.model.predictions",
            "overlap_class_name": "banana",
            "overlap_type": "Any Overlap",
        },
        {
            "type": "roboflow_core/overlap@v1",
            "name": "center_overlap",
            "predictions": "$steps.model.predictions",
            "overlap_class_name": "banana",
            "overlap_type": "Center Overlap",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "center_predictions",
            "coordinates_system": "own",
            "selector": "$steps.center_overlap.overlaps",
        },
        {
            "type": "JsonField",
            "name": "any_predictions",
            "coordinates_system": "own",
            "selector": "$steps.any_overlap.overlaps",
        },
    ],
}

'''
@add_to_workflows_gallery(
    category="Workflows with classical Computer Vision methods",
    use_case_title="Workflow with dynamic zone and perspective converter",
    use_case_description="""
In this example dynamic zone with 4 vertices is calculated from detected segmentations.
Perspective correction is applied to the input image as well as to detected segmentations based on this zone.
    """,
    workflow_definition=OVERLAP_WORKFLOW,
    workflow_name_in_app="dynamic_zone_and_perspective_converter",
)
'''


def test_workflow_with_overlap_all(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OVERLAP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
        }
    )

    assert len(result) == 1, "One set of images provided, so one output expected"

    if not USE_INFERENCE_MODELS:
        # if overlap_type is "Any Overlap", both the apples and orange will overlap the banana
        any_redictions = result[0]["any_predictions"]
        assert len(any_redictions.class_id) == 4
        class_names = any_redictions.data["class_name"]
        assert "banana" not in class_names
        assert "apple" in class_names
        assert "orange" in class_names

        # if overlap_type is "Center Overlap" only the orange will overlap the banana
        any_redictions = result[0]["center_predictions"]
        assert len(any_redictions.class_id) == 1
        class_names = any_redictions.data["class_name"]
        assert "banana" not in class_names
        assert "apple" not in class_names
        assert "orange" in class_names
    else:
        # if overlap_type is "Any Overlap", both the apples and orange will overlap the banana
        any_redictions = result[0]["any_predictions"]
        assert len(any_redictions.class_id) == 5
        class_names = any_redictions.data["class_name"]
        assert "banana" not in class_names
        assert "apple" in class_names
        assert "orange" in class_names

        # if overlap_type is "Center Overlap" only the orange will overlap the apple and orange
        any_redictions = result[0]["center_predictions"]
        assert len(any_redictions.class_id) == 2
        class_names = any_redictions.data["class_name"]
        assert "banana" not in class_names
        assert "apple" in class_names
        assert "orange" in class_names
