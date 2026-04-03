import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

WORKFLOW_WITH_DYNAMIC_CROP_AND_STITCH = {
    "version": "1.0.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "car_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["car"],
        },
        {
            "type": "roboflow_core/dynamic_crop@v1",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.car_detection.predictions",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "plates_detection",
            "image": "$steps.cropping.crops",
            "model_id": "vehicle-registration-plates-trudk/2",
        },
        {
            "type": "roboflow_core/detections_stitch@v1",
            "name": "stitch",
            "reference_image": "$inputs.image",
            "predictions": "$steps.plates_detection.predictions",
            "overlap_filtering_strategy": "nms",
        },
        {
            "type": "DetectionsConsensus",
            "name": "consensus",
            "predictions_batches": [
                "$steps.car_detection.predictions",
                "$steps.stitch.predictions",
            ],
            "required_votes": 1,
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bbox_visualiser",
            "predictions": "$steps.consensus.predictions",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.consensus.predictions",
        },
        {
            "type": "JsonField",
            "name": "visualisation",
            "selector": "$steps.bbox_visualiser.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with visualization blocks",
    use_case_title="Predictions from different models visualised together",
    use_case_description="""
This workflow showcases how predictions from different models (even from nested 
batches created from input images) may be visualised together.

Our scenario covers:

- Detecting cars using YOLOv8 model

- Dynamically cropping input images to run secondary model (license plates detector) for each 
car instance

- Stitching together all predictions for licence plates into single prediction

- Fusing cars detections and license plates detections into single prediction

- Visualizing final predictions
    """,
    workflow_definition=WORKFLOW_WITH_DYNAMIC_CROP_AND_STITCH,
    workflow_name_in_app="visualisation-blocks",
)
def test_workflow_with_stitch_and_dynamic_crop(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_DYNAMIC_CROP_AND_STITCH,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image],
        }
    )

    # then
    assert np.allclose(
        result[0]["predictions"].xyxy,
        np.array(
            [
                [114, 480, 346, 661],
                [766, 385, 1752, 985],
                [422, 494, 583, 614],
                [192, 594, 263, 611],
                [1380, 778, 1608, 842],
                [491, 570, 535, 582],
            ]
        ),
        atol=2,
    ), "Expected bounding boxes to be exactly the same as when test was created"
    assert result[0]["predictions"].class_id.tolist() == [
        2,
        2,
        2,
        0,
        0,
        0,
    ], "Expected predicted classes to be exactly the same as when test was created"
