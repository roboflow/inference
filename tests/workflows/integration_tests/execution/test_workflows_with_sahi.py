import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

SAHI_WORKFLOW = {
    "version": "1.0.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "overlap_filtering_strategy"},
    ],
    "steps": [
        {
            "type": "roboflow_core/image_slicer@v1",
            "name": "image_slicer",
            "image": "$inputs.image",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "detection",
            "image": "$steps.image_slicer.crops",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/detections_stitch@v1",
            "name": "stitch",
            "predictions": "$steps.detection.predictions",
            "overlap_filtering_strategy": "$inputs.overlap_filtering_strategy",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own"
        },
    ],
}


def test_sahi_workflow_with_none_as_filtering_strategy(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=SAHI_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "overlap_filtering_strategy": "none",
        }
    )

    # then
    print(result[0]["predictions"].data.keys())
    print(result[0]["predictions"].data["root_parent_coordinates"])
    raise Exception()