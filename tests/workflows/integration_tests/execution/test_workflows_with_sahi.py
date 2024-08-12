import matplotlib.pyplot as plt
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
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bbox_visualiser",
            "predictions": "$steps.stitch.predictions",
            "image": "$inputs.image",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "visualisation",
            "selector": "$steps.bbox_visualiser.image",
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
    # plt.imshow(result[0]["visualisation"].numpy_image)
    # plt.show()
    # plt.imshow(result[1]["visualisation"].numpy_image)
    # plt.show()

    raise Exception()


def test_sahi_workflow_with_nms_as_filtering_strategy(
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
            "overlap_filtering_strategy": "nms",
        }
    )

    # then
    plt.imshow(result[0]["visualisation"].numpy_image)
    plt.show()
    plt.imshow(result[1]["visualisation"].numpy_image)
    plt.show()

    raise Exception()


def test_sahi_workflow_with_nmm_as_filtering_strategy(
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
            "overlap_filtering_strategy": "nmm",
        }
    )

    # then
    plt.imshow(result[0]["visualisation"].numpy_image)
    plt.show()
    plt.imshow(result[1]["visualisation"].numpy_image)
    plt.show()

    raise Exception()


SAHI_WORKFLOW_THAT_WAS_PREVIOUSLY_CROPPED = {
    "version": "1.0.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "overlap_filtering_strategy"},
    ],
    "steps": [
        {
            "type": "roboflow_core/relative_statoic_crop@v1",
            "name": "static_crop",
            "image": "$inputs.image",
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.9,
            "height": 0.9,
        },
        {
            "type": "roboflow_core/image_slicer@v1",
            "name": "image_slicer",
            "image": "$steps.static_crop.crops",
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
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bbox_visualiser",
            "predictions": "$steps.stitch.predictions",
            "image": "$steps.static_crop.crops",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.stitch.predictions",
            "coordinates_system": "own",
        },
        {
            "type": "JsonField",
            "name": "visualisation",
            "selector": "$steps.bbox_visualiser.image",
        },
    ],
}


def test_sahi_workflow_that_was_initially_cropped_with_nms_as_filtering_strategy(
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
        workflow_definition=SAHI_WORKFLOW_THAT_WAS_PREVIOUSLY_CROPPED,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "overlap_filtering_strategy": "nms",
        }
    )

    # then
    plt.imshow(result[0]["visualisation"].numpy_image)
    plt.show()
    plt.imshow(result[1]["visualisation"].numpy_image)
    plt.show()

    raise Exception()
