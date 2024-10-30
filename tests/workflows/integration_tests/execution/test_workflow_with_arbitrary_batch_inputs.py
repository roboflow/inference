import numpy as np

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

TWO_STAGE_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$steps.cropping.crops",
            "model_id": "dog-breed-xpaq6/1",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.breds_classification.predictions",
        },
    ],
}


OBJECT_DETECTION_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.general_detection.*",
        },
    ],
}


CROP_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowDataBatch",
            "name": "predictions",
            "kind": ["object_detection_prediction"],
        },
    ],
    "steps": [
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$inputs.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.cropping.*",
        },
    ],
}

CLASSIFICATION_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {
            "type": "WorkflowDataBatch",
            "name": "crops",
            "kind": ["image"],
            "dimensionality": 2,
        },
    ],
    "steps": [
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$inputs.crops",
            "model_id": "dog-breed-xpaq6/1",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.breds_classification.*",
        },
    ],
}


def test_debug_execution_of_workflow_for_single_image_without_conditional_evaluation(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    end_to_end_execution_engine = ExecutionEngine.init(
        workflow_definition=TWO_STAGE_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    first_step_execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    second_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CROP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )
    third_step_execution_engine = ExecutionEngine.init(
        workflow_definition=CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    e2e_results = end_to_end_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )
    detection_results = first_step_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )
    cropping_results = second_step_execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "predictions": detection_results[0]["result"]["predictions"],
        }
    )
    classification_results = third_step_execution_engine.run(
        runtime_parameters={
            "crops": [[e["crops"] for e in cropping_results[0]["result"]]],
        }
    )
    print(classification_results)
    raise Exception()
    # assert isinstance(result, list), "Expected list to be delivered"
    # assert len(result) == 1, "Expected 1 element in the output for one input image"
    # assert set(result[0].keys()) == {
    #     "predictions",
    # }, "Expected all declared outputs to be delivered"
    # assert (
    #     len(result[0]["predictions"]) == 2
    # ), "Expected 2 dogs crops on input image, hence 2 nested classification results"
    # assert [result[0]["predictions"][0]["top"], result[0]["predictions"][1]["top"]] == [
    #     "116.Parson_russell_terrier",
    #     "131.Wirehaired_pointing_griffon",
    # ], "Expected predictions to be as measured in reference run"
