import pytest
import numpy as np
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
import json
import copy

FLORENCE2_GROUNDED_CLASSIFICATION_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "detection-grounded-classification",
            "grounding_detection": "$steps.model_1.predictions",
            "grounding_selection_mode": "most-confident",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "model_1",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        }
    ],
}


def test_florence2_grounded_classification(
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
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE2_GROUNDED_CLASSIFICATION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"

    assert json.loads(result[0]["model_predictions"]["raw_output"]).startswith(
        "dog"
    ), "Expected dog to be output by florence2"


FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "detection-grounded-instance-segmentation",
            "grounding_detection": "$steps.model_1.predictions",
            "grounding_selection_mode": "most-confident",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "model_1",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        }
    ],
}


def test_florence2_grounded_instance_segmentation(
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
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"
    polygons = json.loads(result[0]["model_predictions"]["raw_output"])["polygons"]
    assert (
        abs(polygons[0][0][0] - 326.0799865722656) < 0.1
    ), "Expected specific inst seg response"


FLORENCE2_GROUNDED_CAPTION_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "detection-grounded-caption",
            "grounding_detection": "$steps.model_1.predictions",
            "grounding_selection_mode": "most-confident",
        },
        {
            "type": "roboflow_core/roboflow_object_detection_model@v1",
            "name": "model_1",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        }
    ],
}


def test_florence2_grounded_caption(
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
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE2_GROUNDED_CAPTION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"

    assert json.loads(result[0]["model_predictions"]["raw_output"]).startswith(
        "dog"
    ), "Expected dog to be output by florence2"


FLORENCE_OD_TASK_TYPES = [
    "object-detection",
    "open-vocabulary-object-detection",
    "object-detection-and-caption",
    "phrase-grounded-object-detection",
    "region-proposal",
    "ocr-with-text-detection",
]


FLORENCE_VLM_AS_DET_VISUALIZE_DEF = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "object-detection",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v1",
            "name": "vlm_as_detector",
            "image": "$inputs.image",
            "vlm_output": "$steps.model.raw_output",
            "classes": "$steps.model.classes",
            "model_type": "florence-2",
            "task_type": "object-detection",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bounding_box_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.vlm_as_detector.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        },
        {
            "type": "JsonField",
            "name": "vlm_as_detector",
            "coordinates_system": "own",
            "selector": "$steps.vlm_as_detector.*",
        },
        {
            "type": "JsonField",
            "name": "bounding_box_visualization",
            "coordinates_system": "own",
            "selector": "$steps.bounding_box_visualization.image",
        },
    ],
}


def make_visualize_workflow(task_type):
    wf_def = copy.deepcopy(FLORENCE_VLM_AS_DET_VISUALIZE_DEF)
    if task_type == "phrase-grounded-object-detection":
        wf_def["steps"][0]["prompt"] = "dog"
    elif task_type == "open-vocabulary-object-detection":
        wf_def["steps"][0]["classes"] = ["dog"]
    wf_def["steps"][0]["task_type"] = task_type
    wf_def["steps"][1]["task_type"] = task_type
    return wf_def


@pytest.mark.parametrize("task_type", FLORENCE_OD_TASK_TYPES)
def test_florence_visualization_with_vlm_as_detector(
    task_type,
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
    execution_engine = ExecutionEngine.init(
        workflow_definition=make_visualize_workflow(task_type),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert result[0]["bounding_box_visualization"] is not None
