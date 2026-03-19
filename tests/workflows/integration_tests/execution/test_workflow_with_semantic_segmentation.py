import numpy as np
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine


SEMANTIC_SEGMENTATION_WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": "deep-lab-v3-plus/2",
        },
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_semantic_segmentation_model@v1",
            "name": "segmentation",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.segmentation.predictions",
        },
        {
            "type": "JsonField",
            "name": "model_id",
            "selector": "$steps.segmentation.model_id",
        },
    ],
}


def test_semantic_segmentation_workflow_when_single_image_provided(
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
        workflow_definition=SEMANTIC_SEGMENTATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": dogs_image},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    assert set(result[0].keys()) == {
        "predictions",
        "model_id",
    }, "Expected all declared outputs to be delivered"
    detections: sv.Detections = result[0]["predictions"]
    assert isinstance(
        detections, sv.Detections
    ), "Expected predictions to be sv.Detections"
    assert len(detections) > 0, "Expected at least one class detected"
    assert detections.xyxy is not None, "Expected bounding boxes"
    assert detections.class_id is not None, "Expected class IDs"
    assert detections.confidence is not None, "Expected confidence scores"
    assert "class_name" in detections.data, "Expected class_name in detections data"
    assert (
        "rle_mask" in detections.data
    ), "Expected rle_mask in detections data for semantic segmentation"
    for rle in detections.data["rle_mask"]:
        assert "size" in rle, "Expected 'size' key in RLE mask"
        assert "counts" in rle, "Expected 'counts' key in RLE mask"
    assert (
        result[0]["model_id"] == "deep-lab-v3-plus/2"
    ), "Expected model_id output to match input"


def test_semantic_segmentation_workflow_when_batch_input_provided(
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
        workflow_definition=SEMANTIC_SEGMENTATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": [dogs_image, dogs_image]},
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 2, "Two images provided - two outputs expected"
    for i in range(2):
        detections: sv.Detections = result[i]["predictions"]
        assert isinstance(
            detections, sv.Detections
        ), f"Expected predictions for image {i} to be sv.Detections"
        assert len(detections) > 0, f"Expected at least one class detected for image {i}"
        assert "rle_mask" in detections.data, (
            f"Expected rle_mask in detections data for image {i}"
        )


def test_semantic_segmentation_workflow_with_serialization(
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
        workflow_definition=SEMANTIC_SEGMENTATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": dogs_image},
        serialize_results=True,
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    predictions = result[0]["predictions"]
    assert isinstance(predictions, dict), "Expected serialized predictions to be a dict"
    assert "predictions" in predictions, (
        "Expected 'predictions' key in serialized output"
    )
    assert isinstance(predictions["predictions"], list), (
        "Expected serialized predictions list"
    )
    for pred in predictions["predictions"]:
        assert "class" in pred, "Expected 'class' in each serialized prediction"
        assert "confidence" in pred, "Expected 'confidence' in each serialized prediction"
        assert "rle_mask" in pred, (
            "Expected 'rle_mask' in each serialized semantic segmentation prediction"
        )
