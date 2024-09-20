import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.query_language.errors import OperationError
from inference.core.workflows.execution_engine.core import ExecutionEngine


CLASS_RENAME_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {
            "type": "WorkflowImage",
            "name": "image"
        },
        {
            "type": "WorkflowParameter",
            "name": "model_id"
        },
        {
            "type": "WorkflowParameter",
            "name": "confidence",
            "default_value": 0.4
        },
        {
            "type": "WorkflowParameter",
            "name": "classes"
        },
        {
            "type": "WorkflowParameter",
            "name": "class_map"
        },
        {
            "type": "WorkflowParameter",
            "name": "strict",
            "default_value": True
        },
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "DetectionsTransformation",
            "name": "class_rename",
            "predictions": "$steps.model.predictions",
            "operations": [
                {
                    "type": "DetectionsRename",
                    "strict": True,
                    "class_map": {
                        "apple": "fruit",
                        "orange": "fruit",
                        "banana": "fruit"
                    }
                }
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "original_predictions",
            "selector": "$steps.model.predictions",
        },
        {
            "type": "JsonField",
            "name": "renamed_predictions",
            "selector": "$steps.class_rename.predictions",
        },
    ],
}

EXPECTED_ORIGINAL_CLASSES = np.array(
    [
        "apple",
        "apple",
        "apple",
        "orange",
        "banana"
    ]
)


@pytest.mark.parametrize(
    "class_map, strict, expected_renamed_classes, expected_class_ids",
    [
        (
            {"apple": "fruit", "orange": "fruit", "banana": "fruit"},
            True,
            np.array(["fruit", "fruit", "fruit", "fruit", "fruit"]),
            np.array([50, 50, 50, 50, 50]),
        ),
        (
            {"apple": "fruit", "orange": "citrus"},
            False,
            np.array(["fruit", "fruit", "fruit", "citrus", "banana"]),
            np.array([50, 50, 50, 51, 46]),
        ),
        (
            {"apple": "orange", "orange": "apple", "banana": "fruit"},
            True,
            np.array(["orange", "orange", "orange", "apple", "fruit"]),
            np.array([49, 49, 49, 47, 50]),
        ),
    ],
)
def test_class_rename_workflow_to_have_correct_classes(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
    class_map: dict,
    strict: bool,
    expected_renamed_classes: np.ndarray,
    expected_class_ids: np.ndarray,
) -> None:
    # given
    workflow_definition = CLASS_RENAME_WORKFLOW.copy()
    workflow_definition["steps"][1]["operations"][0]["class_map"] = class_map
    workflow_definition["steps"][1]["operations"][0]["strict"] = strict

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
            "model_id": "yolov8n-640",
        },
    )


    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"

    original_predictions: sv.Detections = result[0]["original_predictions"]
    renamed_predictions: sv.Detections = result[0]["renamed_predictions"]

    assert len(original_predictions) == len(EXPECTED_ORIGINAL_CLASSES), "length of original predictions match expected length"
    assert len(renamed_predictions) == len(expected_renamed_classes), "length of renamed predictions match expected length "

    assert np.array_equal(EXPECTED_ORIGINAL_CLASSES, original_predictions.data["class_name"]), "Expected original classes to match predicted classes"
    assert np.array_equal(expected_renamed_classes, renamed_predictions.data["class_name"]), "Expected renamed classes to match block class renaming"

    assert np.array_equal(expected_class_ids, renamed_predictions.class_id), "Expected renamed class ids to match block class renaming"


@pytest.mark.parametrize(
    "class_map, strict, expected_exception",
    [
        (
            {"apple": "fruit", "orange": "fruit"},
            True,
            OperationError, 
        ),
    ],
)
def test_class_rename_workflow_raises_exception_when_strict_is_true(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
    class_map: dict,
    strict: bool,
    expected_exception: type,
) -> None:
    # given
    workflow_definition = CLASS_RENAME_WORKFLOW.copy()
    workflow_definition["steps"][1]["operations"][0]["class_map"] = class_map
    workflow_definition["steps"][1]["operations"][0]["strict"] = strict

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when/then
    with pytest.raises(expected_exception):
        execution_engine.run(
            runtime_parameters={
                "image": fruit_image,
                "model_id": "yolov8n-640",
            },
        )