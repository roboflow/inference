import numpy as np
import pytest
import supervision as sv

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.query_language.errors import (
    EvaluationEngineError,
)
from inference.core.workflows.errors import RuntimeInputError, StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

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
                    "class_map": {
                        "orange": "fruit",
                        "banana": "fruit",
                        "apple": "fruit"
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
EXPECTED_RENAMED_CLASSES = np.array(
    [
        "fruit",
        "fruit",
        "fruit",
        "fruit",
        "fruit"
    ]
)


def test_class_rename_workflow_to_have_correct_classes(
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
        workflow_definition=CLASS_RENAME_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
            "model_id": "yolov8n-640",
        }
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"

    original_predictions: sv.Detections = result[0]["original_predictions"]
    renamed_predictions: sv.Detections = result[0]["renamed_predictions"]

    assert len(original_predictions) == len(EXPECTED_ORIGINAL_CLASSES), "length of original predictions match expected length"
    assert len(renamed_predictions) == len(EXPECTED_RENAMED_CLASSES), "length of renamed predictions match expected length "

    assert np.array_equal(EXPECTED_ORIGINAL_CLASSES, original_predictions.data["class_name"]), "Expected original classes to match predicted classes"
    assert np.array_equal(EXPECTED_RENAMED_CLASSES, renamed_predictions.data["class_name"]), "Expected renamed classes to match block class renaming"