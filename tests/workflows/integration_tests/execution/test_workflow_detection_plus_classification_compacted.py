import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

COMPACTED_DETECTION_PLUS_CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/detection_and_classification@v1",
            "name": "detections_and_classification",
            "image": "$inputs.image",
            "detector_model_id": "yolov8n-640",
            "detector_class_filter": ["dog"],
            "classifier_model_id": "dog-breed-xpaq6/1",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.detections_and_classification.predictions",
        },
    ],
}


def test_compacted_detection_plus_classification_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    crowd_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=COMPACTED_DETECTION_PLUS_CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, crowd_image],
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input image"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs crops on input image, hence 2 nested classification results"
    assert [result[0]["predictions"][0]["top"], result[0]["predictions"][1]["top"]] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected predictions to be as measured in reference run"
    assert (
            len(result[1]["predictions"]) == 0
    ), "Expected 0 dogs crops on input image, hence 0 nested classification results"
