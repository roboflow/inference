import copy
import json
import os

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from tests.workflows.integration_tests.execution.conftest import bool_env
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

MOONDREAM2_ZERO_SHOT_OBJECT_DETECTION_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
    ],
    "steps": [
        {
            "type": "roboflow_core/moondream2@v1",
            "name": "model",
            "images": "$inputs.image",
            "prompt": "dog",
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


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Moondream 2 - object detection",
    use_case_description="""
    Use Moondream2 to detect objects in an image.

    You can pass in a prompt to the model to specify what you want to detect. The model will return a list of detection coordinates corresponding to the prompt.

    This block only works with one class at a time. This is because Moondream2 does not allow zero shot detection on more than one class at once.
    """,
    workflow_definition=MOONDREAM2_ZERO_SHOT_OBJECT_DETECTION_WORKFLOW_DEFINITION,
)
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_MOONDREAM2_TEST", False))
    or bool_env(os.getenv("USE_INFERENCE_MODELS", False)),
    reason="Skipping Moondream 2 test (either disabled or turned off due to malfunctional inference-models checkpoint)",
)
@pytest.mark.parametrize("task_type", ["phrase-grounded-object-detection"])
def test_moondream2_object_detection(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
    task_type: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MOONDREAM2_ZERO_SHOT_OBJECT_DETECTION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"image": dogs_image})

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"

    assert len(result)
    assert isinstance(result[0]["model_predictions"], dict)
