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

QWEN_IMAGE_EDIT_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
    ],
    "steps": [
        {
            "type": "roboflow_core/qwen_image_edit@v1",
            "name": "qwen_image_edit",
            "images": "$inputs.image",
            "prompt": "make the background a clear blue sky",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "edited_image",
            "coordinates_system": "own",
            "selector": "$steps.qwen_image_edit.*",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Image Editing",
    use_case_title="Qwen-Image-Edit",
    use_case_description="""
**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED GPU DEPLOYMENT**

Use Qwen-Image-Edit to transform an image from a natural-language instruction.
The block takes a source image and a text prompt describing the desired edit and
returns the edited image, which can be passed to downstream blocks or surfaced as
a workflow output.
    """,
    workflow_definition=QWEN_IMAGE_EDIT_WORKFLOW_DEFINITION,
    workflow_name_in_app="qwen_image_edit",
)
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_QWEN_IMAGE_EDIT_TEST", True)),
    reason=(
        "Skipping Qwen-Image-Edit test (requires a local GPU; with default "
        "config the first run downloads tens of GB of weights from HuggingFace). "
        "Set SKIP_QWEN_IMAGE_EDIT_TEST=False to opt in."
    ),
)
def test_qwen_image_edit_inference(
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
        workflow_definition=QWEN_IMAGE_EDIT_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "edited_image",
    }, "Expected all declared outputs to be delivered"
    edited = result[0]["edited_image"]["image"]
    assert isinstance(
        edited, WorkflowImageData
    ), "Expected the edited output to be an image"
    assert edited.numpy_image.shape[2] == 3, "Expected a 3-channel edited image"


QWEN_IMAGE_EDIT_PARAMETERIZED_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "prompt"},
        {"type": "WorkflowParameter", "name": "steps", "default_value": 4},
        {"type": "WorkflowParameter", "name": "seed", "default_value": 42},
    ],
    "steps": [
        {
            "type": "roboflow_core/qwen_image_edit@v1",
            "name": "qwen_image_edit",
            "images": "$inputs.image",
            "prompt": "$inputs.prompt",
            "use_lightning_lora": True,
            "num_inference_steps": "$inputs.steps",
            "seed": "$inputs.seed",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "edited_image",
            "coordinates_system": "own",
            "selector": "$steps.qwen_image_edit.image",
        }
    ],
}


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_QWEN_IMAGE_EDIT_TEST", True)),
    reason=(
        "Skipping Qwen-Image-Edit test (requires a local GPU; with default "
        "config the first run downloads tens of GB of weights from HuggingFace). "
        "Set SKIP_QWEN_IMAGE_EDIT_TEST=False to opt in."
    ),
)
def test_qwen_image_edit_with_prompt_input_and_custom_params(
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
        workflow_definition=QWEN_IMAGE_EDIT_PARAMETERIZED_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "prompt": "add a red collar to each dog",
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "edited_image",
    }, "Expected all declared outputs to be delivered"
    assert isinstance(
        result[0]["edited_image"], WorkflowImageData
    ), "Expected the edited output to be an image"
