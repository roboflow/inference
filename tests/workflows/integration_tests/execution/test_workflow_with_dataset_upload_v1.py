from unittest import mock

import numpy as np
import pytest

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload import (
    v1,
    v1_tensor,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine

# Under ENABLE_TENSOR_DATA_REPRESENTATION the loader registers the tensor-native block,
# which calls its own module-level `register_datapoint_at_roboflow`.
DATASET_UPLOAD_V1_MODULE = v1_tensor if ENABLE_TENSOR_DATA_REPRESENTATION else v1


def build_workflow_with_dataset_upload_v1(image_property_name: str) -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_dataset_upload@v1",
                "name": "data_collection",
                image_property_name: "$inputs.image",
                "target_project": "my_project",
                "usage_quota_name": "my_quota",
                "fire_and_forget": False,
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "registration_message",
                "selector": "$steps.data_collection.message",
            },
        ],
    }


@pytest.mark.parametrize("image_property_name", ["image", "images"])
def test_workflow_with_dataset_upload_v1_accepting_image_under_both_property_names(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    image_property_name: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": "my_api_key",
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=build_workflow_with_dataset_upload_v1(
            image_property_name=image_property_name
        ),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with mock.patch.object(
        DATASET_UPLOAD_V1_MODULE,
        "register_datapoint_at_roboflow",
        return_value=(False, "OK"),
    ) as register_datapoint_at_roboflow_mock:
        result = execution_engine.run(
            runtime_parameters={"image": [dogs_image, dogs_image]}
        )

    # then
    assert result == [
        {"registration_message": "OK"},
        {"registration_message": "OK"},
    ], "Expected one registration output per input image"
    assert register_datapoint_at_roboflow_mock.call_count == 2
