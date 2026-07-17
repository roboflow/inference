from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.sinks.roboflow.dataset_upload import (
    v2,
    v2_tensor,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.tensor_input_utils import (
    numpy_image_as_tensor,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the loader registers the tensor-native
# RoboflowDatasetUploadBlockV2 from `v2_tensor`, whose run() calls
# `v2_tensor.maybe_register_datapoint_at_roboflow`. The numpy test below patches the
# seam on the `v2` module, which is no longer the executing module when the flag is
# on (so the real, network-hitting code path runs). The flag-on parity test patches
# the equivalent seam on `v2_tensor` and asserts the same semantic result.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="patches v2.maybe_register_datapoint_at_roboflow; under "
    "ENABLE_TENSOR_DATA_REPRESENTATION the executing block is v2_tensor — see the "
    "*_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

WORKFLOW_WITH_DATASET_UPLOAD_METADATA_SELECTOR = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "location"},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_dataset_upload@v2",
            "name": "data_collection",
            "images": "$inputs.image",
            "predictions": None,
            "target_project": "my_project",
            "usage_quota_name": "my_quota",
            "data_percentage": 100.0,
            "persist_predictions": True,
            "minutely_usage_limit": 10,
            "hourly_usage_limit": 100,
            "daily_usage_limit": 1000,
            "max_image_size": (100, 200),
            "compression_level": 95,
            "registration_tags": [],
            "disable_sink": False,
            "fire_and_forget": False,
            "labeling_batch_prefix": "my_batch",
            "labeling_batches_recreation_frequency": "never",
            "metadata": {"location": "$inputs.location", "source": "edge_camera"},
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


@_NUMPY_ONLY
@mock.patch.object(v2, "maybe_register_datapoint_at_roboflow")
def test_workflow_with_dataset_upload_metadata_selector_inside_dict_for_batch(
    maybe_register_datapoint_at_roboflow_mock: MagicMock,
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    maybe_register_datapoint_at_roboflow_mock.return_value = False, "OK"
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": "my_api_key",
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_DATASET_UPLOAD_METADATA_SELECTOR,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
            "location": "warehouse_a",
        }
    )

    # then
    assert len(result) == 2, "Expected one output per input image"
    assert result[0]["registration_message"] == "OK"
    assert result[1]["registration_message"] == "OK"
    assert maybe_register_datapoint_at_roboflow_mock.call_count == 2
    calls = maybe_register_datapoint_at_roboflow_mock.call_args_list
    assert calls[0].kwargs["metadata"] == {
        "location": "warehouse_a",
        "source": "edge_camera",
    }
    assert calls[1].kwargs["metadata"] == {
        "location": "warehouse_a",
        "source": "edge_camera",
    }


@_TENSOR_ONLY
@mock.patch.object(v2_tensor, "maybe_register_datapoint_at_roboflow")
def test_workflow_with_dataset_upload_metadata_selector_inside_dict_for_batch_with_tensor_input(
    maybe_register_datapoint_at_roboflow_mock: MagicMock,
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # Same as
    # test_workflow_with_dataset_upload_metadata_selector_inside_dict_for_batch_tensor_native,
    # but each image arrives ALREADY materialised as a CHW RGB device tensor
    # (is_tensor_materialised() == True), so the block runs its on-device tensor path.
    # given
    maybe_register_datapoint_at_roboflow_mock.return_value = False, "OK"
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": "my_api_key",
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_DATASET_UPLOAD_METADATA_SELECTOR,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [
                numpy_image_as_tensor(dogs_image),
                numpy_image_as_tensor(dogs_image),
            ],
            "location": "warehouse_a",
        }
    )

    # then
    assert len(result) == 2, "Expected one output per input image"
    assert result[0]["registration_message"] == "OK"
    assert result[1]["registration_message"] == "OK"
    assert maybe_register_datapoint_at_roboflow_mock.call_count == 2
    calls = maybe_register_datapoint_at_roboflow_mock.call_args_list
    assert calls[0].kwargs["metadata"] == {
        "location": "warehouse_a",
        "source": "edge_camera",
    }
    assert calls[1].kwargs["metadata"] == {
        "location": "warehouse_a",
        "source": "edge_camera",
    }


@_TENSOR_ONLY
@mock.patch.object(v2_tensor, "maybe_register_datapoint_at_roboflow")
def test_workflow_with_dataset_upload_metadata_selector_inside_dict_for_batch_tensor_native(
    maybe_register_datapoint_at_roboflow_mock: MagicMock,
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    maybe_register_datapoint_at_roboflow_mock.return_value = False, "OK"
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": "my_api_key",
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_DATASET_UPLOAD_METADATA_SELECTOR,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, dogs_image],
            "location": "warehouse_a",
        }
    )

    # then
    assert len(result) == 2, "Expected one output per input image"
    assert result[0]["registration_message"] == "OK"
    assert result[1]["registration_message"] == "OK"
    assert maybe_register_datapoint_at_roboflow_mock.call_count == 2
    calls = maybe_register_datapoint_at_roboflow_mock.call_args_list
    assert calls[0].kwargs["metadata"] == {
        "location": "warehouse_a",
        "source": "edge_camera",
    }
    assert calls[1].kwargs["metadata"] == {
        "location": "warehouse_a",
        "source": "edge_camera",
    }
