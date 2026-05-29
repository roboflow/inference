"""Unit tests for Depth Estimation block including remote execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.depth_estimation.v1 import (
    BlockManifest,
    DepthEstimationBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def mock_model_manager():
    mock = MagicMock()
    mock.infer_from_request_sync.return_value = MagicMock(
        response={
            "normalized_depth": np.zeros((480, 640)),
            "image": MagicMock(numpy_image=np.zeros((480, 640, 3), dtype=np.uint8)),
        }
    )
    return mock


@pytest.fixture
def mock_workflow_image_data():
    start_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


def test_manifest_parsing_valid():
    data = {
        "type": "roboflow_core/depth_estimation@v1",
        "name": "my_depth_step",
        "images": "$inputs.image",
        "model_version": "depth-anything-v3/small",
    }
    result = BlockManifest.model_validate(data)
    assert result.type == "roboflow_core/depth_estimation@v1"
    assert result.model_version == "depth-anything-v3/small"


def test_manifest_parsing_with_default_model():
    data = {
        "type": "roboflow_core/depth_estimation@v1",
        "name": "my_depth_step",
        "images": "$inputs.image",
    }
    result = BlockManifest.model_validate(data)
    assert result.model_version == "depth-anything-v3/small"


def test_run_locally(mock_model_manager, mock_workflow_image_data):
    block = DepthEstimationBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_version="depth-anything-v3/small",
    )

    assert len(result) == 1
    mock_model_manager.add_model.assert_called_once()
    mock_model_manager.infer_from_request_sync.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.depth_estimation.v1.InferenceHTTPClient"
)
def test_run_remotely_calls_depth_estimation(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    """Test that remote execution uses the depth_estimation client method."""
    mock_client = MagicMock()
    mock_client.depth_estimation.return_value = {
        "normalized_depth": [[0.1, 0.2], [0.3, 0.4]],
        "image": "0000",  # hex-encoded empty image
    }
    mock_client_cls.return_value = mock_client

    block = DepthEstimationBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_version="depth-anything-v3/small",
    )

    assert len(result) == 1
    assert "normalized_depth" in result[0]
    assert "image" in result[0]
    mock_client.depth_estimation.assert_called_once()
