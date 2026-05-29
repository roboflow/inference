from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.clip.v1 import (
    BlockManifest,
    ClipModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def mock_model_manager():
    # Mock a model manager that returns a predictable embedding
    mock = MagicMock()
    mock.infer_from_request_sync.return_value = MagicMock(
        embeddings=[[0.1, 0.2, 0.3]]  # Sample embedding
    )
    return mock


@pytest.fixture
def mock_workflow_image_data():
    # Create a mock WorkflowImageData instance
    start_image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


def test_manifest_parsing_valid():
    data = {
        "type": "roboflow_core/clip@v1",
        "name": "my_clip_step",
        "data": "$inputs.image",
        "version": "RN50",
    }

    result = BlockManifest.model_validate(data)
    assert result.type == "roboflow_core/clip@v1"
    assert result.name == "my_clip_step"
    assert result.data == "$inputs.image"
    assert result.version == "RN50"


def test_manifest_parsing_invalid_missing_type():
    data = {
        "name": "my_clip_step",
        "data": "$inputs.image",
        "version": "RN50",
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_manifest_parsing_invalid_data_type():
    data = {
        "type": "roboflow_core/clip@v1",
        "name": "my_clip_step",
        "data": 123,  # invalid type
        "version": "RN50",
    }
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_run_locally_with_text(mock_model_manager):
    block = ClipModelBlockV1(
        model_manager=mock_model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    # Run with text input
    result = block.run(data="Hello world", version="RN50")

    assert isinstance(result, dict)
    assert len(result["embedding"]) == 3
    assert result["embedding"] == [0.1, 0.2, 0.3]
    mock_model_manager.infer_from_request_sync.assert_called_once()


def test_run_locally_with_image(mock_model_manager, mock_workflow_image_data):
    block = ClipModelBlockV1(
        model_manager=mock_model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(data=mock_workflow_image_data, version="RN50")

    assert isinstance(result, dict)
    assert len(result["embedding"]) == 3
    assert result["embedding"] == [0.1, 0.2, 0.3]
    mock_model_manager.infer_from_request_sync.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.clip.v1.InferenceHTTPClient"
)
def test_run_remotely_with_text(mock_client_cls, mock_model_manager):
    # Mock the remote client and its return value
    mock_client = MagicMock()
    mock_client.get_clip_text_embeddings.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]]
    }
    mock_client_cls.return_value = mock_client

    block = ClipModelBlockV1(
        model_manager=mock_model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(data="Hello world", version="RN50")

    assert result["embedding"] == [0.1, 0.2, 0.3]
    mock_client.get_clip_text_embeddings.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.clip.v1.InferenceHTTPClient"
)
def test_run_remotely_with_image(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    mock_client = MagicMock()
    mock_client.get_clip_image_embeddings.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]]
    }
    mock_client_cls.return_value = mock_client

    block = ClipModelBlockV1(
        model_manager=mock_model_manager,
        api_key=None,
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(data=mock_workflow_image_data, version="RN50")

    assert result["embedding"] == [0.1, 0.2, 0.3]
    mock_client.get_clip_image_embeddings.assert_called_once()
