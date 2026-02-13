"""Unit tests for Gaze Detection block remote execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.gaze.v1 import (
    GazeBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def mock_model_manager():
    return MagicMock()


@pytest.fixture
def mock_workflow_image_data():
    start_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


@patch(
    "inference.core.workflows.core_steps.models.foundation.gaze.v1.InferenceHTTPClient"
)
def test_run_remotely_calls_detect_gazes(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    """Test that remote execution uses the detect_gazes client method."""
    mock_client = MagicMock()
    mock_client.detect_gazes.return_value = [
        {
            "predictions": [
                {
                    "face": {
                        "x": 100,
                        "y": 100,
                        "width": 50,
                        "height": 50,
                        "confidence": 0.95,
                        "landmarks": [
                            {"x": 90, "y": 90},
                            {"x": 110, "y": 90},
                        ],
                    },
                    "yaw": 0.1,
                    "pitch": 0.2,
                }
            ]
        }
    ]
    mock_client_cls.return_value = mock_client

    block = GazeBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        do_run_face_detection=True,
    )

    assert len(result) == 1
    assert "face_predictions" in result[0]
    assert "yaw_degrees" in result[0]
    assert "pitch_degrees" in result[0]
    mock_client.detect_gazes.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.gaze.v1.InferenceHTTPClient"
)
def test_run_remotely_selects_correct_api_mode(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    """Test that remote execution selects the correct API version."""
    mock_client = MagicMock()
    mock_client.detect_gazes.return_value = [{"predictions": []}]
    mock_client_cls.return_value = mock_client

    block = GazeBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    with patch(
        "inference.core.workflows.core_steps.models.foundation.gaze.v1.WORKFLOWS_REMOTE_API_TARGET",
        "self-hosted",
    ):
        block.run(
            images=[mock_workflow_image_data],
            do_run_face_detection=True,
        )

    # Should call select_api_v1 for self-hosted
    mock_client.select_api_v1.assert_called_once()
