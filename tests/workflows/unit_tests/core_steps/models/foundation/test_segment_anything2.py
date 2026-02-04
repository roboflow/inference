"""Unit tests for Segment Anything 2 block including remote execution."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything2.v1 import (
    BlockManifest,
    SegmentAnything2BlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def mock_model_manager():
    mock = MagicMock()
    mock_prediction = MagicMock()
    mock_prediction.masks = [[[0, 0], [100, 0], [100, 100], [0, 100]]]
    mock_prediction.confidence = 0.95
    mock.infer_from_request_sync.return_value = MagicMock(
        predictions=[mock_prediction]
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
        "type": "roboflow_core/segment_anything@v1",
        "images": "$inputs.image",
        "version": "hiera_tiny",
    }
    result = BlockManifest.model_validate(data)
    assert result.type == "roboflow_core/segment_anything@v1"
    assert result.version == "hiera_tiny"


@patch(
    "inference.core.workflows.core_steps.models.foundation.segment_anything2.v1.InferenceHTTPClient"
)
def test_run_remotely_calls_sam2_segment_image(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    """Test that remote execution uses the sam2_segment_image client method."""
    mock_client = MagicMock()
    mock_client.sam2_segment_image.return_value = {
        "predictions": [
            {
                "confidence": 0.95,
                "masks": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
            }
        ]
    }
    mock_client_cls.return_value = mock_client

    block = SegmentAnything2BlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        boxes=None,
        version="hiera_tiny",
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    assert "predictions" in result[0]
    mock_client.sam2_segment_image.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.segment_anything2.v1.InferenceHTTPClient"
)
def test_run_remotely_with_prompts(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    """Test that remote execution passes prompts correctly."""
    import supervision as sv
    
    mock_client = MagicMock()
    mock_client.sam2_segment_image.return_value = {
        "predictions": [
            {
                "confidence": 0.95,
                "masks": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
            }
        ]
    }
    mock_client_cls.return_value = mock_client

    # Create mock detections with boxes
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    detections["class_name"] = np.array(["object"])
    detections["detection_id"] = np.array(["det_1"])

    block = SegmentAnything2BlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        boxes=[detections],
        version="hiera_tiny",
        threshold=0.0,
        multimask_output=True,
    )

    assert len(result) == 1
    mock_client.sam2_segment_image.assert_called_once()
    # Verify prompts were passed
    call_args = mock_client.sam2_segment_image.call_args
    assert call_args.kwargs.get("prompts") is not None
