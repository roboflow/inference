"""Unit tests for Segment Anything 3 3D block including remote execution."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1 import (
    BlockManifest,
    SegmentAnything3_3D_ObjectsBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def mock_model_manager():
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.mesh_glb = b"mock_mesh_data"
    mock_response.gaussian_ply = b"mock_gaussian_data"
    mock_response.time = 1.5
    mock_obj = MagicMock()
    mock_obj.mesh_glb = b"obj_mesh"
    mock_obj.gaussian_ply = b"obj_gaussian"
    mock_obj.metadata.rotation = [0, 0, 0, 1]
    mock_obj.metadata.translation = [0, 0, 0]
    mock_obj.metadata.scale = [1, 1, 1]
    mock_response.objects = [mock_obj]
    mock.infer_from_request_sync.return_value = mock_response
    return mock


@pytest.fixture
def mock_workflow_image_data():
    start_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=start_image,
    )


@pytest.fixture
def mock_mask_input():
    # Polygon format mask
    return [[10, 10, 100, 10, 100, 100, 10, 100]]


def test_manifest_parsing_valid():
    data = {
        "type": "roboflow_core/segment_anything3_3d_objects@v1",
        "images": "$inputs.image",
        "mask_input": "$steps.sam2.predictions",
    }
    result = BlockManifest.model_validate(data)
    assert result.type == "roboflow_core/segment_anything3_3d_objects@v1"


def test_run_locally(mock_model_manager, mock_workflow_image_data, mock_mask_input):
    block = SegmentAnything3_3D_ObjectsBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        mask_input=[mock_mask_input],
    )

    assert len(result) == 1
    assert "mesh_glb" in result[0]
    assert "gaussian_ply" in result[0]
    assert "objects" in result[0]
    assert "inference_time" in result[0]
    mock_model_manager.add_model.assert_called_once()
    mock_model_manager.infer_from_request_sync.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1.InferenceHTTPClient"
)
def test_run_remotely_calls_sam3_3d_infer(
    mock_client_cls, mock_model_manager, mock_workflow_image_data, mock_mask_input
):
    """Test that remote execution uses the sam3_3d_infer client method."""
    mock_client = MagicMock()
    mock_client.sam3_3d_infer.return_value = {
        "mesh_glb": "base64_mesh_data",
        "gaussian_ply": "base64_gaussian_data",
        "objects": [
            {
                "mesh_glb": "obj_mesh_b64",
                "gaussian_ply": "obj_gaussian_b64",
                "metadata": {
                    "rotation": [0, 0, 0, 1],
                    "translation": [0, 0, 0],
                    "scale": [1, 1, 1],
                },
            }
        ],
        "time": 1.5,
    }
    mock_client_cls.return_value = mock_client

    block = SegmentAnything3_3D_ObjectsBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        mask_input=[mock_mask_input],
    )

    assert len(result) == 1
    assert "mesh_glb" in result[0]
    assert "gaussian_ply" in result[0]
    assert "objects" in result[0]
    assert "inference_time" in result[0]
    mock_client.sam3_3d_infer.assert_called_once()


@patch(
    "inference.core.workflows.core_steps.models.foundation.segment_anything3_3d.v1.InferenceHTTPClient"
)
def test_run_remotely_converts_numpy_masks_to_lists(
    mock_client_cls, mock_model_manager, mock_workflow_image_data
):
    """Test that numpy arrays in mask_input are converted to lists for JSON serialization."""
    import supervision as sv

    mock_client = MagicMock()
    mock_client.sam3_3d_infer.return_value = {
        "mesh_glb": "base64_mesh_data",
        "gaussian_ply": "base64_gaussian_data",
        "objects": [],
        "time": 1.0,
    }
    mock_client_cls.return_value = mock_client

    # Create mock detections with numpy masks
    detections = sv.Detections(
        xyxy=np.array([[10, 10, 50, 50]]),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        mask=np.array([np.ones((480, 640), dtype=bool)]),
    )

    block = SegmentAnything3_3D_ObjectsBlockV1(
        model_manager=mock_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        mask_input=[detections],
    )

    assert len(result) == 1
    mock_client.sam3_3d_infer.assert_called_once()
    # Verify mask_input was converted
    call_args = mock_client.sam3_3d_infer.call_args
    mask_input = call_args.kwargs.get("mask_input")
    assert isinstance(mask_input, list)
