"""Unit tests for Depth Estimation block including remote execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.utils.depth_encoding import (
    decode_png_normalized_depth,
    encode_normalized_depth_to_png16,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.depth_estimation.v1 import (
    BlockManifest,
    DepthEstimationBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.depth_estimation.v1_tensor import (
    DepthEstimationBlockV1 as DepthEstimationBlockV1Tensor,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

# The numpy tests below import `v1` explicitly and drive it through mocked
# managers/clients, so they pass in both flag directions and stay unmarked.
# The tensor-native sibling tests target `v1_tensor` and only need to hold
# under ENABLE_TENSOR_DATA_REPRESENTATION.
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
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
    mock_client.depth_estimation.assert_called_once_with(
        inference_input=mock_workflow_image_data.base64_image,
        model_id="depth-anything-v3/small",
        model_id_in_path=True,
        depth_map_format="png16",
    )


@pytest.mark.parametrize("size", ["n", "s", "m", "l", "x"])
def test_manifest_parsing_yolo26_depth_variants(size):
    data = {
        "type": "roboflow_core/depth_estimation@v1",
        "name": "my_depth_step",
        "images": "$inputs.image",
        "model_version": f"yolo26{size}-depth-768",
    }
    result = BlockManifest.model_validate(data)
    assert result.model_version == f"yolo26{size}-depth-768"


def test_supported_model_variants_include_yolo26_depth():
    variants = BlockManifest.get_supported_model_variants()
    # first entry stays the air-gapped cache scanner display name
    assert variants[0] == "depth-anything-v2/small"
    for size in ["n", "s", "m", "l", "x"]:
        assert f"yolo26{size}-depth-768" in variants


def test_run_locally_with_yolo26_depth_variant(
    mock_model_manager, mock_workflow_image_data
):
    block = DepthEstimationBlockV1(
        model_manager=mock_model_manager,
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run(
        images=[mock_workflow_image_data],
        model_version="yolo26n-depth-768",
    )
    assert len(result) == 1
    mock_model_manager.add_model.assert_called_once_with(
        model_id="yolo26n-depth-768", api_key="test_key"
    )
    mock_model_manager.infer_from_request_sync.assert_called_once()


@pytest.fixture
def mock_tensor_native_model_manager():
    """ModelManager mock for the tensor-native path: the block calls
    `run_tensor_native_inference` and receives raw per-image torch depth maps
    (larger == closer - the DepthAnything convention all tensor-native depth
    models / adapters share)."""
    mock = MagicMock(spec=["add_model", "run_tensor_native_inference"])
    mock.run_tensor_native_inference.return_value = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ]
    return mock


@_TENSOR_ONLY
def test_run_locally_tensor_native_normalizes_per_image(
    mock_tensor_native_model_manager, mock_workflow_image_data
):
    block = DepthEstimationBlockV1Tensor(
        model_manager=mock_tensor_native_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_version="depth-anything-v3/small",
    )

    assert len(result) == 1
    mock_tensor_native_model_manager.add_model.assert_called_once_with(
        model_id="depth-anything-v3/small", api_key="test_api_key"
    )
    call = mock_tensor_native_model_manager.run_tensor_native_inference.call_args
    assert call.args == ("depth-anything-v3/small",)
    assert call.kwargs["input_color_format"] == "bgr"
    assert len(call.kwargs["images"]) == 1
    assert isinstance(call.kwargs["images"][0], np.ndarray)
    normalized_depth = result[0]["normalized_depth"]
    assert isinstance(normalized_depth, torch.Tensor)
    # min-max normalization of a larger-means-closer raw map: 1.0 == nearest
    assert torch.allclose(
        normalized_depth.cpu(),
        torch.tensor([[0.0, 1 / 3], [2 / 3, 1.0]]),
        atol=1e-6,
    )
    assert isinstance(result[0]["image"], WorkflowImageData)


@_TENSOR_ONLY
def test_run_locally_tensor_native_uses_materialised_tensor_image(
    mock_tensor_native_model_manager,
):
    tensor_image = torch.zeros((3, 4, 4), dtype=torch.uint8)
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        tensor_image=tensor_image,
    )
    block = DepthEstimationBlockV1Tensor(
        model_manager=mock_tensor_native_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(images=[image], model_version="depth-anything-v3/small")

    assert len(result) == 1
    call = mock_tensor_native_model_manager.run_tensor_native_inference.call_args
    assert call.kwargs["input_color_format"] == "rgb"
    assert isinstance(call.kwargs["images"][0], torch.Tensor)


@_TENSOR_ONLY
def test_run_locally_tensor_native_raises_on_zero_variation_depth_map(
    mock_tensor_native_model_manager, mock_workflow_image_data
):
    # numpy parity: flag-off the same ValueError is raised inside the model's
    # predict() and propagates out of run_locally()
    mock_tensor_native_model_manager.run_tensor_native_inference.return_value = [
        torch.ones((4, 4))
    ]
    block = DepthEstimationBlockV1Tensor(
        model_manager=mock_tensor_native_model_manager,
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    with pytest.raises(ValueError, match="min equals max"):
        block.run(
            images=[mock_workflow_image_data],
            model_version="depth-anything-v3/small",
        )


@_TENSOR_ONLY
def test_run_locally_tensor_native_with_yolo26_depth_variant(
    mock_tensor_native_model_manager, mock_workflow_image_data
):
    """YOLO26 ids run natively through `run_tensor_native_inference`; the
    metric-to-proximity flip lives in InferenceModelsDepthEstimationAdapter,
    so the block-side handling is identical for every model version."""
    block = DepthEstimationBlockV1Tensor(
        model_manager=mock_tensor_native_model_manager,
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    result = block.run(
        images=[mock_workflow_image_data],
        model_version="yolo26n-depth-768",
    )
    assert len(result) == 1
    mock_tensor_native_model_manager.add_model.assert_called_once_with(
        model_id="yolo26n-depth-768", api_key="test_key"
    )
    call = mock_tensor_native_model_manager.run_tensor_native_inference.call_args
    assert call.args == ("yolo26n-depth-768",)


@_TENSOR_ONLY
@patch(
    "inference.core.workflows.core_steps.models.foundation.depth_estimation.v1_tensor.InferenceHTTPClient"
)
def test_run_remotely_tensor_decodes_png16_payload(
    mock_client_cls, mock_workflow_image_data
):
    """The remote path mirrors numpy v1: request png16, receive the ndarray the
    SDK decoded from the base64 PNG16 payload, and (tensor-side) convert it to
    a torch tensor."""
    original_map = np.linspace(0.0, 1.0, num=24, dtype=np.float32).reshape(4, 6)
    decoded_by_sdk = decode_png_normalized_depth(
        encode_normalized_depth_to_png16(original_map)
    )
    mock_client = MagicMock()
    mock_client.depth_estimation.return_value = {
        "normalized_depth": decoded_by_sdk,
        "image": "0000",  # hex-encoded empty image
    }
    mock_client_cls.return_value = mock_client

    block = DepthEstimationBlockV1Tensor(
        model_manager=MagicMock(),
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_version="depth-anything-v3/small",
    )

    assert len(result) == 1
    mock_client.depth_estimation.assert_called_once_with(
        inference_input=mock_workflow_image_data.base64_image,
        model_id="depth-anything-v3/small",
        model_id_in_path=True,
        depth_map_format="png16",
    )
    normalized_depth = result[0]["normalized_depth"]
    assert isinstance(normalized_depth, torch.Tensor)
    assert normalized_depth.dtype == torch.float32
    # png16 quantization step is 1/65535
    assert torch.allclose(
        normalized_depth,
        torch.as_tensor(original_map),
        atol=1.0 / 65535,
    )


@_TENSOR_ONLY
@patch(
    "inference.core.workflows.core_steps.models.foundation.depth_estimation.v1_tensor.InferenceHTTPClient"
)
def test_run_remotely_tensor_handles_legacy_float_list_payload(
    mock_client_cls, mock_workflow_image_data
):
    """Servers that predate `depth_map_format` return the nested float list;
    `np.array` handles both shapes, exactly like numpy v1."""
    mock_client = MagicMock()
    mock_client.depth_estimation.return_value = {
        "normalized_depth": [[0.1, 0.2], [0.3, 0.4]],
        "image": "0000",
    }
    mock_client_cls.return_value = mock_client

    block = DepthEstimationBlockV1Tensor(
        model_manager=MagicMock(),
        api_key="test_api_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_version="depth-anything-v3/small",
    )

    assert len(result) == 1
    normalized_depth = result[0]["normalized_depth"]
    assert isinstance(normalized_depth, torch.Tensor)
    assert torch.allclose(
        normalized_depth,
        torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        atol=1e-6,
    )
