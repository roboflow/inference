import numpy as np
import pytest
from PIL import Image
from unittest.mock import MagicMock, patch
from pydantic import ValidationError


from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.core_steps.models.foundation.hugging_face.depth_anything2.v1 import (
    BlockManifest,
    DepthAnythingV2BlockV1,
    process_depth_map,
    create_visualization,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

@pytest.mark.parametrize(
    "type_alias", ["roboflow_core/depth_anything_v2@v1"]
)
def test_depth_anything_step_validation_when_input_is_valid(type_alias: str) -> None:
    # given
    specification = {
        "type": type_alias,
        "name": "step_1",
        "image": "$inputs.image",
        "model_size": "Small",
        "colormap": "Spectral_r",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result == BlockManifest(
        type=type_alias,
        name="step_1",
        image="$inputs.image",
        model_size="Small",
        colormap="Spectral_r",
    )


@pytest.mark.parametrize("value", ["Invalid", None, 1, True])
def test_depth_anything_step_validation_when_model_size_invalid(value: str) -> None:
    # given
    specification = {
        "type": "DepthAnythingV2",
        "name": "step_1",
        "image": "$inputs.image",
        "model_size": value,
        "colormap": "Spectral_r",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", ["Invalid", None, 1, True])
def test_depth_anything_step_validation_when_colormap_invalid(value: str) -> None:
    # given
    specification = {
        "type": "DepthAnythingV2",
        "name": "step_1",
        "image": "$inputs.image",
        "model_size": "Small",
        "colormap": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_process_depth_map_when_valid():
    # given
    depth_array = np.array([[1, 2], [3, 4]], dtype=np.float32)

    # when
    result = process_depth_map(depth_array)

    # then
    assert np.array_equal(result, depth_array)


def test_process_depth_map_when_invalid():
    # given
    depth_array = np.ones((2, 2), dtype=np.float32)

    # when/then
    with pytest.raises(ValueError, match="Depth map has no variation"):
        process_depth_map(depth_array)


def test_create_visualization():
    # given
    depth_array = np.array([[0, 1], [2, 3]], dtype=np.float32)

    # when
    result = create_visualization(depth_array, "Spectral_r")

    # then
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8


@patch("transformers.pipeline")
def test_depth_anything_block_run(mock_pipeline):
    # given
    mock_depth_output = {"depth": np.ones((10, 10), dtype=np.float32)}
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.return_value = mock_depth_output
    mock_pipeline.return_value = mock_pipeline_instance

    block = DepthAnythingV2BlockV1()
    input_image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="some"),
        numpy_image=np.zeros((10, 10, 3), dtype=np.uint8),
    )

    # when
    result = block.run(
        image=input_image,
        model_size="Small",
        colormap="Spectral_r",
    )

    # then
    assert "image" in result
    assert "normalized_depth" in result
    assert isinstance(result["image"], WorkflowImageData)
    assert isinstance(result["normalized_depth"], np.ndarray)
    assert result["normalized_depth"].shape == (10, 10)
    assert result["image"].numpy_image.shape == (10, 10, 3)

