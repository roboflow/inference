import numpy as np
import pytest
import torch

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    PreProcessingMetadata,
    ResizeMode,
    StaticCropOffset,
    TrainingInputSize,
)
from inference_models.models.rfdetr.pre_processing import (
    _needs_nonsquare_two_step_resize,
    pre_process_network_input,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input as base_pre_process_network_input,
)


# ---------------------------------------------------------------------------
# _needs_nonsquare_two_step_resize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "resize_mode",
    [
        ResizeMode.LETTERBOX,
        ResizeMode.CENTER_CROP,
        ResizeMode.FIT_LONGER_EDGE,
        ResizeMode.LETTERBOX_REFLECT_EDGES,
    ],
)
def test_needs_nonsquare_two_step_resize_true_when_dims_nonsquare_and_mode_is_not_stretch(
    resize_mode: ResizeMode,
) -> None:
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=560, width=560),
        dataset_version_resize_dimensions=TrainingInputSize(height=480, width=640),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=resize_mode,
        input_channels=3,
    )
    assert _needs_nonsquare_two_step_resize(network_input) is True


def test_needs_nonsquare_two_step_resize_false_when_dims_is_none() -> None:
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=560, width=560),
        dataset_version_resize_dimensions=None,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    assert _needs_nonsquare_two_step_resize(network_input) is False


def test_needs_nonsquare_two_step_resize_false_when_dims_omitted() -> None:
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=560, width=560),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    assert _needs_nonsquare_two_step_resize(network_input) is False


def test_needs_nonsquare_two_step_resize_false_when_dims_are_square() -> None:
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=560, width=560),
        dataset_version_resize_dimensions=TrainingInputSize(height=640, width=640),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    assert _needs_nonsquare_two_step_resize(network_input) is False


def test_needs_nonsquare_two_step_resize_false_when_stretch_mode() -> None:
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=560, width=560),
        dataset_version_resize_dimensions=TrainingInputSize(height=480, width=640),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    assert _needs_nonsquare_two_step_resize(network_input) is False


# ---------------------------------------------------------------------------
# pre_process_network_input — two-step path
# ---------------------------------------------------------------------------


def test_two_step_resize_produces_square_tensor_from_nonsquare_intermediate() -> None:
    """Letterbox to 96x64 (WxH) intermediate, then bilinear resize to 64x64.

    Input image 192x168 (HxW) → scale = min(96/168, 64/192) = 1/3
    Scaled image: 56x64 (WxH) → pad_left=20, pad_right=20
    Intermediate tensor: (1, 3, 64, 96)
    Final tensor after interpolation: (1, 3, 64, 64)
    """
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    result_tensor, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    assert isinstance(result_tensor, torch.Tensor)
    assert tuple(result_tensor.shape) == (1, 3, 64, 64)

    meta = result_meta[0]
    assert meta.inference_size == ImageDimensions(height=64, width=64)
    assert meta.nonsquare_intermediate_size == ImageDimensions(height=64, width=96)
    assert meta.pad_left == 20
    assert meta.pad_top == 0
    assert meta.pad_right == 20
    assert meta.pad_bottom == 0
    assert meta.original_size == ImageDimensions(height=192, width=168)
    assert meta.size_after_pre_processing == ImageDimensions(height=192, width=168)
    assert np.isclose(meta.scale_width, 1 / 3)
    assert np.isclose(meta.scale_height, 1 / 3)


def test_two_step_resize_with_image_list() -> None:
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    img_a = np.full((192, 168, 3), 50, dtype=np.uint8)
    img_b = np.full((100, 200, 3), 80, dtype=np.uint8)

    result_tensor, result_meta = pre_process_network_input(
        images=[img_a, img_b],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    assert tuple(result_tensor.shape) == (2, 3, 64, 64)
    assert len(result_meta) == 2
    for meta in result_meta:
        assert meta.inference_size == ImageDimensions(height=64, width=64)
        assert meta.nonsquare_intermediate_size == ImageDimensions(height=64, width=96)


def test_two_step_resize_tensor_values_are_in_expected_range() -> None:
    """After bilinear interpolation the pixel values should remain bounded
    by the original content and padding values (0 for black letterbox)."""
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.full((192, 168, 3), 128, dtype=np.uint8)

    result_tensor, _ = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    assert result_tensor.min() >= 0.0
    assert result_tensor.max() <= 128.0


def test_two_step_resize_with_wide_intermediate() -> None:
    """Non-square intermediate wider than tall (landscape orientation)."""
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=128, width=128),
        dataset_version_resize_dimensions=TrainingInputSize(height=96, width=128),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.full((300, 200, 3), 60, dtype=np.uint8)

    result_tensor, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    assert tuple(result_tensor.shape) == (1, 3, 128, 128)
    meta = result_meta[0]
    assert meta.inference_size == ImageDimensions(height=128, width=128)
    assert meta.nonsquare_intermediate_size == ImageDimensions(height=96, width=128)


def test_two_step_resize_with_tall_intermediate() -> None:
    """Non-square intermediate taller than wide (portrait orientation)."""
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=128, width=128),
        dataset_version_resize_dimensions=TrainingInputSize(height=128, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.full((200, 300, 3), 60, dtype=np.uint8)

    result_tensor, result_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    assert tuple(result_tensor.shape) == (1, 3, 128, 128)
    meta = result_meta[0]
    assert meta.inference_size == ImageDimensions(height=128, width=128)
    assert meta.nonsquare_intermediate_size == ImageDimensions(height=128, width=96)


# ---------------------------------------------------------------------------
# pre_process_network_input — pass-through (no two-step)
# ---------------------------------------------------------------------------


def test_passthrough_when_dataset_version_resize_dims_are_none() -> None:
    """Without dataset_version_resize_dimensions the wrapper should produce
    identical results to the base pre_process_network_input."""
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = (np.ones((192, 168, 3), dtype=np.uint8) * (10, 20, 30)).astype(np.uint8)

    wrapper_tensor, wrapper_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )
    base_tensor, base_meta = base_pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    assert torch.equal(wrapper_tensor, base_tensor)
    assert wrapper_meta[0].nonsquare_intermediate_size is None
    assert wrapper_meta[0].inference_size == base_meta[0].inference_size
    assert wrapper_meta[0].pad_left == base_meta[0].pad_left
    assert wrapper_meta[0].pad_top == base_meta[0].pad_top
    assert wrapper_meta[0].pad_right == base_meta[0].pad_right
    assert wrapper_meta[0].pad_bottom == base_meta[0].pad_bottom


def test_passthrough_when_dataset_version_resize_dims_are_square() -> None:
    """Square dataset_version_resize_dimensions should not trigger two-step."""
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=TrainingInputSize(height=96, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    image = np.full((192, 168, 3), 42, dtype=np.uint8)

    wrapper_tensor, wrapper_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    assert tuple(wrapper_tensor.shape) == (1, 3, 64, 64)
    assert wrapper_meta[0].nonsquare_intermediate_size is None
    assert wrapper_meta[0].inference_size == ImageDimensions(height=64, width=64)


def test_passthrough_when_stretch_mode_with_nonsquare_dims() -> None:
    """STRETCH_TO should never trigger two-step, even with non-square dims."""
    image_pre_processing = ImagePreProcessing()
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=64, width=64),
        dataset_version_resize_dimensions=TrainingInputSize(height=64, width=96),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    image = np.full((192, 168, 3), 42, dtype=np.uint8)

    wrapper_tensor, wrapper_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )

    assert tuple(wrapper_tensor.shape) == (1, 3, 64, 64)
    assert wrapper_meta[0].nonsquare_intermediate_size is None
