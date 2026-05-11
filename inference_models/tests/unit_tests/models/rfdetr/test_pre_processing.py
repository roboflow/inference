import cv2
import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from inference_models.entities import ImageDimensions
from inference_models.models.common.roboflow.model_packages import (
    ColorMode,
    ImagePreProcessing,
    NetworkInputDefinition,
    ResizeMode,
    TrainingInputSize,
)
from inference_models.models.rfdetr.pre_processing import (
    _needs_two_step_resize,
    pre_process_network_input,
)


# ---------------------------------------------------------------------------
# _needs_two_step_resize
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
@pytest.mark.parametrize(
    "dims",
    [
        TrainingInputSize(height=480, width=640),  # non-square
        TrainingInputSize(height=640, width=640),  # square
    ],
)
def test_needs_two_step_resize_true_for_non_stretch_modes(
    resize_mode: ResizeMode, dims: TrainingInputSize
) -> None:
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=560, width=560),
        dataset_version_resize_dimensions=dims,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=resize_mode,
        input_channels=3,
    )
    assert _needs_two_step_resize(network_input) is True


def test_needs_two_step_resize_false_when_dims_is_none() -> None:
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=560, width=560),
        dataset_version_resize_dimensions=None,
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.LETTERBOX,
        input_channels=3,
    )
    assert _needs_two_step_resize(network_input) is False


def test_needs_two_step_resize_false_when_stretch_mode() -> None:
    network_input = NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=560, width=560),
        dataset_version_resize_dimensions=TrainingInputSize(height=480, width=640),
        dynamic_spatial_size_supported=False,
        color_mode=ColorMode.RGB,
        resize_mode=ResizeMode.STRETCH_TO,
        input_channels=3,
    )
    assert _needs_two_step_resize(network_input) is False


# ---------------------------------------------------------------------------
# Tensor-content verification helpers
#
# Both paths share the same final stretch-to-training_input_size + F.to_tensor
# + F.normalize. The reference forward chain replicates that exactly so we can
# diff against pre_process_network_input's tensor output.
# ---------------------------------------------------------------------------


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _reference_pipeline(pil_at_intermediate: Image.Image, target_h: int, target_w: int) -> torch.Tensor:
    """Mirrors `_pre_process_numpy`'s post-resize chain:
    PIL → TF.resize → TF.to_tensor → TF.normalize."""
    resized = TF.resize(pil_at_intermediate, (target_h, target_w))
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD))
    return tensor.unsqueeze(0)


def _build_network_input(
    training_h: int,
    training_w: int,
    resize_mode: ResizeMode,
    dataset_version_dims: TrainingInputSize | None,
    color_mode: ColorMode = ColorMode.RGB,
) -> NetworkInputDefinition:
    return NetworkInputDefinition(
        training_input_size=TrainingInputSize(height=training_h, width=training_w),
        dataset_version_resize_dimensions=dataset_version_dims,
        dynamic_spatial_size_supported=False,
        color_mode=color_mode,
        resize_mode=resize_mode,
        input_channels=3,
        scaling_factor=255,
        normalization=[list(_IMAGENET_MEAN), list(_IMAGENET_STD)],
    )


# ---------------------------------------------------------------------------
# One-step (stretch) path — tensor-content equivalence
# ---------------------------------------------------------------------------


def test_one_step_stretch_tensor_matches_reference_pipeline() -> None:
    image_pre_processing = ImagePreProcessing()
    network_input = _build_network_input(
        training_h=64,
        training_w=64,
        resize_mode=ResizeMode.STRETCH_TO,
        dataset_version_dims=None,
    )
    rng = np.random.default_rng(seed=42)
    image = rng.integers(0, 256, size=(192, 168, 3), dtype=np.uint8)

    actual_tensor, _ = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    expected = _reference_pipeline(Image.fromarray(image), target_h=64, target_w=64)
    torch.testing.assert_close(actual_tensor, expected, atol=1e-6, rtol=0)


def test_one_step_stretch_with_bgr_input_matches_rgb_reference() -> None:
    """When caller passes BGR data with input_color_format='bgr', preprocessor
    swaps to RGB and the resulting tensor matches the RGB reference exactly."""
    image_pre_processing = ImagePreProcessing()
    network_input = _build_network_input(
        training_h=64,
        training_w=64,
        resize_mode=ResizeMode.STRETCH_TO,
        dataset_version_dims=None,
    )
    rng = np.random.default_rng(seed=7)
    rgb_image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    bgr_image = rgb_image[:, :, ::-1].copy()

    actual_tensor, _ = pre_process_network_input(
        images=bgr_image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )

    expected = _reference_pipeline(Image.fromarray(rgb_image), target_h=64, target_w=64)
    torch.testing.assert_close(actual_tensor, expected, atol=1e-6, rtol=0)


def test_one_step_stretch_default_input_color_format_is_treated_as_bgr() -> None:
    """`input_color_format=None` retains the historical loose comparison: when
    the caller doesn't specify and the image arrives in BGR, the swap fires.
    Defensive belt + suspenders alongside the adapter's explicit setting."""
    image_pre_processing = ImagePreProcessing()
    network_input = _build_network_input(
        training_h=64,
        training_w=64,
        resize_mode=ResizeMode.STRETCH_TO,
        dataset_version_dims=None,
    )
    rng = np.random.default_rng(seed=11)
    rgb_image = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    bgr_image = rgb_image[:, :, ::-1].copy()

    none_tensor, _ = pre_process_network_input(
        images=bgr_image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format=None,
    )
    bgr_tensor, _ = pre_process_network_input(
        images=bgr_image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="bgr",
    )
    torch.testing.assert_close(none_tensor, bgr_tensor, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Two-step (non-stretch) path — tensor content + metadata
# ---------------------------------------------------------------------------


def test_two_step_letterbox_tensor_matches_reference() -> None:
    """Replays the two-step chain by hand: cv2 letterbox to non-square dims at
    uint8, then PIL F.resize → F.to_tensor → F.normalize. Output must match
    pre_process_network_input within fp32 noise."""
    image_pre_processing = ImagePreProcessing()
    dataset_version_dims = TrainingInputSize(height=64, width=96)
    network_input = _build_network_input(
        training_h=64,
        training_w=64,
        resize_mode=ResizeMode.LETTERBOX,
        dataset_version_dims=dataset_version_dims,
    )
    network_input = network_input.model_copy(update={"padding_value": 0})
    rng = np.random.default_rng(seed=314)
    image = rng.integers(0, 256, size=(192, 168, 3), dtype=np.uint8)

    actual_tensor, actual_meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )

    # Build the expected intermediate uint8 by hand-running letterbox.
    h, w = image.shape[:2]
    target_h, target_w = dataset_version_dims.height, dataset_version_dims.width
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_top = int((target_h - new_h) / 2)
    pad_left = int((target_w - new_w) / 2)
    scaled = cv2.resize(image, (new_w, new_h))
    intermediate = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    intermediate[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = scaled

    expected = _reference_pipeline(
        Image.fromarray(intermediate), target_h=64, target_w=64
    )
    torch.testing.assert_close(actual_tensor, expected, atol=1e-6, rtol=0)

    meta = actual_meta[0]
    assert meta.original_size == ImageDimensions(height=192, width=168)
    assert meta.inference_size == ImageDimensions(height=64, width=64)


def test_two_step_letterbox_static_crop_offset_propagates() -> None:
    """The shared handler computes a non-zero `static_crop_offset` when static
    crop is configured; verify that it survives through `_pre_process_numpy`'s
    metadata aggregation rather than being silently zeroed."""
    image_pre_processing = ImagePreProcessing.model_validate(
        {
            "static-crop": {
                "enabled": True,
                "x_min": 25,
                "y_min": 10,
                "x_max": 75,
                "y_max": 90,
            }
        }
    )
    dataset_version_dims = TrainingInputSize(height=64, width=96)
    network_input = _build_network_input(
        training_h=64,
        training_w=64,
        resize_mode=ResizeMode.LETTERBOX,
        dataset_version_dims=dataset_version_dims,
    )
    network_input = network_input.model_copy(update={"padding_value": 0})
    image = np.full((100, 100, 3), 64, dtype=np.uint8)

    _, meta = pre_process_network_input(
        images=image,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
    )
    crop = meta[0].static_crop_offset
    assert crop.offset_x == 25
    assert crop.offset_y == 10
    assert crop.crop_width == 50
    assert crop.crop_height == 80


# ---------------------------------------------------------------------------
# Batch / list inputs
# ---------------------------------------------------------------------------


def test_batched_4d_torch_tensor_unbinds_into_per_image_processing() -> None:
    """Caller passes `torch.stack([img, img], dim=0)` → 4D NCHW uint8 tensor.
    The preprocessor must unbind into per-image entries; otherwise
    `Image.fromarray` chokes on a 4D array. Regression test for the integration
    test path that batches via stack rather than passing a list."""
    image_pre_processing = ImagePreProcessing()
    network_input = _build_network_input(
        training_h=64,
        training_w=64,
        resize_mode=ResizeMode.STRETCH_TO,
        dataset_version_dims=None,
    )
    rng = np.random.default_rng(seed=2024)
    img_np = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    chw = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
    batched = torch.stack([chw, chw], dim=0)

    batch_tensor, batch_meta = pre_process_network_input(
        images=batched,
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )
    assert tuple(batch_tensor.shape) == (2, 3, 64, 64)
    assert len(batch_meta) == 2
    expected = _reference_pipeline(Image.fromarray(img_np), target_h=64, target_w=64)
    torch.testing.assert_close(
        batch_tensor[0], expected.squeeze(0), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        batch_tensor[1], expected.squeeze(0), atol=1e-6, rtol=0
    )


def test_batched_input_produces_per_image_metadata() -> None:
    image_pre_processing = ImagePreProcessing()
    network_input = _build_network_input(
        training_h=64,
        training_w=64,
        resize_mode=ResizeMode.STRETCH_TO,
        dataset_version_dims=None,
    )
    rng = np.random.default_rng(seed=99)
    img_a = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    img_b = rng.integers(0, 256, size=(48, 200, 3), dtype=np.uint8)

    batch_tensor, batch_meta = pre_process_network_input(
        images=[img_a, img_b],
        image_pre_processing=image_pre_processing,
        network_input=network_input,
        target_device=torch.device("cpu"),
        input_color_format="rgb",
    )
    assert tuple(batch_tensor.shape) == (2, 3, 64, 64)
    assert len(batch_meta) == 2

    expected_a = _reference_pipeline(Image.fromarray(img_a), target_h=64, target_w=64)
    expected_b = _reference_pipeline(Image.fromarray(img_b), target_h=64, target_w=64)
    torch.testing.assert_close(
        batch_tensor[0], expected_a.squeeze(0), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        batch_tensor[1], expected_b.squeeze(0), atol=1e-6, rtol=0
    )
