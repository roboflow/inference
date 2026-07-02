import numpy as np
import pytest
import torch

from inference_models.errors import ModelInputError
from inference_models.models.pp_ocrv6.pp_ocrv6_common import (
    is_torch_input,
    normalize_input_images,
    normalize_torch_images_to_device,
    rescale_image_to_uint8,
)


def test_rescale_image_to_uint8_keeps_uint8_untouched() -> None:
    image = np.full((4, 4, 3), 200, dtype=np.uint8)

    result = rescale_image_to_uint8(image=image)

    assert result is image


def test_rescale_image_to_uint8_rescales_unit_range_floats() -> None:
    # Regression: float [0, 1] images (the common normalized torch.Tensor case)
    # must be rescaled to [0, 255] — previously they were interpreted as
    # near-black images and produced garbage predictions with no error.
    image = np.full((4, 4, 3), 1.0, dtype=np.float32)

    result = rescale_image_to_uint8(image=image)

    assert result.dtype == np.uint8
    assert result.max() == 255


def test_rescale_image_to_uint8_keeps_255_scale_floats() -> None:
    image = np.full((4, 4, 3), 200.4, dtype=np.float32)

    result = rescale_image_to_uint8(image=image)

    assert result.dtype == np.uint8
    assert result.max() == 200


def test_rescale_image_to_uint8_clips_wide_integers() -> None:
    image = np.full((4, 4, 3), 300, dtype=np.int32)

    result = rescale_image_to_uint8(image=image)

    assert result.dtype == np.uint8
    assert result.max() == 255


def test_normalize_input_images_float_numpy_matches_uint8() -> None:
    image_uint8 = np.random.default_rng(0).integers(
        0, 256, size=(32, 64, 3), dtype=np.uint8
    )

    from_uint8 = normalize_input_images(images=image_uint8)
    from_float = normalize_input_images(images=image_uint8.astype(np.float32) / 255.0)

    assert len(from_uint8) == len(from_float) == 1
    assert np.abs(from_uint8[0].astype(int) - from_float[0].astype(int)).max() <= 1


def test_normalize_input_images_float_torch_matches_uint8_numpy() -> None:
    image_bgr = np.random.default_rng(1).integers(
        0, 256, size=(32, 64, 3), dtype=np.uint8
    )
    image_rgb_tensor = (
        torch.from_numpy(np.ascontiguousarray(image_bgr[:, :, ::-1]))
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    from_numpy = normalize_input_images(images=image_bgr)
    from_tensor = normalize_input_images(images=image_rgb_tensor)

    assert len(from_tensor) == 1
    assert np.abs(from_numpy[0].astype(int) - from_tensor[0].astype(int)).max() <= 1


def test_normalize_input_images_accepts_batched_tensor() -> None:
    images = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)

    result = normalize_input_images(images=images)

    assert len(result) == 2
    assert all(image.shape == (8, 8, 3) for image in result)


def test_normalize_input_images_accepts_list_of_tensors() -> None:
    images = [
        torch.zeros((3, 8, 8), dtype=torch.uint8),
        torch.ones((3, 16, 16), dtype=torch.float32),
    ]

    result = normalize_input_images(images=images)

    assert len(result) == 2
    assert result[0].shape == (8, 8, 3)
    assert result[1].shape == (16, 16, 3)
    assert result[1].max() == 255


def test_normalize_input_images_converts_grayscale() -> None:
    image = np.full((8, 8), 128, dtype=np.uint8)

    result = normalize_input_images(images=image)

    assert result[0].shape == (8, 8, 3)


def test_normalize_input_images_rejects_empty_list() -> None:
    with pytest.raises(ModelInputError):
        normalize_input_images(images=[])


def test_normalize_input_images_rejects_unsupported_type() -> None:
    with pytest.raises(ModelInputError):
        normalize_input_images(images="not-an-image")


def test_normalize_input_images_rejects_wrong_channel_count() -> None:
    with pytest.raises(ModelInputError):
        normalize_input_images(images=np.zeros((8, 8, 4), dtype=np.uint8))


def test_is_torch_input_discriminates_torch_from_numpy() -> None:
    assert is_torch_input(torch.zeros((3, 4, 4)))
    assert is_torch_input([torch.zeros((3, 4, 4))])
    assert not is_torch_input(np.zeros((4, 4, 3), dtype=np.uint8))
    assert not is_torch_input([np.zeros((4, 4, 3), dtype=np.uint8)])
    assert not is_torch_input([])


def test_normalize_torch_images_to_device_matches_numpy_path() -> None:
    # The device-native torch path must agree with the numpy/cv2 path so
    # predictions do not depend on whether the caller passes np.ndarray or
    # torch.Tensor. Torch images are CHW/rgb and get flipped to bgr.
    image_bgr = np.random.default_rng(2).integers(
        0, 256, size=(20, 30, 3), dtype=np.uint8
    )
    rgb_tensor = (
        torch.from_numpy(np.ascontiguousarray(image_bgr[:, :, ::-1]))
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    numpy_ref = normalize_input_images(images=image_bgr)[0]  # HWC uint8 BGR
    torch_out = normalize_torch_images_to_device(
        images=rgb_tensor, input_color_format=None, device=torch.device("cpu")
    )

    assert len(torch_out) == 1
    result = torch_out[0]
    assert result.shape == (3, 20, 30)  # CHW, kept as a tensor (no numpy round-trip)
    assert result.dtype == torch.float32
    as_hwc = result.permute(1, 2, 0).round().to(torch.int32).numpy()
    assert np.abs(as_hwc - numpy_ref.astype(int)).max() <= 1


def test_normalize_torch_images_to_device_rescales_unit_range_and_flips_to_bgr() -> None:
    # A pure-red rgb float image in [0, 1]: rescaled ×255 and flipped to bgr so
    # the red channel lands at index 2 with value 255.
    rgb = torch.zeros((3, 4, 4), dtype=torch.float32)
    rgb[0] = 1.0

    result = normalize_torch_images_to_device(
        images=rgb, input_color_format=None, device=torch.device("cpu")
    )[0]

    assert result.shape == (3, 4, 4)
    assert float(result[0].max()) == 0.0  # blue
    assert float(result[1].max()) == 0.0  # green
    assert float(result[2].max()) == 255.0  # red, rescaled and moved to bgr index


def test_normalize_torch_images_to_device_splits_batched_tensor() -> None:
    images = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)

    result = normalize_torch_images_to_device(
        images=images, input_color_format=None, device=torch.device("cpu")
    )

    assert len(result) == 2
    assert all(image.shape == (3, 8, 8) for image in result)


def test_normalize_torch_images_to_device_accepts_list_and_rescales_floats() -> None:
    images = [
        torch.zeros((3, 8, 8), dtype=torch.uint8),
        torch.ones((3, 4, 4), dtype=torch.float32),
    ]

    result = normalize_torch_images_to_device(
        images=images, input_color_format=None, device=torch.device("cpu")
    )

    assert len(result) == 2
    assert result[0].shape == (3, 8, 8)
    assert float(result[1].max()) == 255.0  # float 1.0 -> 255
