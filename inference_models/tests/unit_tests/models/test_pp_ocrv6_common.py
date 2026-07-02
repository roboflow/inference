import numpy as np
import pytest
import torch

from inference_models.errors import ModelInputError
from inference_models.models.pp_ocrv6.pp_ocrv6_common import (
    normalize_input_images,
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
