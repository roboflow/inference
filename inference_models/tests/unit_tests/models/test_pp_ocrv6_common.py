import numpy as np
import pytest
import torch

from inference_models.errors import ModelInputError
from inference_models.models.pp_ocrv6.pp_ocrv6_common import (
    images_to_numpy_bgr_for_cropping,
    is_torch_input,
    normalize_input_images,
    normalize_torch_images_to_device,
    rescale_image_to_uint8,
)


def test_rescale_image_to_uint8_keeps_uint8_untouched() -> None:
    image = np.full((4, 4, 3), 200, dtype=np.uint8)

    result = rescale_image_to_uint8(image=image)

    assert result is image


def test_rescale_image_to_uint8_treats_floats_as_255_scale() -> None:
    # Float images are assumed to already be on the [0, 255] scale (no [0, 1]
    # auto-detection): a unit-range float is read as near-black, not rescaled.
    image = np.full((4, 4, 3), 1.0, dtype=np.float32)

    result = rescale_image_to_uint8(image=image)

    assert result.dtype == np.uint8
    assert result.max() == 1


def test_rescale_image_to_uint8_rounds_and_casts_255_scale_floats() -> None:
    image = np.full((4, 4, 3), 200.4, dtype=np.float32)

    result = rescale_image_to_uint8(image=image)

    assert result.dtype == np.uint8
    assert result.max() == 200


def test_rescale_image_to_uint8_clips_wide_integers() -> None:
    image = np.full((4, 4, 3), 300, dtype=np.int32)

    result = rescale_image_to_uint8(image=image)

    assert result.dtype == np.uint8
    assert result.max() == 255


def test_normalize_input_images_passes_uint8_through() -> None:
    image = np.random.default_rng(0).integers(0, 256, size=(32, 64, 3), dtype=np.uint8)

    result = normalize_input_images(images=image)

    assert len(result) == 1
    assert result[0].dtype == np.uint8
    # numpy default is bgr; default target is bgr, so no flip
    assert np.array_equal(result[0], image)


def test_normalize_input_images_float_at_255_scale_matches_uint8() -> None:
    image_uint8 = np.random.default_rng(0).integers(
        0, 256, size=(32, 64, 3), dtype=np.uint8
    )

    from_uint8 = normalize_input_images(images=image_uint8)
    from_float = normalize_input_images(images=image_uint8.astype(np.float32))

    assert len(from_uint8) == len(from_float) == 1
    assert np.array_equal(from_uint8[0], from_float[0])


def test_normalize_input_images_applies_target_color_format() -> None:
    image_bgr = np.random.default_rng(3).integers(
        0, 256, size=(8, 8, 3), dtype=np.uint8
    )

    as_rgb = normalize_input_images(images=image_bgr, target_color_format="rgb")[0]

    assert np.array_equal(as_rgb, image_bgr[:, :, ::-1])


def test_normalize_input_images_rejects_torch_tensor() -> None:
    with pytest.raises(ModelInputError):
        normalize_input_images(images=torch.zeros((3, 8, 8)))


def test_normalize_input_images_rejects_list_of_tensors() -> None:
    with pytest.raises(ModelInputError):
        normalize_input_images(images=[torch.zeros((3, 8, 8))])


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


def test_images_to_numpy_bgr_for_cropping_accepts_batched_tensor() -> None:
    images = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)

    result = images_to_numpy_bgr_for_cropping(images=images)

    assert len(result) == 2
    assert all(image.shape == (8, 8, 3) for image in result)


def test_images_to_numpy_bgr_for_cropping_accepts_list_of_tensors() -> None:
    images = [
        torch.zeros((3, 8, 8), dtype=torch.uint8),
        torch.full((3, 16, 16), 255.0, dtype=torch.float32),
    ]

    result = images_to_numpy_bgr_for_cropping(images=images)

    assert len(result) == 2
    assert result[0].shape == (8, 8, 3)
    assert result[1].shape == (16, 16, 3)
    # float 255.0 is read on the [0, 255] scale, not rescaled
    assert result[1].max() == 255


def test_images_to_numpy_bgr_for_cropping_matches_numpy_for_ndarray() -> None:
    image = np.random.default_rng(4).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)

    result = images_to_numpy_bgr_for_cropping(images=image)

    assert np.array_equal(result[0], image)


def test_is_torch_input_discriminates_torch_from_numpy() -> None:
    assert is_torch_input(torch.zeros((3, 4, 4)))
    assert is_torch_input([torch.zeros((3, 4, 4))])
    assert not is_torch_input(np.zeros((4, 4, 3), dtype=np.uint8))
    assert not is_torch_input([np.zeros((4, 4, 3), dtype=np.uint8)])
    assert not is_torch_input([])


def test_normalize_torch_images_to_device_matches_numpy_path() -> None:
    # The device-native torch path must agree with the numpy/cv2 path so
    # predictions do not depend on whether the caller passes np.ndarray or
    # torch.Tensor. Torch images are CHW/rgb and get flipped to bgr; both paths
    # read pixels on the [0, 255] scale.
    image_bgr = np.random.default_rng(2).integers(
        0, 256, size=(20, 30, 3), dtype=np.uint8
    )
    rgb_tensor = (
        torch.from_numpy(np.ascontiguousarray(image_bgr[:, :, ::-1]))
        .permute(2, 0, 1)
        .float()
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
    assert np.array_equal(as_hwc, numpy_ref.astype(int))


def test_normalize_torch_images_to_device_flips_rgb_to_bgr() -> None:
    # A pure-red rgb image on the [0, 255] scale flips to bgr so the red channel
    # lands at index 2 with value 255.
    rgb = torch.zeros((3, 4, 4), dtype=torch.float32)
    rgb[0] = 255.0

    result = normalize_torch_images_to_device(
        images=rgb, input_color_format=None, device=torch.device("cpu")
    )[0]

    assert result.shape == (3, 4, 4)
    assert float(result[0].max()) == 0.0  # blue
    assert float(result[1].max()) == 0.0  # green
    assert float(result[2].max()) == 255.0  # red, moved to bgr index


def test_normalize_torch_images_to_device_target_color_format_no_flip() -> None:
    # With target rgb and rgb input, no net flip: red stays at index 0.
    rgb = torch.zeros((3, 4, 4), dtype=torch.float32)
    rgb[0] = 255.0

    result = normalize_torch_images_to_device(
        images=rgb,
        input_color_format=None,
        device=torch.device("cpu"),
        target_color_format="rgb",
    )[0]

    assert float(result[0].max()) == 255.0  # red kept at index 0
    assert float(result[2].max()) == 0.0


def test_normalize_torch_images_to_device_splits_batched_tensor() -> None:
    images = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)

    result = normalize_torch_images_to_device(
        images=images, input_color_format=None, device=torch.device("cpu")
    )

    assert len(result) == 2
    assert all(image.shape == (3, 8, 8) for image in result)


def test_normalize_torch_images_to_device_reads_floats_on_255_scale() -> None:
    images = [
        torch.zeros((3, 8, 8), dtype=torch.uint8),
        torch.full((3, 4, 4), 255.0, dtype=torch.float32),
    ]

    result = normalize_torch_images_to_device(
        images=images, input_color_format=None, device=torch.device("cpu")
    )

    assert len(result) == 2
    assert result[0].shape == (3, 8, 8)
    assert float(result[1].max()) == 255.0  # float 255.0 read as-is
