import cv2
import numpy as np
import pytest
from PIL import Image

from inference_sdk.http.utils.pre_processing import (
    determine_scaling_aspect_ratio,
    resize_opencv_image,
    resize_pillow_image,
)


def test_determine_scaling_aspect_ratio_when_downscaling_required() -> None:
    # when
    result = determine_scaling_aspect_ratio(
        image_height=512,
        image_width=512,
        max_height=512,
        max_width=256,
    )

    # then
    assert abs(result - 0.5) < 1e-5


def test_determine_scaling_aspect_ratio_when_up_scaling_required() -> None:
    # when
    result = determine_scaling_aspect_ratio(
        image_height=128,
        image_width=265,
        max_height=512,
        max_width=512,
    )

    # then
    assert result is None


def test_resize_pillow_image_when_resize_is_disabled() -> None:
    # given
    image = Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))

    # when
    result = resize_pillow_image(image=image, max_width=None, max_height=None)

    # then
    assert result[0] is image
    assert result[1] is None


def test_resize_pillow_image_when_resize_is_enabled() -> None:
    # given
    image = Image.new(mode="RGB", size=(128, 128), color=(0, 0, 0))

    # when
    result = resize_pillow_image(image=image, max_width=64, max_height=128)

    # then
    assert result[0].size == (64, 64)
    assert abs(result[1] - 0.5) < 1e-5


def test_resize_opencv_image_when_resize_is_disabled() -> None:
    # given
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # when
    result = resize_opencv_image(image=image, max_width=None, max_height=None)

    # then
    assert result[0] is image
    assert result[1] is None


def test_resize_opencv_image_when_resize_is_enabled() -> None:
    # given
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # when
    result = resize_opencv_image(image=image, max_width=64, max_height=128)

    # then
    assert result[0].shape == (64, 64, 3)
    assert abs(result[1] - 0.5) < 1e-5


@pytest.mark.parametrize("height,width", [(1, 4096), (4096, 1), (1, 1280), (1280, 1)])
def test_resize_opencv_image_preserves_nonzero_dimensions(
    height: int, width: int
) -> None:
    # given
    image = np.full((height, width, 3), 128, dtype=np.uint8)

    # when
    resized_image, scaling_factor = resize_opencv_image(
        image=image, max_height=640, max_width=640
    )

    # then
    expected_shape = (1, 640, 3) if height == 1 else (640, 1, 3)
    assert resized_image.shape == expected_shape
    assert np.all(resized_image == 128)
    assert scaling_factor == pytest.approx(640 / max(height, width))


@pytest.mark.parametrize("height,width", [(1, 4096), (4096, 1), (1, 1280), (1280, 1)])
def test_resize_pillow_image_preserves_nonzero_dimensions(
    height: int, width: int
) -> None:
    # given
    image = Image.new(mode="RGB", size=(width, height), color=(128, 128, 128))

    # when
    resized_image, scaling_factor = resize_pillow_image(
        image=image, max_height=640, max_width=640
    )

    # then
    expected_size = (640, 1) if height == 1 else (1, 640)
    assert resized_image.size == expected_size
    assert np.all(np.asarray(resized_image) == 128)
    assert scaling_factor == pytest.approx(640 / max(height, width))


def test_resize_opencv_image_preserves_fractional_scaling() -> None:
    # given
    image = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)
    scaling_factor = 6 / 11
    expected_image = cv2.resize(
        src=image, dsize=None, fx=scaling_factor, fy=scaling_factor
    )

    # when
    resized_image, actual_scaling_factor = resize_opencv_image(
        image=image, max_height=5, max_width=6
    )

    # then
    np.testing.assert_array_equal(resized_image, expected_image)
    assert actual_scaling_factor == pytest.approx(scaling_factor)


def test_resize_pillow_image_preserves_fractional_scaling() -> None:
    # given
    image = Image.fromarray(np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3))
    expected_image = image.resize(size=(6, 4))

    # when
    resized_image, scaling_factor = resize_pillow_image(
        image=image, max_height=5, max_width=6
    )

    # then
    np.testing.assert_array_equal(np.asarray(resized_image), np.asarray(expected_image))
    assert scaling_factor == pytest.approx(6 / 11)


def test_resize_opencv_image_does_not_upscale() -> None:
    # given
    image = np.zeros((1, 64, 3), dtype=np.uint8)

    # when
    resized_image, scaling_factor = resize_opencv_image(
        image=image, max_height=640, max_width=640
    )

    # then
    assert resized_image is image
    assert scaling_factor is None


def test_resize_pillow_image_does_not_upscale() -> None:
    # given
    image = Image.new(mode="RGB", size=(64, 1))

    # when
    resized_image, scaling_factor = resize_pillow_image(
        image=image, max_height=640, max_width=640
    )

    # then
    assert resized_image is image
    assert scaling_factor is None


@pytest.mark.parametrize("max_dimension", [0, -1])
def test_resize_opencv_image_rejects_nonpositive_bounds(max_dimension: int) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    with pytest.raises(cv2.error):
        resize_opencv_image(
            image=image, max_height=max_dimension, max_width=max_dimension
        )


@pytest.mark.parametrize("max_dimension", [0, -1])
def test_resize_pillow_image_rejects_nonpositive_bounds(max_dimension: int) -> None:
    image = Image.new(mode="RGB", size=(64, 64))

    with pytest.raises(ValueError):
        resize_pillow_image(
            image=image, max_height=max_dimension, max_width=max_dimension
        )
