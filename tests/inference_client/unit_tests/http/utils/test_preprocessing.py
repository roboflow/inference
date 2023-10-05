import numpy as np
from PIL import Image

from inference_client.http.utils.pre_processing import (
    determine_scaling_aspect_ratio,
    resize_pillow_image,
    resize_opencv_image,
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
