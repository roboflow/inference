import numpy as np
import pytest

from inference.core.utils.preprocess import (
    downscale_image_keeping_aspect_ratio as server_downscale,
)
from inference.core.workflows.utils.images import (
    _resize_image_keeping_aspect_ratio,
    downscale_image_keeping_aspect_ratio,
)


def test_downscale_matches_the_server_implementation() -> None:
    image = np.arange(200 * 100 * 3, dtype=np.uint8).reshape(100, 200, 3)
    ours, theirs = (
        fn(image, (50, 50))
        for fn in (downscale_image_keeping_aspect_ratio, server_downscale)
    )
    assert ours.shape == theirs.shape and np.array_equal(ours, theirs)
    small = np.zeros((10, 10, 3), dtype=np.uint8)
    assert downscale_image_keeping_aspect_ratio(small, (50, 50)) is small


def test_downscale_matches_the_server_implementation_for_portrait_images() -> None:
    image = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(200, 100, 3)
    ours, theirs = (
        fn(image, (50, 50))
        for fn in (downscale_image_keeping_aspect_ratio, server_downscale)
    )
    assert ours.shape == theirs.shape and np.array_equal(ours, theirs)


def test_resize_image_keeping_aspect_ratio_rejects_non_ndarray_input() -> None:
    with pytest.raises(
        ValueError,
        match="Received an image of unknown type, <class 'list'>",
    ):
        _resize_image_keeping_aspect_ratio(image=[1, 2, 3], desired_size=(50, 50))
