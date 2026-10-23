import base64

import cv2
import numpy as np
import pytest
from PIL import Image

from inference_sdk.http.utils.loaders import (
    load_static_inference_input,
    load_static_inference_input_async,
)


@pytest.mark.parametrize("height,width", [(1, 4096), (4096, 1)])
@pytest.mark.parametrize("use_pillow", [False, True])
def test_load_static_inference_input_with_thin_image(
    height: int, width: int, use_pillow: bool
) -> None:
    # given
    pixels = np.full((height, width, 3), 128, dtype=np.uint8)
    image = Image.fromarray(pixels) if use_pillow else pixels

    # when
    result = load_static_inference_input(
        inference_input=image, max_height=640, max_width=640
    )

    # then
    assert len(result) == 1
    encoded_image, scaling_factor = result[0]
    decoded_image = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded_image), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    expected_shape = (1, 640, 3) if height == 1 else (640, 1, 3)
    assert decoded_image.shape == expected_shape
    assert np.all(decoded_image == 128)
    assert scaling_factor == pytest.approx(640 / max(height, width))


@pytest.mark.asyncio
@pytest.mark.parametrize("height,width", [(1, 4096), (4096, 1)])
@pytest.mark.parametrize("use_pillow", [False, True])
async def test_load_static_inference_input_async_with_thin_image(
    height: int, width: int, use_pillow: bool
) -> None:
    # given
    pixels = np.full((height, width, 3), 128, dtype=np.uint8)
    image = Image.fromarray(pixels) if use_pillow else pixels

    # when
    result = await load_static_inference_input_async(
        inference_input=image, max_height=640, max_width=640
    )

    # then
    assert len(result) == 1
    encoded_image, scaling_factor = result[0]
    decoded_image = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded_image), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    expected_shape = (1, 640, 3) if height == 1 else (640, 1, 3)
    assert decoded_image.shape == expected_shape
    assert np.all(decoded_image == 128)
    assert scaling_factor == pytest.approx(640 / max(height, width))
