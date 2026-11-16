import base64
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from inference_sdk.http.utils.loaders import (
    load_static_inference_input,
    load_static_inference_input_async,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["I;16", "I;16L", "I;16B"])
@pytest.mark.parametrize("resize", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
async def test_load_unsigned_16_bit_pillow_image_preserves_pixels(
    mode: str, resize: bool, use_async: bool
) -> None:
    values = np.linspace(0, 65535, 40 * 24, dtype=np.uint16).reshape(24, 40)
    dtype = ">u2" if mode == "I;16B" else "<u2"
    image = Image.frombytes(mode, (40, 24), values.astype(dtype).tobytes())
    native_image = Image.fromarray(values)
    max_height, max_width = (12, 20) if resize else (None, None)
    expected_payload, expected_scale = load_static_inference_input(
        native_image, max_height=max_height, max_width=max_width
    )[0]

    if use_async:
        result = await load_static_inference_input_async(
            image, max_height=max_height, max_width=max_width
        )
    else:
        result = load_static_inference_input(
            image, max_height=max_height, max_width=max_width
        )

    payload, scale = result[0]
    recovered_image = Image.open(BytesIO(base64.b64decode(payload)))
    expected_image = Image.open(BytesIO(base64.b64decode(expected_payload)))
    assert recovered_image.size == ((20, 12) if resize else (40, 24))
    assert scale == expected_scale == (0.5 if resize else None)
    np.testing.assert_array_equal(
        np.asarray(recovered_image), np.asarray(expected_image)
    )
    assert image.mode == mode
    np.testing.assert_array_equal(np.asarray(image), values)
