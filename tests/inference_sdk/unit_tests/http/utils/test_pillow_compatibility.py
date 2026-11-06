import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
from PIL import Image

from inference_sdk.http.utils.encoding import bytes_to_pillow_image
from inference_sdk.http.utils.loaders import (
    load_static_inference_input,
    load_static_inference_input_async,
)


def _make_image(mode: str) -> Image.Image:
    if mode == "I;16":
        image = Image.fromarray(np.full((48, 80), 40000, dtype=np.uint16))
    elif mode == "P":
        image = Image.new("P", size=(80, 48), color=1)
        image.putpalette([0, 0, 0, 64, 128, 192] + [0] * (768 - 6))
    else:
        colors = {"RGB": (64, 128, 192), "RGBA": (64, 128, 192, 128), "LA": (96, 128)}
        image = Image.new(mode, size=(80, 48), color=colors[mode])

    return image


def _assert_encoded_image(
    result: List[Tuple[str, Optional[float]]],
    *,
    size: Tuple[int, int],
    mode: str,
    pixel: Tuple[int, ...],
    scale: Optional[float],
) -> None:
    assert len(result) == 1
    payload, scaling_factor = result[0]
    assert scaling_factor == scale
    with bytes_to_pillow_image(base64.b64decode(payload)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.mode == mode
        assert decoded.size == size
        pixels = np.asarray(decoded)
        expected = pixel if mode == "RGB" else pixel[0]
        # Uniform colors have a small JPEG rounding error, not a channel swap.
        np.testing.assert_allclose(
            pixels, np.broadcast_to(expected, pixels.shape), atol=3
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("resize", [False, True])
@pytest.mark.parametrize(
    "mode,expected_mode,pixel",
    [
        ("RGB", "RGB", (64, 128, 192)),
        ("RGBA", "RGB", (64, 128, 192)),
        ("LA", "RGB", (96, 96, 96)),
        ("P", "RGB", (64, 128, 192)),
        ("I;16", "L", (156,)),
    ],
)
async def test_pillow_input_preserves_pixels_through_resize_and_jpeg_encoding(
    mode: str,
    expected_mode: str,
    pixel: Tuple[int, ...],
    resize: bool,
    use_async: bool,
) -> None:
    with _make_image(mode) as image:
        original_pixels = image.tobytes()
        kwargs = {
            "inference_input": image,
            "max_height": 20 if resize else None,
            "max_width": 20 if resize else None,
        }
        if use_async:
            result = await load_static_inference_input_async(**kwargs)
        else:
            result = load_static_inference_input(**kwargs)

        _assert_encoded_image(
            result,
            size=(20, 12) if resize else (80, 48),
            mode=expected_mode,
            pixel=pixel,
            scale=0.25 if resize else None,
        )
        assert image.tobytes() == original_pixels


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("mode", ["RGBA", "LA"])
async def test_fully_transparent_resize_retains_pillow_premultiplied_alpha_behavior(
    mode: str, use_async: bool
) -> None:
    color = (64, 128, 192, 0) if mode == "RGBA" else (96, 0)
    with Image.new(mode, size=(80, 48), color=color) as image:
        if use_async:
            result = await load_static_inference_input_async(
                image, max_width=20, max_height=20
            )
        else:
            result = load_static_inference_input(image, max_width=20, max_height=20)

    # Pillow's resize discards RGB hidden under zero alpha before JPEG conversion.
    _assert_encoded_image(
        result, size=(20, 12), mode="RGB", pixel=(0, 0, 0), scale=0.25
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("input_form", ["numpy", "file", "base64"])
async def test_image_references_preserve_color_when_resized(
    input_form: str, use_async: bool, tmp_path: Path
) -> None:
    with Image.new("RGB", size=(80, 48), color=(64, 128, 192)) as image:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()

    if input_form == "numpy":
        reference = np.full((48, 80, 3), (192, 128, 64), dtype=np.uint8)
    elif input_form == "file":
        file_path = tmp_path / "image.png"
        file_path.write_bytes(payload)
        reference = str(file_path)
    else:
        reference = base64.b64encode(payload).decode("ascii")

    if use_async:
        result = await load_static_inference_input_async(
            reference, max_width=20, max_height=20
        )
    else:
        result = load_static_inference_input(reference, max_width=20, max_height=20)

    _assert_encoded_image(
        result, size=(20, 12), mode="RGB", pixel=(64, 128, 192), scale=0.25
    )
