import base64
import re
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pytest
from aioresponses import CallbackResult, aioresponses
from PIL import Image
from requests_mock.mocker import Mocker

from inference_sdk.http.client import InferenceHTTPClient
from inference_sdk.http.entities import InferenceConfiguration
from inference_sdk.http.utils.pre_processing import resize_pillow_image


def _palette_image(*, multicolor: bool, alpha: str) -> Tuple[Image.Image, Image.Image]:
    colors = [(64, 128, 192), (180, 80, 40), (40, 160, 90), (220, 190, 60)]
    indices = [1, 203, 7, 31]
    palette = [0] * 768
    labels = np.zeros((48, 80), dtype=np.uint8)
    rgb = np.zeros((48, 80, 3), dtype=np.uint8)
    for column in range(4):
        choice = column if multicolor else 0
        start, end = column * 20, (column + 1) * 20
        labels[:, start:end] = indices[choice]
        rgb[:, start:end] = colors[choice]
        palette[3 * indices[choice] : 3 * indices[choice] + 3] = colors[choice]

    opacity = np.full((48, 80), 255, dtype=np.uint8)
    if alpha == "mixed":
        opacity[:, :20] = 0
        opacity[:, 20:40] = 64
        opacity[:, 40:60] = 128
    elif alpha == "zero":
        opacity[:] = 0

    image = Image.frombytes("P", (80, 48), labels.tobytes())
    image.putpalette(palette)
    image = image.convert("PA")
    image.putalpha(Image.fromarray(opacity))
    reference = Image.fromarray(np.dstack((rgb, opacity)))
    np.testing.assert_array_equal(np.asarray(image.convert("RGBA")), reference)

    return image, reference


@pytest.mark.parametrize("multicolor", [False, True])
@pytest.mark.parametrize("alpha", ["opaque", "mixed", "zero"])
def test_resize_pa_image_preserves_palette_colors_and_alpha(
    multicolor: bool, alpha: str
) -> None:
    image, reference = _palette_image(multicolor=multicolor, alpha=alpha)
    original_pixels = image.tobytes()
    original_palette = image.palette.tobytes()

    resized, scale = resize_pillow_image(image=image, max_width=20, max_height=20)

    expected = reference.resize((20, 12))
    np.testing.assert_array_equal(np.asarray(resized.convert("RGBA")), expected)
    assert resized.size == (20, 12)
    assert scale == pytest.approx(0.25)
    assert image.mode == "PA"
    assert image.tobytes() == original_pixels
    assert image.palette.tobytes() == original_palette


@pytest.mark.parametrize(
    "max_width,max_height", [(None, None), (None, 20), (20, None), (80, 48), (160, 96)]
)
def test_pa_image_without_downsizing_keeps_original_object(
    max_width: Optional[int], max_height: Optional[int]
) -> None:
    image, _ = _palette_image(multicolor=True, alpha="mixed")

    resized, scale = resize_pillow_image(
        image=image, max_width=max_width, max_height=max_height
    )

    assert resized is image
    assert scale is None


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "P", "L", "LA", "I;16"])
def test_other_pillow_modes_keep_existing_resampling(mode: str) -> None:
    _, reference = _palette_image(multicolor=True, alpha="mixed")
    if mode == "I;16":
        image = Image.fromarray(np.arange(48 * 80, dtype=np.uint16).reshape(48, 80))
    else:
        image = reference.convert(mode)

    expected = image.resize((20, 12))
    resized, scale = resize_pillow_image(image=image, max_width=20, max_height=20)

    assert resized.mode == expected.mode
    assert scale == pytest.approx(0.25)
    np.testing.assert_array_equal(np.asarray(resized), expected)
    np.testing.assert_array_equal(
        np.asarray(resized.convert("RGB")), expected.convert("RGB")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("downsize", [False, True])
@pytest.mark.parametrize("alpha", ["opaque", "mixed"])
async def test_public_sdk_request_preserves_pa_colors_and_prediction_geometry(
    requests_mock: Mocker, use_async: bool, downsize: bool, alpha: str
) -> None:
    image, reference = _palette_image(multicolor=True, alpha=alpha)
    expected_size = (20, 12) if downsize else image.size
    expected = reference.resize(expected_size) if downsize else reference
    with BytesIO() as buffer:
        expected.convert("RGB").save(buffer, format="JPEG")
        with Image.open(BytesIO(buffer.getvalue())) as decoded:
            expected_pixels = np.asarray(decoded).copy()

    client = InferenceHTTPClient(api_url="http://palette.test", api_key="test-key")
    if downsize:
        configuration = InferenceConfiguration(
            api_key_transport="legacy",
            client_downsizing_disabled=False,
            default_max_input_size=20,
        )
    else:
        configuration = InferenceConfiguration(api_key_transport="legacy")

    client.configure(inference_configuration=configuration)
    captured = []

    def _response(payload: bytes) -> dict:
        with Image.open(BytesIO(base64.b64decode(payload))) as decoded:
            assert decoded.format == "JPEG"
            assert decoded.size == expected_size
            np.testing.assert_array_equal(np.asarray(decoded), expected_pixels)
            width, height = decoded.size

        captured.append(payload)
        result = {
            "image": {"width": width, "height": height},
            "predictions": [
                {
                    "x": width / 2,
                    "y": height / 2,
                    "width": width,
                    "height": height,
                    "confidence": 0.9,
                    "class": "object",
                }
            ],
        }

        return result

    if use_async:

        def _async_response(url, **kwargs) -> CallbackResult:
            result = CallbackResult(payload=_response(kwargs["data"]))

            return result

        with aioresponses() as mocked:
            mocked.post(
                re.compile(r"^http://palette\.test/demo/1(?:\?.*)?$"),
                callback=_async_response,
            )
            result = await client.infer_from_api_v0_async(image, model_id="demo/1")
    else:

        def _sync_response(request, context) -> dict:
            result = _response(request.body)

            return result

        requests_mock.post("http://palette.test/demo/1", json=_sync_response)
        result = client.infer_from_api_v0(image, model_id="demo/1")

    assert len(captured) == 1
    assert result["image"] == {"width": 80, "height": 48}
    assert result["predictions"] == [
        {
            "x": 40,
            "y": 24,
            "width": 80,
            "height": 48,
            "confidence": 0.9,
            "class": "object",
        }
    ]
