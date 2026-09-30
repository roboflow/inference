import base64
import io

import numpy as np
import pytest
from fastapi import Request
from PIL import Image

from inference_server.legacy.common import (
    as_image_list,
    decode_inline_image,
    image_dims,
    load_request_images,
    resolve_api_key,
)
from inference_server.legacy.errors import LegacyHTTPError


def _jpeg(w=7, h=5) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="JPEG")
    return buf.getvalue()


def _request(headers=None):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "headers": headers or [],
    }
    return Request(scope)


def test_image_dims_reads_jpeg_header():
    assert image_dims(_jpeg(7, 5)) == (7, 5)


@pytest.mark.parametrize(
    "orientation,expected", [(1, (7, 5)), (3, (7, 5)), (6, (5, 7)), (8, (5, 7))]
)
def test_image_dims_reports_exif_oriented_size(orientation, expected):
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    Image.new("RGB", (7, 5)).save(buf, format="JPEG", exif=exif)
    assert image_dims(buf.getvalue()) == expected


@pytest.mark.parametrize("orientation", [3, 6, 8])
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "TIFF", "WEBP"])
def test_image_dims_match_decoded_pixels_per_format(fmt, orientation):
    """Reported size follows cv2.imdecode: JPEG/PNG/TIFF oriented, WebP not."""
    from inference_model_manager.backends.decode import make_decoder

    exif = Image.Exif()
    exif[0x0112] = orientation
    options = {"WEBP": {"lossless": True}}.get(fmt, {})
    buf = io.BytesIO()
    Image.new("RGB", (7, 5)).save(buf, fmt, exif=exif.tobytes(), **options)
    data = buf.getvalue()
    with Image.open(io.BytesIO(data)) as image:
        assert image.getexif().get(0x0112) == orientation
    decoded = make_decoder("imagecodecs")(data)
    assert image_dims(data) == (decoded.shape[1], decoded.shape[0])
    transposed = fmt != "WEBP" and orientation in (6, 8)
    assert image_dims(data) == ((5, 7) if transposed else (7, 5))


def test_image_dims_reads_npy_header():
    buf = io.BytesIO()
    np.save(buf, np.zeros((5, 7, 3), dtype=np.uint8), allow_pickle=False)
    assert image_dims(buf.getvalue()) == (7, 5)


def test_decode_inline_image_base64_with_data_url_prefix():
    payload = "data:image/jpeg;base64," + base64.b64encode(_jpeg()).decode()
    result = decode_inline_image({"type": "base64", "value": payload}, ndarray_ok=False)
    assert isinstance(result.data, bytes) and (result.width, result.height) == (7, 5)


def test_decode_inline_image_numpy_object_kept_as_ndarray_when_allowed():
    arr = np.zeros((5, 7, 3), dtype=np.uint8)
    result = decode_inline_image(
        {"type": "numpy_object", "value": arr}, ndarray_ok=True
    )
    assert result.data is arr and (result.width, result.height) == (7, 5)


def test_decode_inline_image_numpy_object_becomes_npy_bytes_when_not_allowed():
    arr = np.zeros((5, 7, 3), dtype=np.uint8)
    result = decode_inline_image(
        {"type": "numpy_object", "value": arr}, ndarray_ok=False
    )
    assert result.data[:6] == b"\x93NUMPY"


def test_decode_inline_image_pickled_numpy_refused():
    with pytest.raises(LegacyHTTPError) as exc:
        decode_inline_image({"type": "numpy", "value": b"x"}, ndarray_ok=False)
    assert exc.value.status_code == 501


@pytest.mark.asyncio
async def test_load_request_images_fetches_urls_in_one_batch(monkeypatch):
    seen = []

    async def _fetch(urls):
        seen.append(urls)
        return [_jpeg(3, 2) for _ in urls], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)
    out = await load_request_images(
        [
            {"type": "url", "value": "https://a/1.jpg"},
            {"type": "base64", "value": base64.b64encode(_jpeg()).decode()},
            {"type": "url", "value": "https://a/2.jpg"},
        ],
        ndarray_ok=False,
    )
    assert seen == [["https://a/1.jpg", "https://a/2.jpg"]]
    assert [(p.width, p.height) for p in out] == [(3, 2), (7, 5), (3, 2)]


@pytest.mark.asyncio
async def test_load_request_images_enforces_count_limit(monkeypatch):
    monkeypatch.setattr(
        "inference_server.framework.input_parsers.image_limits.configuration.MAX_IMAGES_PER_REQUEST",
        1,
    )
    with pytest.raises(LegacyHTTPError) as exc:
        await load_request_images(
            [{"type": "base64", "value": "x"}] * 2, ndarray_ok=False
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_load_request_images_refuses_urls_offline(monkeypatch):
    monkeypatch.setattr("inference_server.legacy.common.OFFLINE_MODE", True)
    with pytest.raises(LegacyHTTPError) as exc:
        await load_request_images(
            [{"type": "url", "value": "https://a/1.jpg"}], ndarray_ok=False
        )
    assert exc.value.status_code == 400


def test_resolve_api_key_precedence(monkeypatch):
    monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", None)
    req = _request(headers=[(b"authorization", b"Bearer H")])
    assert resolve_api_key(req, "Q", "B") == "Q"
    assert resolve_api_key(req, None, "B") == "H"
    assert resolve_api_key(_request(), None, "B") == "B"
    assert resolve_api_key(_request(), None, None) is None
    monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", "ENV")
    assert resolve_api_key(_request(), None, None) == "ENV"


def test_as_image_list():
    assert as_image_list({"type": "base64", "value": "x"}) == (
        [{"type": "base64", "value": "x"}],
        False,
    )
    assert as_image_list([1, 2]) == ([1, 2], True)
