"""Unit tests for v2 infer input extraction (multipart, JSON+base64, raw, batch)."""

from __future__ import annotations

import base64

import pytest

# Minimal JPEG (smallest valid JPEG — JFIF header + EOI)
_JPEG = bytes(
    [
        0xFF,
        0xD8,
        0xFF,
        0xE0,
        0x00,
        0x10,
        0x4A,
        0x46,
        0x49,
        0x46,
        0x00,
        0x01,
        0x01,
        0x00,
        0x00,
        0x01,
        0x00,
        0x01,
        0x00,
        0x00,
        0xFF,
        0xD9,
    ]
)


class _FakeUploadFile:
    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    def __init__(
        self, content_type: str = "", body: bytes = b"", form_data=None, json_body=None
    ):
        self.headers = {"content-type": content_type}
        self._body = body
        self._form_data = form_data
        self._json_body = json_body

    async def form(self):
        return self._form_data or _FakeFormData({})

    async def json(self):
        if self._json_body is not None:
            return self._json_body
        raise ValueError("no json")

    async def stream(self):
        if self._body:
            yield self._body


class _FakeFormData(dict):
    def multi_items(self):
        for k, v in self.items():
            if isinstance(v, list):
                for item in v:
                    yield k, item
            else:
                yield k, v


# ---------------------------------------------------------------------------
# Single-image tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_body():
    from inference_server.framework.input_parsers import extract_images_and_params

    req = _FakeRequest(content_type="image/jpeg", body=_JPEG)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params == {}


@pytest.mark.asyncio
async def test_multipart_form_single():
    from inference_server.framework.input_parsers import extract_images_and_params

    form = _FakeFormData({"image": _FakeUploadFile(_JPEG), "confidence": "0.5"})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params["confidence"] == "0.5"


@pytest.mark.asyncio
async def test_multipart_form_with_inputs_json():
    from inference_server.framework.input_parsers import extract_images_and_params

    form = _FakeFormData(
        {
            "image": _FakeUploadFile(_JPEG),
            "inputs": '{"confidence": 0.3, "iou": 0.5}',
        }
    )
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert params["confidence"] == 0.3
    assert params["iou"] == 0.5


@pytest.mark.asyncio
async def test_multipart_form_missing_image_passes_through_params():
    # Zero image parts is legal at the extractor layer (params-only requests);
    # each handler-family parser enforces its own image requirement.
    from inference_server.framework.input_parsers import extract_images_and_params

    form = _FakeFormData({"confidence": "0.5"})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert imgs == []
    assert params["confidence"] == "0.5"


@pytest.mark.asyncio
async def test_json_base64_single():
    from inference_server.framework.input_parsers import extract_images_and_params

    b64 = base64.b64encode(_JPEG).decode()
    body = {"inputs": {"image": {"type": "base64", "value": b64}, "confidence": 0.5}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params["confidence"] == 0.5


@pytest.mark.asyncio
async def test_json_missing_image():
    from inference_server.framework.input_parsers import extract_images_and_params

    body = {"inputs": {"confidence": 0.5}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await extract_images_and_params(req)
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_json_invalid_base64():
    from inference_server.framework.input_parsers import extract_images_and_params

    body = {
        "inputs": {"image": {"type": "base64", "value": "!!!===not valid base64===!!!"}}
    }
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await extract_images_and_params(req)
    # base64.b64decode is lenient — no crash is the requirement
    assert err is None or err.status_code == 400


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multipart_form_batch():
    from inference_server.framework.input_parsers import extract_images_and_params

    jpeg2 = _JPEG + b"\x00"  # slightly different
    form = _FakeFormData({"image": [_FakeUploadFile(_JPEG), _FakeUploadFile(jpeg2)]})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 2
    assert imgs[0] == _JPEG
    assert imgs[1] == jpeg2


@pytest.mark.asyncio
async def test_json_base64_batch():
    from inference_server.framework.input_parsers import extract_images_and_params

    b64 = base64.b64encode(_JPEG).decode()
    body = {
        "inputs": {
            "image": [
                {"type": "base64", "value": b64},
                {"type": "base64", "value": b64},
            ]
        }
    }
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 2


# ---------------------------------------------------------------------------
# URL fetch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_image_from_url_invalid_scheme():
    from inference_server.framework.input_parsers import fetch_image_from_url

    _, err = await fetch_image_from_url("ftp://example.com/image.jpg")
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_fetch_image_from_url_bad_domain():
    from inference_server.framework.input_parsers import fetch_image_from_url

    _, err = await fetch_image_from_url(
        "https://this-domain-does-not-exist-12345.invalid/img.jpg"
    )
    assert err is not None
    assert err.status_code in (502, 504)


class _FakeContent:
    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    def iter_chunked(self, size: int):
        return self._aiter()

    async def _aiter(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResp:
    def __init__(self, chunks: list[bytes], content_length=None):
        self.status = 200
        self.content_length = content_length
        self.content = _FakeContent(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, resp: _FakeResp):
        self._resp = resp

    def get(self, url):
        return self._resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _patch_http(chunks: list[bytes], content_length=None):
    from unittest.mock import patch

    resp = _FakeResp(chunks, content_length=content_length)
    return patch(
        "inference_server.framework.input_parsers.url_fetch.aiohttp.ClientSession",
        return_value=_FakeSession(resp),
    )


@pytest.mark.asyncio
async def test_fetch_image_from_url_chunked_over_cap_413():
    """No Content-Length on the response: cap must fire while streaming."""
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_image_from_url

    with _patch_http([b"x" * 8] * 10), patch(
        "inference_server.framework.input_parsers.url_fetch.URL_FETCH_MAX_BYTES", 16
    ):
        data, err = await fetch_image_from_url("https://example.com/img.jpg")
    assert data is None
    assert err.status_code == 413


@pytest.mark.asyncio
async def test_fetch_images_from_urls_too_many_urls_400():
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_images_from_urls

    with patch(
        "inference_server.framework.input_parsers.url_fetch.configuration.MAX_IMAGE_URLS",
        2,
    ):
        images, err = await fetch_images_from_urls(
            ["https://e.com/1.jpg", "https://e.com/2.jpg", "https://e.com/3.jpg"]
        )
    assert images is None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_fetch_images_from_urls_aggregate_over_budget_413():
    """Each URL is under the per-fetch cap but the sum exceeds the budget."""
    from unittest.mock import patch

    from inference_server.framework.input_parsers import fetch_images_from_urls

    with _patch_http([b"x" * 8]), patch(
        "inference_server.framework.input_parsers.url_fetch.configuration.MAX_BODY_BYTES",
        20,
    ):
        images, err = await fetch_images_from_urls(
            ["https://e.com/1.jpg", "https://e.com/2.jpg", "https://e.com/3.jpg"]
        )
    assert images is None
    assert err.status_code == 413


@pytest.mark.asyncio
async def test_fetch_images_from_urls_under_limits_ok():
    from inference_server.framework.input_parsers import fetch_images_from_urls

    with _patch_http([b"x" * 8]):
        images, err = await fetch_images_from_urls(
            ["https://e.com/1.jpg", "https://e.com/2.jpg"]
        )
    assert err is None
    assert images == [b"x" * 8, b"x" * 8]
