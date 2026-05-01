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
    from inference_server.routers.v2_models import _extract_images_and_params

    req = _FakeRequest(content_type="image/jpeg", body=_JPEG)
    imgs, params, err = await _extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params == {}


@pytest.mark.asyncio
async def test_multipart_form_single():
    from inference_server.routers.v2_models import _extract_images_and_params

    form = _FakeFormData({"image": _FakeUploadFile(_JPEG), "confidence": "0.5"})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await _extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params["confidence"] == "0.5"


@pytest.mark.asyncio
async def test_multipart_form_with_inputs_json():
    from inference_server.routers.v2_models import _extract_images_and_params

    form = _FakeFormData(
        {
            "image": _FakeUploadFile(_JPEG),
            "inputs": '{"confidence": 0.3, "iou": 0.5}',
        }
    )
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await _extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert params["confidence"] == 0.3
    assert params["iou"] == 0.5


@pytest.mark.asyncio
async def test_multipart_form_missing_image():
    from inference_server.routers.v2_models import _extract_images_and_params

    form = _FakeFormData({"confidence": "0.5"})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await _extract_images_and_params(req)
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_json_base64_single():
    from inference_server.routers.v2_models import _extract_images_and_params

    b64 = base64.b64encode(_JPEG).decode()
    body = {"inputs": {"image": {"type": "base64", "value": b64}, "confidence": 0.5}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await _extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 1
    assert imgs[0] == _JPEG
    assert params["confidence"] == 0.5


@pytest.mark.asyncio
async def test_json_missing_image():
    from inference_server.routers.v2_models import _extract_images_and_params

    body = {"inputs": {"confidence": 0.5}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await _extract_images_and_params(req)
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_json_invalid_base64():
    from inference_server.routers.v2_models import _extract_images_and_params

    body = {
        "inputs": {"image": {"type": "base64", "value": "!!!===not valid base64===!!!"}}
    }
    req = _FakeRequest(content_type="application/json", json_body=body)
    imgs, params, err = await _extract_images_and_params(req)
    # base64.b64decode is lenient — no crash is the requirement
    assert err is None or err.status_code == 400


# ---------------------------------------------------------------------------
# Batch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multipart_form_batch():
    from inference_server.routers.v2_models import _extract_images_and_params

    jpeg2 = _JPEG + b"\x00"  # slightly different
    form = _FakeFormData({"image": [_FakeUploadFile(_JPEG), _FakeUploadFile(jpeg2)]})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    imgs, params, err = await _extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 2
    assert imgs[0] == _JPEG
    assert imgs[1] == jpeg2


@pytest.mark.asyncio
async def test_json_base64_batch():
    from inference_server.routers.v2_models import _extract_images_and_params

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
    imgs, params, err = await _extract_images_and_params(req)
    assert err is None
    assert len(imgs) == 2


# ---------------------------------------------------------------------------
# URL fetch tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_image_from_url_invalid_scheme():
    from inference_server.routers.v2_models import _fetch_image_from_url

    _, err = await _fetch_image_from_url("ftp://example.com/image.jpg")
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_fetch_image_from_url_bad_domain():
    from inference_server.routers.v2_models import _fetch_image_from_url

    _, err = await _fetch_image_from_url(
        "https://this-domain-does-not-exist-12345.invalid/img.jpg"
    )
    assert err is not None
    assert err.status_code in (502, 504)
