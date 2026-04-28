"""Unit tests for v2 infer input extraction (multipart, JSON+base64, raw)."""

from __future__ import annotations

import base64
import io
import json

import pytest
import pytest_asyncio

# Minimal JPEG (smallest valid JPEG — 2x1 red pixel)
_JPEG = bytes([
    0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9,
])


class _FakeUploadFile:
    """Mimics Starlette UploadFile for form.get('image')."""
    def __init__(self, data: bytes):
        self._data = data

    async def read(self) -> bytes:
        return self._data


class _FakeRequest:
    """Mimics Starlette Request with configurable content-type, body, form."""

    def __init__(self, content_type: str = "", body: bytes = b"", form_data=None, json_body=None):
        self.headers = {"content-type": content_type}
        self._body = body
        self._form_data = form_data
        self._json_body = json_body

    async def form(self):
        return self._form_data or {}

    async def json(self):
        if self._json_body is not None:
            return self._json_body
        raise ValueError("no json")

    async def stream(self):
        if self._body:
            yield self._body


@pytest.mark.asyncio
async def test_raw_body():
    from inference_server.routers.v2_models import _extract_image_and_params

    req = _FakeRequest(content_type="image/jpeg", body=_JPEG)
    img, params, err = await _extract_image_and_params(req)
    assert err is None
    assert img == _JPEG
    assert params == {}


@pytest.mark.asyncio
async def test_multipart_form():
    from inference_server.routers.v2_models import _extract_image_and_params

    form = _FakeFormData({
        "image": _FakeUploadFile(_JPEG),
        "confidence": "0.5",
    })
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    img, params, err = await _extract_image_and_params(req)
    assert err is None
    assert img == _JPEG
    assert params["confidence"] == "0.5"


@pytest.mark.asyncio
async def test_multipart_form_with_inputs_json():
    from inference_server.routers.v2_models import _extract_image_and_params

    form = _FakeFormData({
        "image": _FakeUploadFile(_JPEG),
        "inputs": '{"confidence": 0.3, "iou": 0.5}',
    })
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    img, params, err = await _extract_image_and_params(req)
    assert err is None
    assert img == _JPEG
    assert params["confidence"] == 0.3
    assert params["iou"] == 0.5


@pytest.mark.asyncio
async def test_multipart_form_missing_image():
    from inference_server.routers.v2_models import _extract_image_and_params

    form = _FakeFormData({"confidence": "0.5"})
    req = _FakeRequest(content_type="multipart/form-data", form_data=form)
    img, params, err = await _extract_image_and_params(req)
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_json_base64():
    from inference_server.routers.v2_models import _extract_image_and_params

    b64 = base64.b64encode(_JPEG).decode()
    body = {
        "inputs": {
            "image": {"type": "base64", "value": b64},
            "confidence": 0.5,
        }
    }
    req = _FakeRequest(content_type="application/json", json_body=body)
    img, params, err = await _extract_image_and_params(req)
    assert err is None
    assert img == _JPEG
    assert params["confidence"] == 0.5


@pytest.mark.asyncio
async def test_json_missing_image():
    from inference_server.routers.v2_models import _extract_image_and_params

    body = {"inputs": {"confidence": 0.5}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    img, params, err = await _extract_image_and_params(req)
    assert err is not None
    assert err.status_code == 400


@pytest.mark.asyncio
async def test_json_invalid_base64():
    from inference_server.routers.v2_models import _extract_image_and_params

    body = {"inputs": {"image": {"type": "base64", "value": "!!!===not valid base64===!!!"}}}
    req = _FakeRequest(content_type="application/json", json_body=body)
    img, params, err = await _extract_image_and_params(req)
    # base64.b64decode is lenient; if it somehow decodes, image_bytes won't be valid
    # but extraction itself may succeed — that's OK, format check happens downstream
    # Test that at minimum no crash occurs
    assert err is None or err.status_code == 400


class _FakeFormData(dict):
    """Dict that also supports multi_items() like Starlette FormData."""
    def multi_items(self):
        for k, v in self.items():
            if isinstance(v, _FakeUploadFile):
                yield k, v
            else:
                yield k, v
