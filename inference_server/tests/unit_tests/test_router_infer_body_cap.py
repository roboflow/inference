"""Body-size ceiling on the legacy POST /infer path (bundled + chunked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Request

from inference_server.routers.infer import infer

_JPEG = bytes(
    [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46]
    + [0x00] * 12
    + [0xFF, 0xD9]
)


def _request(headers: list | None = None, chunks: list[bytes] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/infer",
        "query_string": b"model_id=m",
        "headers": headers or [],
    }
    pending = list(chunks or [])

    async def recv():
        if pending:
            body = pending.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(pending)}
        return {"type": "http.disconnect"}

    return Request(scope, recv)


class _BundledProxy:
    """MMWrapper-like proxy: no infer_stream, no shm_data_size."""

    def __init__(self):
        self.ensure_loaded = AsyncMock(return_value=("model_ready",))
        self.infer = AsyncMock(return_value=b"pickled-result")


def _stat_ok():
    return patch(
        "inference_server.routers.infer.stat_model_while_checking_auth",
        new=AsyncMock(return_value=("object-detection", "infer")),
    )


def _cap(n: int):
    return patch("inference_server.routers.infer.configuration.MAX_BODY_BYTES", n)


@pytest.mark.asyncio
async def test_bundled_content_length_over_cap_returns_413():
    mm = _BundledProxy()
    req = _request(
        headers=[(b"content-length", b"100")],
        chunks=[_JPEG + b"x" * 76],
    )
    with _stat_ok(), _cap(16):
        r = await infer(req, api_key="k", mm=mm)
    assert r.status_code == 413
    mm.infer.assert_not_awaited()


@pytest.mark.asyncio
async def test_bundled_chunked_body_over_cap_returns_413():
    """No Content-Length: cap must fire while buffering the body."""
    mm = _BundledProxy()
    req = _request(chunks=[_JPEG, b"x" * 50, b"x" * 50])
    with _stat_ok(), _cap(16):
        r = await infer(req, api_key="k", mm=mm)
    assert r.status_code == 413
    mm.infer.assert_not_awaited()


@pytest.mark.asyncio
async def test_bundled_body_under_cap_still_infers():
    mm = _BundledProxy()
    req = _request(chunks=[_JPEG])
    with _stat_ok(), _cap(1024):
        r = await infer(req, api_key="k", mm=mm)
    assert r.status_code == 200
    mm.infer.assert_awaited_once()
