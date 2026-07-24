"""Unit tests for image_headers.image_pixels — the resolution reject gate."""

from __future__ import annotations

import io

import pytest

from inference_model_manager.backends.utils.image_headers import (
    _OVERSIZED,
    image_dims,
    image_pixels,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _encode(w: int, h: int, fmt: str) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (w, h)).save(b, format=fmt)
    return b.getvalue()


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_pixels_from_header(fmt):
    assert image_pixels(_encode(320, 240, fmt)) == 320 * 240


def test_full_bytes_yield_dims_without_decode():
    # Pillow stays lazy on full bytes — dims correct, no decode triggered.
    data = _encode(4032, 3024, "JPEG")
    assert image_pixels(data) == 4032 * 3024


def test_unknown_bytes_return_none():
    # Unparseable → None so the gate never false-rejects.
    assert image_pixels(b"\x00\x01\x02\x03not-an-image") is None


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_dims_from_header(fmt):
    assert image_dims(_encode(320, 240, fmt)) == (320, 240)


def test_dims_unknown_bytes_return_none():
    assert image_dims(b"\x00\x01\x02\x03not-an-image") is None


def test_bomb_returns_sentinel_not_none_and_no_global_mutation():
    # A decompression-bomb (Pillow refuses to report size) must reject, not slip
    # through as None. We return the _OVERSIZED sentinel and never touch the
    # process-global MAX_IMAGE_PIXELS.
    data = _encode(4032, 3024, "JPEG")
    saved = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 2000  # force the bomb guard to trip on open
    try:
        assert image_pixels(data) == _OVERSIZED
        assert Image.MAX_IMAGE_PIXELS == 2000  # helper did not mutate it
    finally:
        Image.MAX_IMAGE_PIXELS = saved
