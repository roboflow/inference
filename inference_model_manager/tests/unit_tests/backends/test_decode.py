"""Unit tests for decode.py — make_decoder and make_batch_decoder."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest

from inference_model_manager.backends.decode import (
    _decode_ic,
    _select_codec,
    make_decoder,
)


@contextmanager
def _capture_fd_stderr():
    """Capture OS-level fd 2 so C-library prints (OpenEXR) are seen, not just
    Python sys.stderr. Yields a dict whose 'text' key holds captured output
    after the context exits."""
    holder: dict[str, str] = {}
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    try:
        yield holder
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        tmp.flush()
        tmp.seek(0)
        holder["text"] = tmp.read().decode("utf-8", "replace")
        tmp.close()


# ---------------------------------------------------------------------------
# Fixtures — minimal valid JPEG and PNG bytes
# ---------------------------------------------------------------------------


def _make_rgb_array(h: int = 8, w: int = 8) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def jpeg_bytes() -> bytes:
    imagecodecs = pytest.importorskip("imagecodecs")
    return bytes(imagecodecs.jpeg_encode(_make_rgb_array()))


@pytest.fixture(scope="module")
def png_bytes() -> bytes:
    imagecodecs = pytest.importorskip("imagecodecs")
    return bytes(imagecodecs.png_encode(_make_rgb_array()))


@pytest.fixture(scope="module")
def webp_bytes() -> bytes:
    imagecodecs = pytest.importorskip("imagecodecs")
    return bytes(imagecodecs.webp_encode(_make_rgb_array()))


@pytest.fixture(scope="module")
def gray_jpeg_bytes() -> bytes:
    imagecodecs = pytest.importorskip("imagecodecs")
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 255, (8, 8), dtype=np.uint8)
    return bytes(imagecodecs.jpeg_encode(gray))


@pytest.fixture(scope="module")
def rgba_png_bytes() -> bytes:
    imagecodecs = pytest.importorskip("imagecodecs")
    rng = np.random.default_rng(42)
    rgba = rng.integers(0, 255, (8, 8, 4), dtype=np.uint8)
    return bytes(imagecodecs.png_encode(rgba))


# ---------------------------------------------------------------------------
# _select_codec / _decode_ic — explicit codec dispatch (no imread all-codec
# probe, which makes the bundled OpenEXR codec spam stderr on every image)
# ---------------------------------------------------------------------------


class TestSelectCodec:
    @pytest.mark.parametrize(
        "head,codec",
        [
            (b"\xff\xd8\xff\xe0", "jpeg"),
            (b"\xff\xd8\xff\xe1", "jpeg"),
            (b"\x89PNG\r\n\x1a\n", "png"),
            (b"RIFF\x00\x00\x00\x00WEBP", "webp"),
            (b"GIF89a", "gif"),
            (b"II*\x00", "tiff"),
            (b"MM\x00*", "tiff"),
            (b"BM\x00\x00", "bmp"),
            (b"\x00\x00\x00\x0cjP  ", "jpeg2k"),
            (b"\xff\x4f\xff\x51", "jpeg2k"),
        ],
    )
    def test_known_magic(self, head, codec):
        assert _select_codec(head) == codec

    def test_unknown_returns_none(self):
        assert _select_codec(bytes(range(12))) is None


class TestDecodeIcNoExrProbe:
    """Regression: decoding must not invoke the OpenEXR codec on non-EXR images.

    imagecodecs.imread() probes every codec, so the bundled OpenEXR codec writes
    'EXR_ERR_FILE_BAD_HEADER' to C stderr on every JPEG/WebP. _decode_ic dispatches
    by magic and must stay silent.
    """

    @pytest.mark.parametrize("fixture", ["jpeg_bytes", "png_bytes", "webp_bytes"])
    def test_no_exr_stderr(self, fixture, request):
        data = request.getfixturevalue(fixture)
        with _capture_fd_stderr() as cap:
            out = _decode_ic(data)
        err = cap["text"]
        assert "EXR" not in err, f"EXR probe leaked to stderr: {err!r}"
        assert isinstance(out, np.ndarray)
        assert out.shape[2] == 3  # HWC RGB


class TestDecodeIcChannelNormalization:
    """_decode_ic promises RGB HWC — grayscale/alpha inputs must be normalized."""

    def test_gray_jpeg_returns_hwc_3(self, gray_jpeg_bytes):
        out = _decode_ic(gray_jpeg_bytes)
        assert out.ndim == 3
        assert out.shape[2] == 3
        assert out.dtype == np.uint8

    def test_rgba_png_returns_hwc_3(self, rgba_png_bytes):
        out = _decode_ic(rgba_png_bytes)
        assert out.ndim == 3
        assert out.shape[2] == 3
        assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# make_decoder — single-image
# ---------------------------------------------------------------------------


class TestMakeDecoder:
    def test_imagecodecs_returns_rgb_hwc_numpy(self, jpeg_bytes):
        decode = make_decoder("imagecodecs")
        result = decode(jpeg_bytes)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 3  # HWC

    def test_imagecodecs_uint8(self, jpeg_bytes):
        decode = make_decoder("imagecodecs")
        assert decode(jpeg_bytes).dtype == np.uint8

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown decoder"):
            make_decoder("cv2")
