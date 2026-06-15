"""Unit tests for decode.py — make_decoder and make_batch_decoder."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from inference_model_manager.backends.decode import (
    Decoder,
    _decode_ic,
    _select_codec,
    make_batch_decoder,
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

    def test_nvjpeg_jpeg_returns_chw_tensor(self, jpeg_bytes):
        decode = make_decoder("nvjpeg", device="cpu")
        result = decode(jpeg_bytes)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.shape[0] == 3  # CHW

    def test_nvjpeg_jpeg_uint8(self, jpeg_bytes):
        decode = make_decoder("nvjpeg", device="cpu")
        assert decode(jpeg_bytes).dtype == torch.uint8

    def test_nvjpeg_non_jpeg_falls_back_to_imagecodecs(self, png_bytes):
        """Non-JPEG input with nvjpeg decoder falls back to imagecodecs."""
        decode = make_decoder("nvjpeg", device="cpu")
        result = decode(png_bytes)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 3
        assert result.shape[0] == 3  # CHW

    def test_nvjpeg_gray_jpeg_returns_3_channels(self, gray_jpeg_bytes):
        decode = make_decoder("nvjpeg", device="cpu")
        result = decode(gray_jpeg_bytes)
        assert result.shape[0] == 3

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown decoder"):
            make_decoder("cv2")


# ---------------------------------------------------------------------------
# make_batch_decoder — decoder=Decoder.IMAGECODECS (imagecodecs path)
# ---------------------------------------------------------------------------


class TestMakeBatchDecoderImagecodecs:
    def test_jpeg_batch_shape(self, jpeg_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.IMAGECODECS)
        results = decode([memoryview(jpeg_bytes), memoryview(jpeg_bytes)])
        assert len(results) == 2
        for t in results:
            assert isinstance(t, torch.Tensor)
            assert t.ndim == 3
            assert t.shape[0] == 3  # CHW

    def test_png_batch_shape(self, png_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.IMAGECODECS)
        results = decode([memoryview(png_bytes)])
        assert len(results) == 1
        assert results[0].shape[0] == 3

    def test_mixed_batch_length_preserved(self, jpeg_bytes, png_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.IMAGECODECS)
        mvs = [memoryview(jpeg_bytes), memoryview(png_bytes), memoryview(jpeg_bytes)]
        results = decode(mvs)
        assert len(results) == 3

    def test_output_uint8(self, jpeg_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.IMAGECODECS)
        results = decode([memoryview(jpeg_bytes)])
        assert results[0].dtype == torch.uint8

    def test_output_cpu_tensor(self, jpeg_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.IMAGECODECS)
        results = decode([memoryview(jpeg_bytes)])
        assert results[0].device.type == "cpu"

    def test_gray_jpeg_returns_3_channels(self, gray_jpeg_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.IMAGECODECS)
        results = decode([memoryview(gray_jpeg_bytes)])
        assert results[0].shape[0] == 3

    def test_single_image_matches_make_decoder(self, jpeg_bytes):
        """Batch decoder with one JPEG should produce same CHW shape as single decoder."""
        batch_decode = make_batch_decoder("cpu", decoder=Decoder.IMAGECODECS)
        single_decode = make_decoder("imagecodecs")
        single_result = single_decode(jpeg_bytes)  # HWC numpy
        batch_result = batch_decode([memoryview(jpeg_bytes)])[0]  # CHW tensor
        # shape consistency: HWC vs CHW — spatial dims match
        assert batch_result.shape[1] == single_result.shape[0]  # H
        assert batch_result.shape[2] == single_result.shape[1]  # W
        assert batch_result.shape[0] == single_result.shape[2]  # C


# ---------------------------------------------------------------------------
# make_batch_decoder — decoder=Decoder.NVJPEG
# ---------------------------------------------------------------------------


class TestMakeBatchDecoderNvjpeg:
    def test_jpeg_batch_shape(self, jpeg_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.NVJPEG)
        results = decode([memoryview(jpeg_bytes), memoryview(jpeg_bytes)])
        assert len(results) == 2
        for t in results:
            assert isinstance(t, torch.Tensor)
            assert t.ndim == 3
            assert t.shape[0] == 3

    def test_jpeg_uint8(self, jpeg_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.NVJPEG)
        results = decode([memoryview(jpeg_bytes)])
        assert results[0].dtype == torch.uint8

    def test_non_jpeg_falls_back_to_imagecodecs(self, png_bytes):
        """decoder=Decoder.NVJPEG must still handle non-JPEG via imagecodecs."""
        decode = make_batch_decoder("cpu", decoder=Decoder.NVJPEG)
        results = decode([memoryview(png_bytes)])
        assert len(results) == 1
        assert results[0].shape[0] == 3

    def test_mixed_batch_order_preserved(self, jpeg_bytes, png_bytes):
        """JPEG and non-JPEG items must appear at their original indices."""
        decode = make_batch_decoder("cpu", decoder=Decoder.NVJPEG)
        # index 0=JPEG, 1=PNG, 2=JPEG
        mvs = [memoryview(jpeg_bytes), memoryview(png_bytes), memoryview(jpeg_bytes)]
        results = decode(mvs)
        assert len(results) == 3
        for t in results:
            assert t.shape[0] == 3

    def test_mixed_batch_spatial_dims_consistent(self, jpeg_bytes, png_bytes):
        """All outputs in a mixed batch should have same H, W (same source image)."""
        decode = make_batch_decoder("cpu", decoder=Decoder.NVJPEG)
        results = decode([memoryview(jpeg_bytes), memoryview(png_bytes)])
        # Both encoded from the same 8x8 image
        assert results[0].shape[1:] == results[1].shape[1:]

    def test_pure_jpeg_batch_only_uses_torchvision(self, jpeg_bytes, monkeypatch):
        """For an all-JPEG batch, imagecodecs.imread should never be called."""
        import inference_model_manager.backends.decode as _dec

        original_ic = pytest.importorskip("imagecodecs")

        called = []

        def _spy_imread(data):
            called.append(True)
            return original_ic.imread(data)

        monkeypatch.setattr(
            _dec,
            "make_batch_decoder",
            lambda *a, **kw: make_batch_decoder(*a, **kw),
        )
        # Patch imagecodecs inside the closure by creating a fresh decoder
        # and verifying no fallback is triggered for JPEG-only input.
        decode = make_batch_decoder("cpu", decoder=Decoder.NVJPEG)

        # Wrap the actual imagecodecs in the already-closed-over module ref
        import imagecodecs as ic_module

        original_imread = ic_module.imread
        ic_module.imread = _spy_imread
        try:
            decode([memoryview(jpeg_bytes), memoryview(jpeg_bytes)])
        finally:
            ic_module.imread = original_imread

        assert not called, "imagecodecs.imread should not be called for JPEG-only batch"

    def test_gray_jpeg_returns_3_channels(self, gray_jpeg_bytes):
        decode = make_batch_decoder("cpu", decoder=Decoder.NVJPEG)
        results = decode([memoryview(gray_jpeg_bytes)])
        assert results[0].shape[0] == 3

    def test_gray_jpeg_cpu_fallback_returns_3_channels(
        self, gray_jpeg_bytes, monkeypatch
    ):
        """Regression: grayscale JPEG decoded via the CPU fallback (nvjpeg batch
        failed) crashed permute(2, 0, 1) on the 2D array and left the slot None."""
        import torchvision.io

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated nvjpeg failure")

        monkeypatch.setattr(torchvision.io, "decode_jpeg", _raise)
        decode = make_batch_decoder("cpu", decoder=Decoder.NVJPEG)
        results = decode([memoryview(gray_jpeg_bytes)])
        assert results[0] is not None
        assert isinstance(results[0], torch.Tensor)
        assert results[0].shape[0] == 3

    @pytest.mark.cuda
    def test_cuda_output_device(self, jpeg_bytes):
        decode = make_batch_decoder("cuda:0", decoder=Decoder.NVJPEG)
        results = decode([memoryview(jpeg_bytes)])
        assert results[0].device.type == "cuda"
        assert results[0].shape[0] == 3

    @pytest.mark.cuda
    def test_cuda_non_jpeg_on_device(self, png_bytes):
        decode = make_batch_decoder("cuda:0", decoder=Decoder.NVJPEG)
        results = decode([memoryview(png_bytes)])
        assert results[0].device.type == "cuda"
