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


# ---------------------------------------------------------------------------
# Decompression-bomb gate
# ---------------------------------------------------------------------------


class TestDecodedPixelGate:
    def test_header_pixels_reads_jpeg_dimensions(self, jpeg_bytes):
        from inference_model_manager.backends.decode import header_pixels

        assert header_pixels(jpeg_bytes) == 8 * 8

    def test_header_pixels_reads_png_dimensions(self, png_bytes):
        from inference_model_manager.backends.decode import header_pixels

        assert header_pixels(png_bytes) == 8 * 8

    def test_header_pixels_unknown_format_is_zero(self, webp_bytes):
        from inference_model_manager.backends.decode import header_pixels

        assert header_pixels(webp_bytes) == 0
        assert header_pixels(b"") == 0
        assert header_pixels(b"not an image at all") == 0

    def test_header_pixels_sees_a_forged_giant_png_header(self):
        from inference_model_manager.backends.decode import header_pixels

        header = (
            b"\x89PNG\r\n\x1a\n"
            + (13).to_bytes(4, "big")
            + b"IHDR"
            + (60000).to_bytes(4, "big")
            + (60000).to_bytes(4, "big")
        )
        assert header_pixels(header) == 60000 * 60000

    def test_gate_is_nonzero_by_default(self):
        from inference_model_manager.backends.decode import max_decoded_pixels

        assert max_decoded_pixels() > 0

    def test_oversized_png_rejected_from_header_without_decoding(self, monkeypatch):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 0.00001)

        def _explode(_data):
            raise AssertionError("decode must not run for an oversized header")

        monkeypatch.setattr(decode_mod, "_decode_ic", _explode)
        decode = decode_mod.make_decoder("imagecodecs")
        header = (
            b"\x89PNG\r\n\x1a\n"
            + (13).to_bytes(4, "big")
            + b"IHDR"
            + (60000).to_bytes(4, "big")
            + (60000).to_bytes(4, "big")
        )
        with pytest.raises(ValueError, match="megapixel decode limit"):
            decode(header)

    def test_oversized_jpeg_rejected_from_header(self, monkeypatch, jpeg_bytes):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 0.00001)
        decode = decode_mod.make_decoder("imagecodecs")
        with pytest.raises(ValueError, match="header"):
            decode(jpeg_bytes)

    def test_backstop_rejects_after_decode_when_header_is_unreadable(
        self, monkeypatch, webp_bytes
    ):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 0.00001)
        decode = decode_mod.make_decoder("imagecodecs")
        with pytest.raises(ValueError, match="decoded"):
            decode(webp_bytes)

    def test_normal_image_passes_the_gate(self, jpeg_bytes, png_bytes, webp_bytes):
        from inference_model_manager.backends.decode import make_decoder

        decode = make_decoder("imagecodecs")
        for data in (jpeg_bytes, png_bytes, webp_bytes):
            assert decode(data).shape[:2] == (8, 8)

    def test_gate_can_be_disabled(self, monkeypatch, webp_bytes):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 0.0)
        decode = decode_mod.make_decoder("imagecodecs")
        assert decode(webp_bytes).shape[:2] == (8, 8)

    def test_decoded_pixels_reads_hwc_and_chw_alike(self):
        from inference_model_manager.backends.decode import decoded_pixels

        assert decoded_pixels(np.zeros((16, 32, 3), dtype=np.uint8)) == 16 * 32
        assert decoded_pixels(np.zeros((3, 16, 32), dtype=np.uint8)) == 16 * 32
        assert decoded_pixels(np.zeros((16, 32), dtype=np.uint8)) == 16 * 32
        assert decoded_pixels(b"not an array") == 0


def jpeg_with_sof_past(width: int, height: int, pad_segments: int = 2) -> bytes:
    """A valid JPEG whose SOF marker sits behind fat APP segments."""
    out = bytearray(b"\xff\xd8")
    payload = b"\x00" * 65533  # the largest a length-prefixed segment can carry
    for marker in (b"\xff\xe1", b"\xff\xe2")[:pad_segments]:
        out += marker + (len(payload) + 2).to_bytes(2, "big") + payload
    out += b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08"
    out += height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03"
    out += b"\x00" * 9
    out += b"\xff\xd9"
    return bytes(out)


class TestHeaderWalkIsNotTruncated:
    """A fat EXIF/ICC chain can push SOF well past any fixed prefix."""

    def test_sof_behind_64kib_of_app_segments_is_found(self):
        from inference_model_manager.backends.decode import header_pixels

        data = jpeg_with_sof_past(40000, 40000)
        assert len(data) > 65536
        assert header_pixels(data) == 40000 * 40000

    def test_header_walk_accepts_a_memoryview(self, jpeg_bytes, png_bytes):
        from inference_model_manager.backends.decode import header_pixels

        assert header_pixels(memoryview(jpeg_bytes)) == 8 * 8
        assert header_pixels(memoryview(png_bytes)) == 8 * 8
        assert header_pixels(memoryview(jpeg_with_sof_past(40000, 40000))) == 40000**2

    def test_oversized_padded_jpeg_rejected_without_decoding(self, monkeypatch):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 100.0)

        def _explode(_data):
            raise AssertionError("decode must not run for an oversized header")

        monkeypatch.setattr(decode_mod, "_decode_ic", _explode)
        decode = decode_mod.make_decoder("imagecodecs")
        with pytest.raises(ValueError, match="header"):
            decode(jpeg_with_sof_past(40000, 40000))


class TestImagecodecsRejectsBeforeTheCopy:
    """Headerless formats reach the backstop; it must fire before the BGR copy
    so an oversized decode is never allocated twice."""

    _WEBP_LIKE = b"RIFF\x00\x00\x00\x00WEBPVP8 "

    def test_factory_itself_raises(self, monkeypatch):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 0.00001)
        monkeypatch.setattr(
            decode_mod, "_decode_ic", lambda data: np.zeros((8, 8, 3), dtype=np.uint8)
        )
        # The factory, not make_decoder — the wrapper is not involved here.
        decode = decode_mod._imagecodecs_factory("cpu")
        with pytest.raises(ValueError, match="decoded"):
            decode(self._WEBP_LIKE)

    def test_error_shape_matches_the_wrapper(self, monkeypatch):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 0.00001)
        oversized = np.zeros((8, 8, 3), dtype=np.uint8)
        monkeypatch.setattr(decode_mod, "_decode_ic", lambda data: oversized)

        with pytest.raises(ValueError) as from_factory:
            decode_mod._imagecodecs_factory("cpu")(self._WEBP_LIKE)
        with pytest.raises(ValueError) as from_wrapper:
            decode_mod._guarded(lambda data: oversized)(self._WEBP_LIKE)

        assert str(from_factory.value) == str(from_wrapper.value)

    def test_normal_image_still_returns_bgr(self, webp_bytes):
        import inference_model_manager.backends.decode as decode_mod

        out = decode_mod._imagecodecs_factory("cpu")(webp_bytes)
        assert out.shape == (8, 8, 3)


class TestPluginDecoderGate:
    """Entry-point decoders (nvjpeg/nvimgcodec) go through the same gate."""

    _GIANT_PNG_HEADER = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + (60000).to_bytes(4, "big")
        + (60000).to_bytes(4, "big")
    )

    @contextmanager
    def _registered(self, decode):
        import inference_model_manager.backends.decode as decode_mod

        decode_mod.register_decoder("fake-gpu", lambda device: decode)
        try:
            yield decode_mod.make_decoder("fake-gpu", device="cpu")
        finally:
            decode_mod.DECODER_FACTORIES.pop("fake-gpu", None)

    def test_rejected_on_oversized_header_without_decoding(self, monkeypatch):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 0.00001)

        def _explode(_data):
            raise AssertionError("plugin decode must not run for an oversized header")

        with self._registered(_explode) as decode:
            with pytest.raises(ValueError, match="megapixel decode limit"):
                decode(self._GIANT_PNG_HEADER)

    def test_rejected_on_oversized_decoded_output(self, monkeypatch, webp_bytes):
        import inference_model_manager.backends.decode as decode_mod

        monkeypatch.setattr(decode_mod.cfg, "INFERENCE_DECODE_MAX_MEGAPIXELS", 0.00001)
        # CHW, the layout the GPU plugin decoders emit.
        chw = np.zeros((3, 8, 8), dtype=np.uint8)

        with self._registered(lambda data: chw) as decode:
            with pytest.raises(ValueError, match="decoded"):
                decode(webp_bytes)

    def test_normal_output_passes(self, webp_bytes):
        chw = np.zeros((3, 8, 8), dtype=np.uint8)
        with self._registered(lambda data: chw) as decode:
            assert decode(webp_bytes) is chw
