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
    apply_exif_orientation,
    decoded_dims,
    exif_orientation,
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
            (b"\x00\x00\x00\x1cftypavif", "avif"),
            (b"\x00\x00\x00\x1cftypavis", "avif"),
        ],
    )
    def test_known_magic(self, head, codec):
        assert _select_codec(head) == codec

    def test_unknown_returns_none(self):
        assert _select_codec(bytes(range(12))) is None

    def test_heic_is_not_dispatched(self):
        assert _select_codec(b"\x00\x00\x00\x1cftypheic") is None


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
# EXIF orientation — the decoder must return upright pixels, like cv2.imdecode
# ---------------------------------------------------------------------------


def _upright_blocks(h: int = 32, w: int = 48) -> np.ndarray:
    """Four flat quadrants: survive JPEG and reveal any rotation or flip."""
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[: h // 2, : w // 2] = (255, 0, 0)
    image[: h // 2, w // 2 :] = (0, 255, 0)
    image[h // 2 :, : w // 2] = (0, 0, 255)
    image[h // 2 :, w // 2 :] = (255, 255, 0)
    return image


def _jpeg_with_orientation(pixels: np.ndarray, orientation: int) -> bytes:
    import io

    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=95, exif=exif)
    return buffer.getvalue()


def _assert_close(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    assert diff.mean() < 8, f"mean abs diff {diff.mean():.1f}"


class TestExifOrientation:
    @pytest.mark.parametrize(
        "orientation,stored",
        [
            # Pixels stored so that applying the tag restores the upright image.
            (3, lambda u: u[::-1, ::-1]),
            (6, lambda u: np.rot90(u, k=1)),
            (8, lambda u: np.rot90(u, k=-1)),
        ],
    )
    def test_decoder_returns_upright_bgr(self, orientation, stored):
        pytest.importorskip("imagecodecs")
        upright = _upright_blocks()
        pixels = np.ascontiguousarray(stored(upright))
        data = _jpeg_with_orientation(pixels, orientation)
        result = make_decoder("imagecodecs")(data)
        _assert_close(result, upright[..., ::-1])
        assert result.flags["C_CONTIGUOUS"]

    @pytest.mark.parametrize("orientation", range(1, 9))
    def test_matches_pil_exif_transpose(self, orientation):
        pytest.importorskip("imagecodecs")
        import io

        from PIL import Image, ImageOps

        data = _jpeg_with_orientation(_upright_blocks(), orientation)
        with Image.open(io.BytesIO(data)) as image:
            expected = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        _assert_close(_decode_ic(data), expected)

    def test_reads_tag_from_pil_jpeg(self):
        for orientation in range(1, 9):
            data = _jpeg_with_orientation(_upright_blocks(), orientation)
            assert exif_orientation(data) == orientation

    def test_reads_big_endian_tiff_block(self):
        tiff = (
            b"MM\x00\x2a\x00\x00\x00\x08"  # header, IFD0 at offset 8
            + b"\x00\x01"  # one entry
            + b"\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00"
            + b"\x00\x00\x00\x00"
        )
        payload = b"Exif\x00\x00" + tiff
        app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
        assert exif_orientation(b"\xff\xd8" + app1 + b"\xff\xda\x00\x02") == 6

    def test_missing_or_malformed_exif_is_upright(self, jpeg_bytes, png_bytes):
        assert exif_orientation(jpeg_bytes) == 1
        assert exif_orientation(png_bytes) == 1
        assert exif_orientation(b"") == 1
        tagged = _jpeg_with_orientation(_upright_blocks(), 6)
        exif_at = tagged.index(b"Exif")
        # Truncated inside the TIFF block.
        assert exif_orientation(tagged[: exif_at + 12]) == 1
        # Out-of-range tag value.
        payload = b"Exif\x00\x00" + (
            b"II\x2a\x00\x08\x00\x00\x00\x01\x00"
            + b"\x12\x01\x03\x00\x01\x00\x00\x00\x09\x00\x00\x00"
        )
        app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
        assert exif_orientation(b"\xff\xd8" + app1 + b"\xff\xda\x00\x02") == 1


# ---------------------------------------------------------------------------
# EXIF orientation per format — exactly what cv2.imdecode(IMREAD_COLOR) does:
# JPEG, PNG and TIFF are oriented, WebP is not.
# ---------------------------------------------------------------------------

_ORIENTED_FORMATS = {"JPEG": True, "PNG": True, "TIFF": True, "WEBP": False}


def _encode_with_orientation(fmt: str, orientation: int) -> bytes:
    import io

    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = orientation
    options = {"JPEG": {"quality": 95}, "WEBP": {"lossless": True}}.get(fmt, {})
    buffer = io.BytesIO()
    Image.fromarray(_upright_blocks()).save(buffer, fmt, exif=exif.tobytes(), **options)
    data = buffer.getvalue()
    with Image.open(io.BytesIO(data)) as image:
        assert image.getexif().get(0x0112) == orientation, "tag not written"
    return data


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    import zlib

    crc = zlib.crc32(chunk_type + payload)
    return (
        len(payload).to_bytes(4, "big") + chunk_type + payload + crc.to_bytes(4, "big")
    )


def _png_with_exif_chunk(payload: bytes, *, after_idat: bool = False) -> bytes:
    """PNG whose eXIf chunk carries ``payload``, before or after the IDAT."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(_upright_blocks()).save(buffer, "PNG")
    png = buffer.getvalue()
    iend = png.rindex(b"IEND") - 4
    exif_chunk = _png_chunk(b"eXIf", payload)
    if after_idat:
        return png[:iend] + exif_chunk + png[iend:]
    ihdr_end = 8 + 12 + 13
    return png[:ihdr_end] + exif_chunk + png[ihdr_end:]


def _orientation_6_tiff_block() -> bytes:
    from PIL import Image

    exif = Image.Exif()
    exif[0x0112] = 6
    return exif.tobytes()[len(b"Exif\x00\x00") :]


class TestExifOrientationMatchesOpenCV:
    @pytest.mark.parametrize("orientation", [3, 6, 8])
    @pytest.mark.parametrize("fmt", sorted(_ORIENTED_FORMATS))
    def test_decoded_pixels_equal_cv2_imdecode(self, fmt, orientation):
        pytest.importorskip("imagecodecs")
        cv2 = pytest.importorskip("cv2")
        data = _encode_with_orientation(fmt, orientation)
        expected = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        result = make_decoder("imagecodecs")(data)
        if fmt == "JPEG":
            _assert_close(result, expected)
        else:
            np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("orientation", [3, 6, 8])
    @pytest.mark.parametrize("fmt", sorted(_ORIENTED_FORMATS))
    def test_orientation_is_format_aware(self, fmt, orientation):
        data = _encode_with_orientation(fmt, orientation)
        expected = orientation if _ORIENTED_FORMATS[fmt] else 1
        assert exif_orientation(data) == expected
        # Batch decoders hand over memoryviews.
        assert exif_orientation(memoryview(data)) == expected

    @pytest.mark.parametrize("orientation", [3, 6, 8])
    @pytest.mark.parametrize("fmt", sorted(_ORIENTED_FORMATS))
    def test_decoded_dims_match_decoded_pixels(self, fmt, orientation):
        import io

        from PIL import Image

        pytest.importorskip("imagecodecs")
        data = _encode_with_orientation(fmt, orientation)
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        result = make_decoder("imagecodecs")(data)
        assert decoded_dims(data, width, height) == (
            result.shape[1],
            result.shape[0],
        )

    @pytest.mark.parametrize("orientation", range(1, 9))
    @pytest.mark.parametrize("fmt", ["PNG", "TIFF"])
    def test_every_orientation_is_pixel_exact(self, fmt, orientation):
        pytest.importorskip("imagecodecs")
        cv2 = pytest.importorskip("cv2")
        data = _encode_with_orientation(fmt, orientation)
        expected = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(make_decoder("imagecodecs")(data), expected)

    @pytest.mark.parametrize(
        "case,expected",
        [("before_idat", 6), ("after_idat", 6), ("exif_prefix", 1), ("bad_crc", 1)],
    )
    def test_png_exif_chunk_placement_and_validity(self, case, expected):
        """libpng reads eXIf anywhere, but rejects a prefixed or corrupt one."""
        tiff = _orientation_6_tiff_block()
        if case == "exif_prefix":
            data = _png_with_exif_chunk(b"Exif\x00\x00" + tiff)
        else:
            data = _png_with_exif_chunk(tiff, after_idat=case == "after_idat")
        if case == "bad_crc":
            at = data.index(b"eXIf") + 4 + len(tiff)
            data = data[:at] + bytes(4) + data[at + 4 :]
        assert exif_orientation(data) == expected
        cv2 = pytest.importorskip("cv2")
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        stored_shape = decoded.shape[:2] == _upright_blocks().shape[:2]
        assert stored_shape == (expected == 1)

    def test_decoded_dims_ignores_pillow_tiff_swap(self):
        """Pillow reports TIFF size already swapped; the stored size is used."""
        data = _encode_with_orientation("TIFF", 6)
        h, w = _upright_blocks().shape[:2]
        # Same result whether the caller passes the stored or the swapped size.
        assert decoded_dims(data, w, h) == (h, w)
        assert decoded_dims(data, h, w) == (h, w)


# ---------------------------------------------------------------------------
# TIFFs that libtiff converts to RGB itself (CMYK, YCbCr) arrive already
# flipped by libtiff: the decoder must add only what is still missing.
# ---------------------------------------------------------------------------


def _tiff_variant(variant: str, orientation: int) -> bytes:
    import io

    from PIL import Image

    upright = _upright_blocks()
    if variant.startswith("ycbcr"):
        tifffile = pytest.importorskip("tifffile")
        buffer = io.BytesIO()
        tifffile.imwrite(
            buffer,
            upright,
            photometric="ycbcr",
            subsampling=(1, 1),
            compression="zlib" if variant == "ycbcr_deflate" else None,
            extratags=[(0x0112, 3, 1, orientation, True)],
        )
        return buffer.getvalue()
    options = {
        "cmyk": {},
        "cmyk_lzw": {"compression": "tiff_lzw"},
        "cmyk_jpeg": {"compression": "jpeg"},
    }[variant]
    exif = Image.Exif()
    exif[0x0112] = orientation
    buffer = io.BytesIO()
    Image.fromarray(upright).convert("CMYK").save(
        buffer, "TIFF", exif=exif.tobytes(), **options
    )
    return buffer.getvalue()


# CMYK+JPEG is decoded by libjpeg, not by libtiff's RGBA reader: nothing is
# pre-flipped there, and imagecodecs' colours differ from OpenCV's.
_LIBTIFF_RGB_VARIANTS = ["cmyk", "cmyk_lzw", "ycbcr", "ycbcr_deflate"]
_TIFF_VARIANTS = _LIBTIFF_RGB_VARIANTS + ["cmyk_jpeg"]


class TestTiffPreOrientedByLibtiff:
    @pytest.mark.parametrize("orientation", range(1, 9))
    @pytest.mark.parametrize("variant", _LIBTIFF_RGB_VARIANTS)
    def test_pixel_exact_vs_cv2_imdecode(self, variant, orientation):
        pytest.importorskip("imagecodecs")
        cv2 = pytest.importorskip("cv2")
        data = _tiff_variant(variant, orientation)
        expected = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(make_decoder("imagecodecs")(data), expected)

    @pytest.mark.parametrize("orientation", range(1, 9))
    @pytest.mark.parametrize("variant", _TIFF_VARIANTS)
    def test_output_is_stored_image_oriented_once(self, variant, orientation):
        pytest.importorskip("imagecodecs")
        decode = make_decoder("imagecodecs")
        stored = decode(_tiff_variant(variant, 1))
        expected = apply_exif_orientation(stored, orientation)
        result = decode(_tiff_variant(variant, orientation))
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("orientation", range(1, 9))
    @pytest.mark.parametrize("variant", _TIFF_VARIANTS)
    def test_decoded_dims_match_decoded_pixels(self, variant, orientation):
        import io

        from PIL import Image

        pytest.importorskip("imagecodecs")
        data = _tiff_variant(variant, orientation)
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        result = make_decoder("imagecodecs")(data)
        assert decoded_dims(data, width, height) == (
            result.shape[1],
            result.shape[0],
        )


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


# ---------------------------------------------------------------------------
# Truncated JPEG — cv2.imdecode refuses one, the decoder must too
# ---------------------------------------------------------------------------


def _complete_jpeg() -> bytes:
    imagecodecs = pytest.importorskip("imagecodecs")
    return bytes(imagecodecs.jpeg_encode(_make_rgb_array(64, 64)))


def _scan_start(data: bytes) -> int:
    return data.index(b"\xff\xda")


def _with_thumbnail_segment(data: bytes) -> bytes:
    """Main JPEG preceded by an APP1 Exif segment embedding a complete JPEG."""
    imagecodecs = pytest.importorskip("imagecodecs")
    thumbnail = bytes(imagecodecs.jpeg_encode(_make_rgb_array(8, 8)))
    payload = b"Exif\x00\x00II\x2a\x00\x08\x00\x00\x00\x00\x00" + thumbnail
    app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    return data[:2] + app1 + data[2:]


def _progressive_jpeg() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(_make_rgb_array(64, 64)).save(
        buffer, format="JPEG", progressive=True
    )
    return buffer.getvalue()


def _restart_marker_jpeg() -> bytes:
    cv2 = pytest.importorskip("cv2")
    _, encoded = cv2.imencode(
        ".jpg", _make_rgb_array(64, 64), [cv2.IMWRITE_JPEG_RST_INTERVAL, 1]
    )
    return encoded.tobytes()


def _segment(marker: bytes, payload: bytes) -> bytes:
    return marker + (len(payload) + 2).to_bytes(2, "big") + payload


def _scan_cases() -> dict[str, tuple[bytes, bool]]:
    progressive = _progressive_jpeg()
    scans = [
        i for i in range(len(progressive)) if progressive[i : i + 2] == b"\xff\xda"
    ]
    restart = _restart_marker_jpeg()
    app1_with_eoi = _segment(b"\xff\xe1", b"Exif\x00\x00junk\xff\xd9junk")
    comment_with_eoi = _segment(b"\xff\xfe", b"note \xff\xd9 note")
    cases = {
        "progressive_complete": (progressive, False),
        "progressive_missing_eoi": (progressive[:-2], True),
        "progressive_cut_before_second_scan": (progressive[: scans[1]], True),
        "progressive_cut_in_second_scan": (progressive[: scans[1] + 30], True),
        "missing_eoi_then_app1_holding_eoi": (progressive[:-2] + app1_with_eoi, True),
        "missing_eoi_then_comment_holding_eoi": (
            progressive[:-2] + comment_with_eoi,
            True,
        ),
        "app1_after_scan_then_eoi": (
            progressive[:-2] + app1_with_eoi + b"\xff\xd9",
            False,
        ),
        "restart_markers_complete": (restart, False),
        "restart_markers_missing_eoi": (restart[:-2], True),
        "fill_bytes_before_eoi": (progressive[:-2] + b"\xff\xff\xff\xd9", False),
        "progressive_trailing_bytes_after_eoi": (progressive + b"\xff\xd8junk", False),
    }
    return cases


def _truncation_cases() -> dict[str, tuple[bytes, bool]]:
    data = _complete_jpeg()
    scan = _scan_start(data)
    with_thumbnail = _with_thumbnail_segment(data)
    cases = {
        "complete": (data, False),
        "cut_in_scan": (data[: scan + (len(data) - scan) // 2], True),
        "cut_after_scan_header": (data[: scan + 20], True),
        "cut_before_scan": (data[:scan], True),
        "missing_eoi": (data[:-2], True),
        "trailing_bytes_after_eoi": (data + b"\x00trailing", False),
        "thumbnail_complete": (with_thumbnail, False),
        "thumbnail_main_missing_eoi": (with_thumbnail[:-2], True),
        "thumbnail_main_cut_in_scan": (
            with_thumbnail[: len(with_thumbnail) - 40],
            True,
        ),
        **_scan_cases(),
    }
    return cases


class TestJpegIsTruncated:
    @pytest.mark.parametrize("case", sorted(_truncation_cases()))
    def test_detects_truncation(self, case):
        from inference_model_manager.backends.decode import jpeg_is_truncated

        data, truncated = _truncation_cases()[case]
        assert jpeg_is_truncated(data) is truncated
        assert jpeg_is_truncated(memoryview(data)) is truncated

    @pytest.mark.parametrize("case", sorted(_truncation_cases()))
    def test_agrees_with_cv2_imdecode(self, case):
        from inference_model_manager.backends.decode import jpeg_is_truncated

        cv2 = pytest.importorskip("cv2")
        data, _ = _truncation_cases()[case]
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert jpeg_is_truncated(data) is (decoded is None)

    def test_non_jpeg_is_never_truncated(self, png_bytes, webp_bytes):
        from inference_model_manager.backends.decode import jpeg_is_truncated

        assert jpeg_is_truncated(png_bytes) is False
        assert jpeg_is_truncated(png_bytes[: len(png_bytes) // 2]) is False
        assert jpeg_is_truncated(webp_bytes) is False
        assert jpeg_is_truncated(b"") is False

    @pytest.mark.parametrize("case", ["cut_in_scan", "missing_eoi"])
    def test_decoder_rejects_truncated_jpeg_as_input_error(self, case):
        data, _ = _truncation_cases()[case]
        with pytest.raises(ValueError, match="truncated"):
            make_decoder("imagecodecs")(data)

    def test_plugin_decoder_never_sees_truncated_jpeg(self):
        import inference_model_manager.backends.decode as decode_mod

        data = _complete_jpeg()

        def _explode(_data):
            raise AssertionError("plugin decode must not run for a truncated JPEG")

        decode_mod.register_decoder("fake-truncation", lambda device: _explode)
        try:
            decode = decode_mod.make_decoder("fake-truncation", device="cpu")
            with pytest.raises(ValueError, match="truncated"):
                decode(data[:-2])
        finally:
            decode_mod.DECODER_FACTORIES.pop("fake-truncation", None)

    @pytest.mark.parametrize(
        "case,truncated",
        [
            ("before_restart_marker", False),
            ("before_end_of_data", True),
            ("before_eoi", False),
            ("before_comment_holding_eoi", True),
            ("before_comment_then_eoi", False),
        ],
    )
    def test_long_fill_runs_are_scanned_in_linear_time(self, case, truncated):
        import time

        from inference_model_manager.backends.decode import jpeg_is_truncated

        fill = b"\xff" * (64 * 1024)
        progressive = _progressive_jpeg()[:-2]
        comment = _segment(b"\xff\xfe", b"note \xff\xd9 note")
        restart = _restart_marker_jpeg()
        first_restart = restart.index(b"\xff\xd0")
        data = {
            "before_restart_marker": (
                restart[:first_restart] + fill + restart[first_restart:]
            ),
            "before_end_of_data": progressive + fill,
            "before_eoi": progressive + fill + b"\xd9",
            "before_comment_holding_eoi": progressive + fill + comment,
            "before_comment_then_eoi": progressive + fill + comment + b"\xff\xd9",
        }[case]

        started = time.perf_counter()
        result = jpeg_is_truncated(data)
        elapsed = time.perf_counter() - started

        assert result is truncated
        assert elapsed < 0.1, f"{elapsed:.3f}s"

    @pytest.mark.parametrize("complete", [False, True])
    def test_large_view_is_scanned_without_copying(self, complete):
        import tracemalloc

        from inference_model_manager.backends.decode import jpeg_is_truncated

        size = 16 * 1024 * 1024
        head = _complete_jpeg()[:-2]
        buffer = bytearray(head + bytes(size) + (b"\xff\xd9" if complete else b""))
        view = memoryview(buffer)

        tracemalloc.start()
        try:
            truncated = jpeg_is_truncated(view)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert truncated is not complete
        assert peak < 1024 * 1024

    def test_complete_jpeg_with_trailing_bytes_still_decodes(self):
        data = _complete_jpeg() + b"\x00trailing"
        assert make_decoder("imagecodecs")(data).shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# Resetting the EXIF orientation — cv2.IMREAD_IGNORE_ORIENTATION as bytes
# ---------------------------------------------------------------------------


def _orientation_tiff_block(orientation: int, byte_order: str) -> bytes:
    order = "little" if byte_order == "II" else "big"
    return (
        byte_order.encode()
        + (42).to_bytes(2, order)
        + (8).to_bytes(4, order)
        + (1).to_bytes(2, order)
        + (0x0112).to_bytes(2, order)
        + (3).to_bytes(2, order)
        + (1).to_bytes(4, order)
        + orientation.to_bytes(2, order)
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"
    )


def _jpeg_with_tiff_block(tiff: bytes) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(_upright_blocks()).save(buffer, format="JPEG", quality=95)
    data = buffer.getvalue()
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload
    return data[:2] + app1 + data[2:]


def _tiff_with_orientation(orientation: int, byte_order: str) -> bytes:
    import io

    tifffile = pytest.importorskip("tifffile")
    buffer = io.BytesIO()
    tifffile.imwrite(
        buffer,
        _upright_blocks(),
        byteorder="<" if byte_order == "II" else ">",
        extratags=[(0x0112, 3, 1, orientation, True)],
    )
    return buffer.getvalue()


def _oriented(fmt: str, orientation: int, byte_order: str) -> bytes:
    if fmt == "JPEG":
        return _jpeg_with_tiff_block(_orientation_tiff_block(orientation, byte_order))
    if fmt == "PNG":
        return _png_with_exif_chunk(_orientation_tiff_block(orientation, byte_order))
    return _tiff_with_orientation(orientation, byte_order)


def _png_crcs_valid(data: bytes) -> bool:
    import zlib

    offset = 8
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 8 + length
        crc = int.from_bytes(data[end : end + 4], "big")
        if zlib.crc32(data[offset + 4 : end]) != crc:
            return False
        offset = end + 4
    return True


class TestResetExifOrientation:
    @pytest.mark.parametrize("byte_order", ["II", "MM"])
    @pytest.mark.parametrize("orientation", [6, 8])
    @pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
    def test_result_is_upright_tagged_and_decodes_as_stored(
        self, fmt, orientation, byte_order
    ):
        from inference_model_manager.backends.decode import reset_exif_orientation

        cv2 = pytest.importorskip("cv2")
        data = _oriented(fmt, orientation, byte_order)
        assert exif_orientation(data) == orientation

        result = reset_exif_orientation(data)

        assert exif_orientation(result) == 1
        decoded = make_decoder("imagecodecs")(result)
        as_stored = cv2.imdecode(
            np.frombuffer(data, np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        if fmt == "JPEG":
            _assert_close(decoded, as_stored)
        else:
            np.testing.assert_array_equal(decoded, as_stored)
            np.testing.assert_array_equal(decoded, _upright_blocks()[..., ::-1])

    @pytest.mark.parametrize("byte_order", ["II", "MM"])
    @pytest.mark.parametrize("orientation", [6, 8])
    def test_tiff_stays_oriented_like_cv2(self, orientation, byte_order):
        from inference_model_manager.backends.decode import reset_exif_orientation

        cv2 = pytest.importorskip("cv2")
        data = _oriented("TIFF", orientation, byte_order)

        result = reset_exif_orientation(data)

        assert result == data
        as_stored = cv2.imdecode(
            np.frombuffer(data, np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
        )
        np.testing.assert_array_equal(make_decoder("imagecodecs")(result), as_stored)

    @pytest.mark.parametrize("byte_order", ["II", "MM"])
    @pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
    def test_only_the_tag_value_changes(self, fmt, byte_order):
        from inference_model_manager.backends.decode import reset_exif_orientation

        data = _oriented(fmt, 6, byte_order)

        result = reset_exif_orientation(data)

        assert len(result) == len(data)
        changed = [i for i, (a, b) in enumerate(zip(data, result)) if a != b]
        assert len(changed) >= 1
        if fmt == "PNG":
            assert _png_crcs_valid(result)
            tag_byte, *crc_bytes = changed
            crc_at = data.index(b"eXIf") + 4 + len(_orientation_tiff_block(6, "II"))
            assert all(crc_at <= i < crc_at + 4 for i in crc_bytes)
            changed = [tag_byte]
        assert len(changed) == 1
        assert data[changed[0]] == 6 and result[changed[0]] == 1

    def test_pil_written_jpeg_exif_is_reset(self):
        from inference_model_manager.backends.decode import reset_exif_orientation

        data = _jpeg_with_orientation(_upright_blocks(), 6)

        result = reset_exif_orientation(data)

        assert exif_orientation(result) == 1
        assert sum(a != b for a, b in zip(data, result)) == 1

    def test_unoriented_inputs_are_returned_unchanged(self, jpeg_bytes, png_bytes):
        from inference_model_manager.backends.decode import reset_exif_orientation

        tiff = _orientation_tiff_block(6, "II")
        bad_crc = _png_with_exif_chunk(tiff)
        at = bad_crc.index(b"eXIf") + 4 + len(tiff)
        bad_crc = bad_crc[:at] + bytes(4) + bad_crc[at + 4 :]
        unchanged = [
            jpeg_bytes,
            png_bytes,
            b"",
            b"not an image",
            _encode_with_orientation("WEBP", 6),
            _jpeg_with_orientation(_upright_blocks(), 1),
            _png_with_exif_chunk(b"Exif\x00\x00" + tiff),
            bad_crc,
            _jpeg_with_tiff_block(_orientation_tiff_block(9, "MM")),
        ]
        for data in unchanged:
            assert reset_exif_orientation(data) == data
            assert reset_exif_orientation(memoryview(data)) == data
