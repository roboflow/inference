"""Image decoder factory and registry.

Single-image decoder: make_decoder(name, device) → (bytes) -> image

Decoders:
  imagecodecs  — CPU, RGB HWC uint8 numpy (replaces cv2)
"""

from __future__ import annotations

import functools
import re
import threading
import zlib
from typing import Any, Callable, Iterator

import imagecodecs
import numpy as np

from inference_model_manager import configuration as cfg


def max_decoded_pixels() -> int:
    """Decoded-pixel ceiling, 0 when the gate is disabled."""
    return max(0, int(cfg.INFERENCE_DECODE_MAX_MEGAPIXELS * 1_000_000))


def header_pixels(data: bytes | memoryview) -> int:
    """Pixel count read from the container header, or 0 when unknown.

    JPEG and PNG carry their dimensions in a fixed header, so a bomb in those
    formats is rejected without allocating anything. Other formats fall back to
    the post-decode backstop.
    """
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        if bytes(data[12:16]) != b"IHDR":
            return 0
        width = int.from_bytes(bytes(data[16:20]), "big")
        height = int.from_bytes(bytes(data[20:24]), "big")
        return width * height
    if len(data) >= 4 and data[:3] == b"\xff\xd8\xff":
        return _jpeg_header_pixels(data)
    return 0


def _jpeg_header_pixels(data: bytes | memoryview) -> int:
    """Width*height from the JPEG SOF segment, or 0 if not found."""
    size = len(data)
    offset = 2
    while offset + 1 < size:
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        # Padding fill byte, or a standalone marker with no length field.
        if marker == 0xFF:
            offset += 1
            continue
        if marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if offset + 3 >= size:
            return 0
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if offset + 8 >= size:
                return 0
            height = int.from_bytes(data[offset + 5 : offset + 7], "big")
            width = int.from_bytes(data[offset + 7 : offset + 9], "big")
            return width * height
        segment_length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if segment_length < 2:
            return 0
        offset += 2 + segment_length
    return 0


def jpeg_is_truncated(data: bytes | memoryview) -> bool:
    """Tell whether ``data`` is a JPEG cut short, which cv2.imdecode refuses.

    A JPEG is complete only when its end-of-image marker follows the first
    start-of-scan segment. The stream is walked marker by marker: segments
    carry their length and are skipped whole, so an end-of-image marker
    inside metadata (an EXIF thumbnail, a comment) does not count; scan data
    byte-stuffs 0xFF and is crossed up to the next real marker. Bytes after
    the end-of-image marker are allowed. The buffer is never copied.

    Args:
        data: Encoded image bytes.

    Returns:
        True for a JPEG whose stream ends before its end-of-image marker,
        False for a complete JPEG and for every other format.
    """
    if bytes(data[:3]) != b"\xff\xd8\xff":
        return False

    scan_data = _jpeg_scan_data_offset(data)
    if scan_data is None:
        return True

    truncated = not _jpeg_reaches_eoi(data, scan_data)

    return truncated


# Last 0xFF before a real marker (not 00, RSTn or fill); `\xff+` is quadratic.
_SCAN_MARKER = re.compile(rb"\xff[^\x00\xd0-\xd7\xff]")


def _jpeg_reaches_eoi(data: bytes | memoryview, offset: int) -> bool:
    """Whether the stream from scan data at ``offset`` reaches its EOI marker."""
    size = len(data)
    while True:
        match = _SCAN_MARKER.search(data, offset)
        if match is None:
            return False
        offset = match.end()
        marker = data[offset - 1]
        if marker == 0xD9:
            return True
        if marker in (0x01, 0xD8):
            continue
        if offset + 2 > size:
            return False
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2:
            return False
        offset += segment_length


def _jpeg_scan_data_offset(data: bytes | memoryview) -> int | None:
    """Offset right after the first SOS segment header, None if there is none."""
    size = len(data)
    offset = 2
    while offset + 3 < size:
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker == 0xD9:
            return None
        segment_length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if segment_length < 2:
            return None
        if marker == 0xDA:
            return offset + 2 + segment_length
        offset += 2 + segment_length
    return None


_EXIF_ORIENTATION_TAG = 0x0112
_TIFF_MAGICS = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
# Orientations 5-8 store the image transposed: decoded width and height swap.
EXIF_TRANSPOSING_ORIENTATIONS = frozenset((5, 6, 7, 8))


def exif_orientation(data: bytes | memoryview) -> int:
    """EXIF Orientation (1-8) the decoder applies to ``data``, 1 when none.

    Follows cv2.imdecode(IMREAD_COLOR), the previous server's decoder, format
    by format: JPEG (APP1 Exif block), PNG (eXIf chunk) and TIFF (IFD0 tag)
    are oriented; WebP and every other format are not, even when they carry
    the tag. Only headers and metadata are read. Malformed EXIF is ignored,
    as OpenCV does.
    """
    span = _orientation_tiff_span(data)
    if span is None:
        return 1
    start, end = span
    return _tiff_orientation(bytes(data[start:end])) or 1


def reset_exif_orientation(data: bytes | memoryview) -> bytes:
    """Rewrite the EXIF Orientation the decoder applies to ``data`` as 1.

    Decoding the result yields what cv2.imdecode with IMREAD_IGNORE_ORIENTATION
    yields from ``data``: JPEG and PNG as stored, TIFF still oriented, because
    OpenCV's TIFF reader orients regardless of that flag. Only the tag value
    ``exif_orientation`` reads is patched, in its own byte order and width;
    the CRC of a PNG eXIf chunk is recomputed. Everything else is copied.

    Args:
        data: Encoded image bytes.

    Returns:
        The patched copy, or the bytes unchanged when there is nothing to
        reset.
    """
    raw = bytes(data)
    if raw[:4] in _TIFF_MAGICS:
        return raw

    patched = _with_orientation_1(raw)

    return patched


def _with_orientation_1(raw: bytes) -> bytes:
    """Copy of ``raw`` with the Orientation ``exif_orientation`` reads set to 1."""
    span = _orientation_tiff_span(raw)
    if span is None:
        return raw
    start, end = span
    tiff = raw[start:end]
    field = _tiff_ifd0_fields(tiff, (_EXIF_ORIENTATION_TAG,)).get(_EXIF_ORIENTATION_TAG)
    if field is None or _tiff_orientation(tiff) is None:
        return raw

    value_offset, value_size, order, _ = field
    patched = bytearray(raw)
    value_start = start + value_offset
    patched[value_start : value_start + value_size] = (1).to_bytes(value_size, order)
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        crc = zlib.crc32(patched[start - 4 : end])
        patched[end : end + 4] = crc.to_bytes(4, "big")

    return bytes(patched)


def _orientation_tiff_span(data: bytes | memoryview) -> tuple[int, int] | None:
    """(start, end) of the TIFF block whose Orientation the decoder honours."""
    head = bytes(data[:8])
    if head[:3] == b"\xff\xd8\xff":
        return _jpeg_exif_span(data)
    if head == b"\x89PNG\r\n\x1a\n":
        return _png_exif_span(data)
    if head[:4] in _TIFF_MAGICS:
        return 0, len(data)
    return None


def decoded_dims(data: bytes | memoryview, width: int, height: int) -> tuple[int, int]:
    """(width, height) of the image the decoder returns for ``data``.

    ``width`` and ``height`` are the size a header reader (e.g. Pillow)
    reported. TIFF re-reads its stored size from IFD0 instead, because Pillow
    already swaps the reported TIFF size for orientations 5-8. The swap below
    then uses the same rule as the decoder, so size and pixels always agree:
    the decoder's output is transposed exactly for orientations 5-8 of the
    stored image.
    """
    if bytes(data[:4]) in _TIFF_MAGICS:
        stored = _tiff_ifd0(data, (_TIFF_WIDTH_TAG, _TIFF_HEIGHT_TAG))
        if _TIFF_WIDTH_TAG in stored and _TIFF_HEIGHT_TAG in stored:
            width, height = stored[_TIFF_WIDTH_TAG], stored[_TIFF_HEIGHT_TAG]
    if exif_orientation(data) in EXIF_TRANSPOSING_ORIENTATIONS:
        return height, width
    return width, height


def _jpeg_exif_span(data: bytes | memoryview) -> tuple[int, int] | None:
    """TIFF block of the first APP1 Exif segment with a valid Orientation,
    walking segments up to the scan."""
    size = len(data)
    offset = 2
    while offset + 3 < size:
        if data[offset] != 0xFF:
            return None
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        # Start of scan / end of image: the metadata segments are behind us.
        if marker in (0xDA, 0xD9):
            return None
        segment_length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if segment_length < 2:
            return None
        payload_start = offset + 4
        payload_end = min(offset + 2 + segment_length, size)
        if marker == 0xE1 and bytes(data[payload_start : payload_start + 6]) == (
            b"Exif\x00\x00"
        ):
            span = (payload_start + 6, payload_end)
            if _tiff_orientation(bytes(data[span[0] : span[1]])) is not None:
                return span
        offset += 2 + segment_length
    return None


def _png_exif_span(data: bytes | memoryview) -> tuple[int, int] | None:
    """TIFF block of the first eXIf chunk, None when absent or invalid.

    libpng (behind cv2.imdecode) honours eXIf before or after the image data,
    requires the payload to start with the TIFF byte-order mark (no
    "Exif\\0\\0" prefix) and drops the chunk when its CRC is wrong.
    """
    size = len(data)
    offset = 8
    while offset + 8 <= size:
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = bytes(data[offset + 4 : offset + 8])
        payload_end = offset + 8 + length
        if chunk_type == b"IEND" or payload_end + 4 > size:
            return None
        if chunk_type == b"eXIf":
            crc = int.from_bytes(data[payload_end : payload_end + 4], "big")
            if zlib.crc32(data[offset + 4 : payload_end]) != crc:
                return None
            return offset + 8, payload_end
        offset = payload_end + 4
    return None


_TIFF_WIDTH_TAG = 0x0100
_TIFF_HEIGHT_TAG = 0x0101
_TIFF_PHOTOMETRIC_TAG = 0x0106
_TIFF_PHOTOMETRIC_RGB = 2
_TIFF_ROWS_PER_STRIP_TAG = 0x0116
_TIFF_BITS_PER_SAMPLE_TAG = 0x0102
_TIFF_SAMPLES_PER_PIXEL_TAG = 0x0115
_TIFF_PLANAR_CONFIG_TAG = 0x011C
_TIFF_TILE_WIDTH_TAG = 0x0142
_TIFF_TILE_LENGTH_TAG = 0x0143
_TIFF_EXTRA_SAMPLES_TAG = 0x0152
_TIFF_PLANAR_SEPARATE = 2
_TIFF_EXTRA_SAMPLE_ASSOCIATED_ALPHA = 1
_TIFF_VALUE_SIZES = {3: 2, 4: 4, 16: 8}


def _tiff_ifd0(tiff: bytes | memoryview, tags: tuple[int, ...]) -> dict[int, int]:
    """SHORT/LONG values of ``tags`` in IFD0 of a TIFF block; malformed → {}."""
    fields = _tiff_ifd0_fields(tiff, tags)
    values = {
        tag: int.from_bytes(tiff[offset : offset + size], order)
        for tag, (offset, size, order, _) in fields.items()
    }

    return values


def _tiff_ifd0_fields(
    tiff: bytes | memoryview, tags: tuple[int, ...]
) -> dict[int, tuple[int, int, str, int]]:
    """(offset, size, byte order, count) of the first SHORT/LONG/LONG8 value of
    ``tags`` in IFD0 of a classic TIFF or BigTIFF block."""
    if len(tiff) < 8:
        return {}
    if tiff[:2] == b"II":
        order = "little"
    elif tiff[:2] == b"MM":
        order = "big"
    else:
        return {}
    version = int.from_bytes(tiff[2:4], order)
    if version == 42:
        ifd = int.from_bytes(tiff[4:8], order)
        count_size, entry_size, inline = 2, 12, 4
    elif version == 43 and len(tiff) >= 16 and int.from_bytes(tiff[4:6], order) == 8:
        ifd = int.from_bytes(tiff[8:16], order)
        count_size, entry_size, inline = 8, 20, 8
    else:
        return {}
    if ifd + count_size > len(tiff):
        return {}
    entries = int.from_bytes(tiff[ifd : ifd + count_size], order)
    found: dict[int, tuple[int, int, str, int]] = {}
    for index in range(entries):
        entry = ifd + count_size + entry_size * index
        if entry + entry_size > len(tiff):
            return {}
        tag = int.from_bytes(tiff[entry : entry + 2], order)
        if tag not in tags:
            continue
        value_type = int.from_bytes(tiff[entry + 2 : entry + 4], order)
        size = _TIFF_VALUE_SIZES.get(value_type)
        if size is None:
            continue
        count = int.from_bytes(tiff[entry + 4 : entry + 4 + inline], order)
        value = entry + 4 + inline
        if count * size > inline:
            value = int.from_bytes(tiff[value : value + inline], order)
            if value + size > len(tiff):
                continue
        found[tag] = (value, size, order, count)
    return found


def _tiff_orientation(tiff: bytes | memoryview) -> int | None:
    """Orientation tag from IFD0 of a TIFF block, None when absent or invalid."""
    value = _tiff_ifd0(tiff, (_EXIF_ORIENTATION_TAG,)).get(_EXIF_ORIENTATION_TAG)
    return value if value is not None and 1 <= value <= 8 else None


def apply_exif_orientation(image: np.ndarray, orientation: int) -> np.ndarray:
    """Rotate/flip an HWC array to upright, matching PIL's exif_transpose.

    Returns a view; the caller makes it contiguous.
    """
    if orientation == 2:
        return image[:, ::-1]
    if orientation == 3:
        return image[::-1, ::-1]
    if orientation == 4:
        return image[::-1]
    if orientation == 5:
        return image.swapaxes(0, 1)
    if orientation == 6:
        return image.swapaxes(0, 1)[:, ::-1]
    if orientation == 7:
        return image.swapaxes(0, 1)[::-1, ::-1]
    if orientation == 8:
        return image.swapaxes(0, 1)[::-1]
    return image


def _reject_oversized(pixels: int, limit: int, stage: str) -> None:
    if limit and pixels > limit:
        raise ValueError(
            f"image is {pixels / 1e6:.1f} megapixels ({stage}), over the "
            f"{limit / 1e6:.0f} megapixel decode limit"
        )


def _guard_encoded(data: bytes | memoryview) -> int:
    """Reject a decompression bomb from its header. Returns the active limit."""
    limit = max_decoded_pixels()
    if limit:
        _reject_oversized(header_pixels(data), limit, "header")
    return limit


def decoded_pixels(image: Any) -> int:
    """Pixel count of a decoded image, 0 when it has no readable shape.

    The two largest dimensions are the spatial ones whether the decoder emits
    HWC ndarrays or CHW device tensors.
    """
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        return 0
    dims = sorted((int(d) for d in shape), reverse=True)
    return dims[0] * dims[1]


def _guard_decoded(image: Any, limit: int) -> None:
    """Backstop for formats whose header we cannot read cheaply."""
    if limit:
        _reject_oversized(decoded_pixels(image), limit, "decoded")


def guard_decoder(decode: Callable[[bytes], Any]) -> Callable[[bytes], Any]:
    """Gate a decoder with the decompression-bomb limits of every backend decoder.

    Args:
        decode: Function turning encoded image bytes into a decoded image.

    Returns:
        The same decoder, refusing images over the decode limit before and
        after decoding, and truncated JPEGs, exactly as the backends do.
    """
    return _guarded(decode)


def _guarded(decode: Callable[[bytes], Any]) -> Callable[[bytes], Any]:
    """Wrap a decoder with the decompression-bomb and truncated-JPEG gates.

    Every decoder reaching a backend goes through here, so a plugin decoder
    registered through the entry-point group is gated exactly like the
    built-in one.
    """

    def _decode(data: bytes) -> Any:
        limit = _guard_encoded(data)
        if jpeg_is_truncated(data):
            raise ValueError(
                "JPEG data is truncated: no end-of-image marker after the scan"
            )
        image = decode(data)
        _guard_decoded(image, limit)
        return image

    return _decode


def _select_codec(head: bytes) -> str | None:
    """Map header magic bytes to an imagecodecs codec name, or None if unknown.

    Dispatch explicitly instead of imagecodecs.imread(): imread() probes every
    registered codec when the format is unknown, and the bundled OpenEXR codec
    writes "EXR_ERR_FILE_BAD_HEADER" to C stderr on every non-EXR image before
    the real codec succeeds. Explicit dispatch never touches the EXR codec.
    Unrecognised headers return None and fall back to imread() probing.
    """
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:3] == b"GIF":
        return "gif"
    if head[:4] in _TIFF_MAGICS:
        return "tiff"
    if head[:2] == b"BM":
        return "bmp"
    if head[:2] == b"\xff\x4f" or (
        head[:4] == b"\x00\x00\x00\x0c" and head[4:8] == b"jP  "
    ):
        return "jpeg2k"
    if head[4:8] == b"ftyp" and head[8:12] in (b"avif", b"avis"):
        return "avif"
    return None


def _to_rgb_hwc(img: np.ndarray) -> np.ndarray:
    """Normalize a decoded array to (H, W, 3): grayscale → replicate,
    gray+alpha → replicate gray, RGBA → drop alpha."""
    if img.ndim == 2:
        return np.stack((img, img, img), axis=-1)
    channels = img.shape[2]
    if channels == 1:
        return np.repeat(img, 3, axis=2)
    if channels == 2:
        return np.repeat(img[:, :, :1], 3, axis=2)
    if channels == 4:
        return img[:, :, :3]
    return img


_OPENCV_TIFF_SCANLINE_BYTES = (1 << 30) * 95 // 100


def _opencv_reads_tiff_scanlines(tiff: bytes | memoryview) -> bool:
    """Whether OpenCV 4.8-4.12 decodes this TIFF to 8 bits scanline by scanline.

    Mirrors the ``dst_bpp == 8`` branch of ``TiffDecoder::readData`` in
    grfmt_tiff.cpp: a large, untiled, single-strip, contiguous 8/16-bit gray or
    RGB image (RGBA only with associated alpha) is read scanline by scanline,
    which keeps the high byte of 16-bit samples; everything else goes through
    libtiff's RGBA reader.
    """
    fields = _tiff_ifd0_fields(
        tiff,
        (
            _TIFF_WIDTH_TAG,
            _TIFF_HEIGHT_TAG,
            _TIFF_BITS_PER_SAMPLE_TAG,
            _TIFF_PHOTOMETRIC_TAG,
            _TIFF_SAMPLES_PER_PIXEL_TAG,
            _TIFF_ROWS_PER_STRIP_TAG,
            _TIFF_PLANAR_CONFIG_TAG,
            _TIFF_TILE_WIDTH_TAG,
            _TIFF_TILE_LENGTH_TAG,
            _TIFF_EXTRA_SAMPLES_TAG,
        ),
    )
    tags = {
        tag: int.from_bytes(tiff[offset : offset + size], order)
        for tag, (offset, size, order, _) in fields.items()
    }
    if _TIFF_TILE_WIDTH_TAG in tags or _TIFF_TILE_LENGTH_TAG in tags:
        return False

    width = tags.get(_TIFF_WIDTH_TAG, 0)
    height = tags.get(_TIFF_HEIGHT_TAG, 0)
    rows = tags.get(_TIFF_ROWS_PER_STRIP_TAG, 0)
    if rows in (0, 0xFFFFFFFF):
        rows = height
    if width * rows * 4 < _OPENCV_TIFF_SCANLINE_BYTES:
        return False

    samples = tags.get(_TIFF_SAMPLES_PER_PIXEL_TAG, 1)
    scanline = (
        samples in (1, 3, 4)
        and tags.get(_TIFF_BITS_PER_SAMPLE_TAG, 1) in (8, 16)
        and rows == height
        and tags.get(_TIFF_PHOTOMETRIC_TAG) in (0, 1, _TIFF_PHOTOMETRIC_RGB)
        and tags.get(_TIFF_PLANAR_CONFIG_TAG, 1) != _TIFF_PLANAR_SEPARATE
    )
    if scanline and samples == 4:
        extra = fields.get(_TIFF_EXTRA_SAMPLES_TAG)
        scanline = (
            extra is not None
            and extra[3] == 1
            and tags[_TIFF_EXTRA_SAMPLES_TAG] == _TIFF_EXTRA_SAMPLE_ASSOCIATED_ALPHA
        )

    return scanline


def _boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """Yield (type, payload start, payload end) of the JP2 / ISO-BMFF boxes in
    ``data[start:end]``, stopping at the first malformed box."""
    offset = start
    while offset + 8 <= end:
        size = int.from_bytes(data[offset : offset + 4], "big")
        header = 8
        if size == 1:
            if offset + 16 > end:
                return
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or offset + size > end:
            return
        yield bytes(data[offset + 4 : offset + 8]), offset + header, offset + size
        offset += size


def _child_box(data: bytes, start: int, end: int, kind: bytes) -> tuple[int, int]:
    """(payload start, payload end) of the first ``kind`` box, (0, 0) if none."""
    for box_kind, payload_start, payload_end in _boxes(data, start, end):
        if box_kind == kind:
            return payload_start, payload_end
    return 0, 0


def _jpeg2k_precision(data: bytes) -> int:
    """Highest component precision in the SIZ segment, 16 when unreadable."""
    codestream = 0
    if data[:4] != b"\xff\x4f\xff\x51":
        codestream, _ = _child_box(data, 0, len(data), b"jp2c")
    if data[codestream : codestream + 4] != b"\xff\x4f\xff\x51":
        return 16
    siz = codestream + 2
    if siz + 40 > len(data):
        return 16
    components = int.from_bytes(data[siz + 38 : siz + 40], "big")
    sizes = data[siz + 40 : siz + 40 + 3 * components : 3]
    if not sizes:
        return 16
    precision = max((size & 0x7F) + 1 for size in sizes)

    return precision


def _avif_depth(data: bytes) -> int:
    """Bit depth from the av1C property in meta/iprp/ipco, 16 when unreadable."""
    meta_start, meta_end = _child_box(data, 0, len(data), b"meta")
    # meta is a FullBox: 4 bytes of version and flags precede its children.
    start, end = _child_box(data, meta_start + 4, meta_end, b"iprp")
    start, end = _child_box(data, start, end, b"ipco")
    start, end = _child_box(data, start, end, b"av1C")
    if end - start < 3:
        return 16
    flags = data[start + 2]
    if not flags & 0x40:
        return 8
    depth = 12 if flags & 0x20 else 10

    return depth


@functools.lru_cache(maxsize=None)
def _reduction_table(rule: str, bits: int) -> np.ndarray:
    values = np.arange(1 << 16)
    if rule == "shift":
        table = values >> max(bits - 8, 0)
    elif rule == "round":
        table = np.clip(np.rint(values / (1 << (bits - 8))), 0, 255)
    else:
        table = np.clip(np.rint(values * 255 / ((1 << bits) - 1)), 0, 255)
    return table.astype(np.uint8)


_REDUCTION_CHUNK_ELEMENTS = 1 << 16


def _reduce(img: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Look ``img`` up in ``table`` a few rows at a time, so temporaries stay small."""
    reduced = np.empty(img.shape, dtype=np.uint8)
    row_elements = int(np.prod(img.shape[1:]))
    rows = max(1, _REDUCTION_CHUNK_ELEMENTS // max(1, row_elements))
    for start in range(0, img.shape[0], rows):
        reduced[start : start + rows] = table[img[start : start + rows]]
    return reduced


def _opencv_depth_rule(
    img: np.ndarray, codec: str | None, data: bytes
) -> tuple[str, int]:
    """(rule, bits) cv2.imdecode(IMREAD_COLOR) applies to 16-bit storage."""
    if codec == "jpeg2k":
        return "shift", _jpeg2k_precision(data)
    if codec == "avif":
        rule = "round" if img.ndim == 2 or img.shape[2] < 3 else "scale"
        return rule, _avif_depth(data)
    if (
        codec == "tiff"
        and _tiff_ifd0(data, (_TIFF_PHOTOMETRIC_TAG,)).get(_TIFF_PHOTOMETRIC_TAG)
        == _TIFF_PHOTOMETRIC_RGB
        and not _opencv_reads_tiff_scanlines(data)
    ):
        return "scale", 16
    return "shift", 16


def _to_uint8(img: np.ndarray, codec: str | None, data: bytes) -> np.ndarray:
    """Reduce 8/16-bit integer samples to uint8 exactly as cv2.imdecode does.

    Signed samples are read as their unsigned bit patterns. 16-bit storage
    keeps its high byte (PNG, gray TIFF, TIFF read scanline by scanline),
    except where OpenCV converts by the codec's real depth: libtiff's RGBA
    reader rounds ``value / 257`` for RGB TIFFs, JPEG 2000 shifts by
    ``precision - 8`` and AVIF scales from its 10/12-bit range. Other dtypes
    are returned as is.
    """
    if img.dtype.kind == "i" and img.dtype.itemsize <= 2:
        img = img.view(f"u{img.dtype.itemsize}")
    if img.dtype != np.uint16:
        return img

    rule, bits = _opencv_depth_rule(img, codec, data)
    reduced = _reduce(img, _reduction_table(rule, bits))

    return reduced


def _decode_ic(data: bytes | memoryview) -> np.ndarray:
    """Decode compressed image bytes to RGB HWC uint8 via an explicit codec.

    EXIF orientation is applied where cv2.imdecode applies it (JPEG, PNG,
    TIFF), so the result is upright. The result may be a non-contiguous view.

    Falls back to imagecodecs.imread() probing when the header is unrecognised.
    """
    raw = bytes(data)
    codec = _select_codec(raw)
    if codec is None:
        decoded = imagecodecs.imread(raw)
        _guard_decoded(decoded, max_decoded_pixels())
        return _to_rgb_hwc(_to_uint8(decoded, codec, raw))
    orientation = exif_orientation(raw)
    # Some imagecodecs versions use libtiff's RGBA reader, which flips by the tag.
    if codec == "tiff" and orientation != 1:
        raw = _with_orientation_1(raw)
    decoded = getattr(imagecodecs, f"{codec}_decode")(raw)
    _guard_decoded(decoded, max_decoded_pixels())
    image = _to_rgb_hwc(_to_uint8(decoded, codec, raw))
    # cv2.imdecode (the previous server) honours EXIF for JPEG, PNG and TIFF, so
    # a rotated phone photo must reach the model upright. exif_orientation
    # returns 1 for the formats OpenCV leaves as is.
    return apply_exif_orientation(image, orientation)


DECODER_FACTORIES: dict[str, Callable[[str], Callable[[bytes], Any]]] = {}
_ENTRY_POINT_DECODERS_LOADED = False


def register_decoder(
    name: str, factory: Callable[[str], Callable[[bytes], Any]]
) -> None:
    """Register a decoder factory under ``name``.

    DECODER OUTPUT CONTRACT: ``factory(device)`` must return a callable that
    maps encoded bytes to MODEL-READY input — a BGR HWC uint8 ndarray, or a
    device tensor the ``inference_models`` models accept. The backend applies
    no further conversion on top of it.
    """
    DECODER_FACTORIES[name] = factory


def _imagecodecs_factory(device: str) -> Callable[[bytes], Any]:
    def _decode_imagecodecs(data: bytes) -> Any:
        decoded = _decode_ic(data)
        # Before the BGR copy: a headerless oversized image must not be
        # allocated twice on its way to being rejected.
        _guard_decoded(decoded, max_decoded_pixels())
        return decoded[..., ::-1].copy()

    return _decode_imagecodecs


register_decoder("imagecodecs", _imagecodecs_factory)


_ENTRY_POINT_DECODERS_LOCK = threading.Lock()


def _load_entry_point_decoders() -> None:
    global _ENTRY_POINT_DECODERS_LOADED
    if _ENTRY_POINT_DECODERS_LOADED:
        return
    with _ENTRY_POINT_DECODERS_LOCK:
        if _ENTRY_POINT_DECODERS_LOADED:
            return
        import importlib.metadata as md

        for ep in md.entry_points(group="inference_model_manager.decoders"):
            if ep.name not in DECODER_FACTORIES:
                register_decoder(ep.name, ep.load())
        _ENTRY_POINT_DECODERS_LOADED = True


def _reset_entry_point_decoders_for_tests() -> None:
    global _ENTRY_POINT_DECODERS_LOADED
    with _ENTRY_POINT_DECODERS_LOCK:
        _ENTRY_POINT_DECODERS_LOADED = False


def make_decoder(name: str, device: str = "cuda:0") -> Callable[[bytes], Any]:
    """Resolve ``name`` to a decoder factory and instantiate it for ``device``.

    Looks up ``name`` in ``DECODER_FACTORIES`` first, then — on a miss —
    triggers one-time discovery of the ``inference_model_manager.decoders``
    entry-point group before retrying. Whatever the factory returns is wrapped
    in the decompression-bomb gate.

    DECODER OUTPUT CONTRACT: the returned callable maps encoded bytes to
    MODEL-READY input — a BGR HWC uint8 ndarray, or a device tensor the
    ``inference_models`` models accept. The backend applies no further
    conversion on top of it.

    Args:
        name: Registered decoder name, e.g. ``"imagecodecs"``.
        device: Device string forwarded to the factory (used by GPU decoders).

    Raises:
        ValueError: If ``name`` is not registered directly or via the
            ``inference_model_manager.decoders`` entry-point group.
    """
    factory = DECODER_FACTORIES.get(name)
    if factory is None:
        _load_entry_point_decoders()
        factory = DECODER_FACTORIES.get(name)
    if factory is None:
        raise ValueError(
            f"Unknown decoder: {name!r}. Known: {sorted(DECODER_FACTORIES)}"
        )
    return _guarded(factory(device))
