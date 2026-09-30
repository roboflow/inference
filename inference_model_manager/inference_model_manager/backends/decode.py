"""Image decoder factory and registry.

Single-image decoder: make_decoder(name, device) → (bytes) -> image

Decoders:
  imagecodecs  — CPU, RGB HWC uint8 numpy (replaces cv2)
"""

from __future__ import annotations

import threading
import zlib
from typing import Any, Callable

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


_EXIF_ORIENTATION_TAG = 0x0112
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
    head = bytes(data[:8])
    if head[:3] == b"\xff\xd8\xff":
        return _jpeg_orientation(data)
    if head == b"\x89PNG\r\n\x1a\n":
        return _png_orientation(data)
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return _tiff_orientation(data) or 1
    return 1


def decoded_dims(data: bytes | memoryview, width: int, height: int) -> tuple[int, int]:
    """(width, height) of the image the decoder returns for ``data``.

    ``width`` and ``height`` are the size a header reader (e.g. Pillow)
    reported. TIFF re-reads its stored size from IFD0 instead, because Pillow
    already swaps the reported TIFF size for orientations 5-8. The swap below
    then uses the same rule as the decoder, so size and pixels always agree.
    """
    if bytes(data[:4]) in (b"II*\x00", b"MM\x00*"):
        stored = _tiff_ifd0(data, (_TIFF_WIDTH_TAG, _TIFF_HEIGHT_TAG))
        if _TIFF_WIDTH_TAG in stored and _TIFF_HEIGHT_TAG in stored:
            width, height = stored[_TIFF_WIDTH_TAG], stored[_TIFF_HEIGHT_TAG]
    if exif_orientation(data) in EXIF_TRANSPOSING_ORIENTATIONS:
        return height, width
    return width, height


def _jpeg_orientation(data: bytes | memoryview) -> int:
    """Orientation from the APP1 Exif block, walking segments up to the scan."""
    size = len(data)
    offset = 2
    while offset + 3 < size:
        if data[offset] != 0xFF:
            return 1
        marker = data[offset + 1]
        if marker == 0xFF:
            offset += 1
            continue
        # Start of scan / end of image: the metadata segments are behind us.
        if marker in (0xDA, 0xD9):
            return 1
        segment_length = int.from_bytes(data[offset + 2 : offset + 4], "big")
        if segment_length < 2:
            return 1
        payload_start = offset + 4
        payload_end = min(offset + 2 + segment_length, size)
        if marker == 0xE1 and bytes(data[payload_start : payload_start + 6]) == (
            b"Exif\x00\x00"
        ):
            orientation = _tiff_orientation(
                bytes(data[payload_start + 6 : payload_end])
            )
            if orientation is not None:
                return orientation
        offset += 2 + segment_length
    return 1


def _png_orientation(data: bytes | memoryview) -> int:
    """Orientation from the first eXIf chunk, 1 when absent or invalid.

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
            return 1
        if chunk_type == b"eXIf":
            crc = int.from_bytes(data[payload_end : payload_end + 4], "big")
            if zlib.crc32(data[offset + 4 : payload_end]) != crc:
                return 1
            return _tiff_orientation(bytes(data[offset + 8 : payload_end])) or 1
        offset = payload_end + 4
    return 1


_TIFF_WIDTH_TAG = 0x0100
_TIFF_HEIGHT_TAG = 0x0101


def _tiff_ifd0(tiff: bytes | memoryview, tags: tuple[int, ...]) -> dict[int, int]:
    """SHORT/LONG values of ``tags`` in IFD0 of a TIFF block; malformed → {}."""
    if len(tiff) < 8:
        return {}
    if tiff[:2] == b"II":
        order = "little"
    elif tiff[:2] == b"MM":
        order = "big"
    else:
        return {}
    if int.from_bytes(tiff[2:4], order) != 42:
        return {}
    ifd = int.from_bytes(tiff[4:8], order)
    if ifd + 2 > len(tiff):
        return {}
    entries = int.from_bytes(tiff[ifd : ifd + 2], order)
    found: dict[int, int] = {}
    for index in range(entries):
        entry = ifd + 2 + 12 * index
        if entry + 12 > len(tiff):
            return {}
        tag = int.from_bytes(tiff[entry : entry + 2], order)
        if tag not in tags:
            continue
        value_type = int.from_bytes(tiff[entry + 2 : entry + 4], order)
        if value_type == 3:  # SHORT
            found[tag] = int.from_bytes(tiff[entry + 8 : entry + 10], order)
        elif value_type == 4:  # LONG
            found[tag] = int.from_bytes(tiff[entry + 8 : entry + 12], order)
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


def _guarded(decode: Callable[[bytes], Any]) -> Callable[[bytes], Any]:
    """Wrap a decoder with the decompression-bomb gate.

    Every decoder reaching a backend goes through here, so a plugin decoder
    registered through the entry-point group is gated exactly like the
    built-in one.
    """

    def _decode(data: bytes) -> Any:
        limit = _guard_encoded(data)
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
    if head[:4] in (b"II*\x00", b"MM\x00*"):
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


def _decode_ic(data: bytes | memoryview) -> np.ndarray:
    """Decode compressed image bytes to RGB HWC uint8 via an explicit codec.

    EXIF orientation is applied where cv2.imdecode applies it (JPEG, PNG,
    TIFF), so the result is upright. The result may be a non-contiguous view.

    Falls back to imagecodecs.imread() probing when the header is unrecognised.
    """
    raw = bytes(data)
    codec = _select_codec(raw)
    if codec is None:
        return _to_rgb_hwc(imagecodecs.imread(raw))
    image = _to_rgb_hwc(getattr(imagecodecs, f"{codec}_decode")(raw))
    # imagecodecs ignores EXIF; cv2.imdecode (the previous server) honours it
    # for JPEG, PNG and TIFF, so a rotated phone photo must reach the model
    # upright. exif_orientation returns 1 for the formats OpenCV leaves as is.
    return apply_exif_orientation(image, exif_orientation(raw))


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
