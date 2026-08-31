"""Image decoder factory and registry.

Single-image decoder: make_decoder(name, device) → (bytes) -> image

Decoders:
  imagecodecs  — CPU, RGB HWC uint8 numpy (replaces cv2)
"""

from __future__ import annotations

import io
import threading
from typing import Any, Callable

import imagecodecs
import numpy as np
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HAS_HEIF = True
except ImportError:
    _HAS_HEIF = False

# HEIC/AVIF: ISO-BMFF `ftyp` box at offset 4 + a HEIF/AVIF brand at offset 8.
# The brand check matters: every ISO-BMFF file (MP4, MOV) has `ftyp` at 4.
_FTYP_OFFSET = 4
_FTYP_MAGIC = b"ftyp"
_HEIF_BRANDS = frozenset(
    (
        b"heic",
        b"heix",
        b"hevc",
        b"heim",
        b"heis",
        b"hevm",
        b"hevs",
        b"mif1",
        b"msf1",
        b"avif",
        b"avis",
    )
)


def _is_heif(data: bytes | memoryview) -> bool:
    return (
        bytes(data[_FTYP_OFFSET : _FTYP_OFFSET + 4]) == _FTYP_MAGIC
        and bytes(data[8:12]) in _HEIF_BRANDS
    )


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

    Falls back to imagecodecs.imread() probing when the header is unrecognised.
    """
    raw = bytes(data)
    codec = _select_codec(raw)
    if codec is None:
        return _to_rgb_hwc(imagecodecs.imread(raw))
    return _to_rgb_hwc(getattr(imagecodecs, f"{codec}_decode")(raw))


def _decode_heif(data: bytes) -> np.ndarray:
    """Decode HEIC/AVIF via Pillow+pillow-heif → RGB HWC uint8 numpy."""
    if not _HAS_HEIF:
        raise ValueError(
            "HEIC/AVIF image received but pillow-heif is not installed. "
            "Install with: pip install pillow-heif"
        )
    img = Image.open(io.BytesIO(data))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


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
        decoded = _decode_heif(data) if _is_heif(data) else _decode_ic(data)
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
    entry-point group before retrying.

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
    return factory(device)
