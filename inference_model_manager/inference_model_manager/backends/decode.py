"""Image decoder factory.

Single-image decoder: make_decoder(name, device) → (bytes) -> image
Batch decoder:        make_batch_decoder(device, use_nvjpeg) → (list[memoryview]) -> list[Tensor]

Decoders:
  imagecodecs  — CPU, RGB HWC uint8 numpy (replaces cv2)
  nvjpeg       — torchvision.io.decode_jpeg on GPU for JPEG;
                 imagecodecs fallback for non-JPEG formats

Batch decoder output is always List[(C,H,W) RGB uint8 tensor]:
  use_nvjpeg=True  — JPEG path: torchvision.io.decode_jpeg(batch, device) — one GPU call
                     non-JPEG path: imagecodecs.imread(mv) → permute(2,0,1).to(device)
  use_nvjpeg=False — imagecodecs for everything → permute(2,0,1).to(device)
"""

from __future__ import annotations

import io
import logging
from typing import Any, Callable, List

import numpy as np
from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    _HAS_HEIF = True
except ImportError:
    _HAS_HEIF = False

log = logging.getLogger(__name__)

# HEIC/AVIF ftyp magic at offset 4
_FTYP_OFFSET = 4
_FTYP_MAGIC = b"ftyp"


def _is_heif(data: bytes | memoryview) -> bool:
    return bytes(data[_FTYP_OFFSET : _FTYP_OFFSET + 4]) == _FTYP_MAGIC


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


def make_decoder(name: str, device: str = "cuda:0") -> Callable[[bytes], Any]:
    """Create a single-image decoder callable.

    Args:
        name:   ``"imagecodecs"`` (CPU, RGB numpy) or ``"nvjpeg"`` (GPU tensor).
        device: CUDA device string — used by ``"nvjpeg"`` only.

    Returns:
        Callable ``(bytes) -> image``.
    """
    if name == "imagecodecs":
        import imagecodecs  # noqa: PLC0415

        def _decode_imagecodecs(data: bytes) -> Any:
            if _is_heif(data):
                return _decode_heif(data)
            return imagecodecs.imread(data)  # RGB HWC uint8 numpy

        return _decode_imagecodecs

    if name == "nvjpeg":
        import imagecodecs  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torchvision.io  # noqa: PLC0415

        torch_device = torch.device(device)

        def _decode_nvjpeg(data: bytes) -> Any:
            if data[:2] == b"\xff\xd8":
                buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                return torchvision.io.decode_jpeg(buf, device=torch_device)
            # Non-JPEG fallback
            img = _decode_heif(data) if _is_heif(data) else imagecodecs.imread(data)
            return (
                torch.from_numpy(np.ascontiguousarray(img))
                .permute(2, 0, 1)
                .to(torch_device)
            )

        return _decode_nvjpeg

    raise ValueError(f"Unknown decoder: {name!r}. Supported: 'imagecodecs', 'nvjpeg'")


def make_batch_decoder(
    device: str,
    *,
    use_nvjpeg: bool = False,
) -> Callable[[List[memoryview]], List[Any]]:
    """Create a batch image decoder.

    Input:  list of memoryviews pointing to raw compressed image bytes.
    Output: list of ``(C, H, W)`` RGB uint8 tensors, same order as input.

    Args:
        device:     Target device for output tensors (e.g. ``"cuda:0"``, ``"cpu"``).
        use_nvjpeg: When ``True``, JPEG images are decoded via
                    ``torchvision.io.decode_jpeg(batch, device)`` — one GPU call for
                    the whole JPEG batch (nvjpeg on CUDA, libjpeg-turbo on CPU).
                    Non-JPEG images always fall back to imagecodecs.
                    When ``False`` (default), imagecodecs handles all formats.
    """
    import imagecodecs as _ic  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    torch_device = torch.device(device)
    _JPEG_MAGIC = b"\xff\xd8"

    if use_nvjpeg:
        import torchvision.io as _tvio  # noqa: PLC0415

        def _batch_decode_nvjpeg(mvs: List[memoryview]) -> List[Any]:
            jpeg_idx: list[int] = []
            jpeg_bufs: list[Any] = []
            other_idx: list[int] = []
            other_mvs: list[memoryview] = []

            for i, mv in enumerate(mvs):
                if bytes(mv[:2]) == _JPEG_MAGIC:
                    jpeg_idx.append(i)
                    # Zero-copy: frombuffer references the SHM memoryview directly.
                    # PyTorch warns "buffer is not writable" because SHM is read-only,
                    # but the tensor is only used as input to decode_jpeg (never written to).
                    # Using bytearray(mv) would silence it but adds a full CPU memcpy per JPEG.
                    jpeg_bufs.append(torch.frombuffer(mv, dtype=torch.uint8))
                else:
                    other_idx.append(i)
                    other_mvs.append(mv)

            out: list[Any] = [None] * len(mvs)

            # One decode_jpeg call for all JPEGs in the batch
            if jpeg_bufs:
                try:
                    decoded = _tvio.decode_jpeg(jpeg_bufs, device=torch_device)
                    for i, t in zip(jpeg_idx, decoded):
                        out[i] = t
                except Exception:
                    log.warning(
                        "nvjpeg batch decode failed for %d JPEG(s), "
                        "falling back to CPU imagecodecs",
                        len(jpeg_bufs),
                    )
                    for i in jpeg_idx:
                        try:
                            raw = bytes(mvs[i])
                            img = _ic.imread(raw)
                            out[i] = (
                                torch.from_numpy(np.ascontiguousarray(img))
                                .permute(2, 0, 1)
                                .to(torch_device)
                            )
                        except Exception:
                            log.exception(
                                "CPU fallback decode also failed for slot index %d", i
                            )

            # Non-JPEG: imagecodecs (or HEIF fallback) per image → CHW RGB tensor
            for i, mv in zip(other_idx, other_mvs):
                raw = bytes(mv)
                img = _decode_heif(raw) if _is_heif(raw) else _ic.imread(raw)
                out[i] = (
                    torch.from_numpy(np.ascontiguousarray(img))
                    .permute(2, 0, 1)
                    .to(torch_device)
                )

            return out

        return _batch_decode_nvjpeg

    # imagecodecs for everything (HEIF fallback when needed)
    def _batch_decode_imagecodecs(mvs: List[memoryview]) -> List[Any]:
        out = []
        for mv in mvs:
            raw = bytes(mv)
            img = _decode_heif(raw) if _is_heif(raw) else _ic.imread(raw)
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
            if torch_device.type != "cpu":
                t = t.to(torch_device)
            out.append(t)
        return out

    return _batch_decode_imagecodecs
