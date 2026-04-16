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

from typing import Any, Callable, List


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
            return imagecodecs.imread(data)   # RGB HWC uint8 numpy

        return _decode_imagecodecs

    if name == "nvjpeg":
        import torch  # noqa: PLC0415
        import torchvision.io  # noqa: PLC0415

        torch_device = torch.device(device)

        def _decode_nvjpeg(data: bytes) -> Any:
            if data[:2] == b"\xff\xd8":
                buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                return torchvision.io.decode_jpeg(buf, device=torch_device)
            # Non-JPEG: imagecodecs fallback → (C,H,W) tensor
            import imagecodecs  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            img = imagecodecs.imread(data)   # RGB HWC uint8
            return (
                torch.from_numpy(np.ascontiguousarray(img))
                .permute(2, 0, 1)
                .to(torch_device)
            )

        return _decode_nvjpeg

    raise ValueError(
        f"Unknown decoder: {name!r}. Supported: 'imagecodecs', 'nvjpeg'"
    )


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
    _JPEG_MAGIC  = b"\xff\xd8"

    if use_nvjpeg:
        import torchvision.io as _tvio  # noqa: PLC0415

        def _batch_decode_nvjpeg(mvs: List[memoryview]) -> List[Any]:
            jpeg_idx:  list[int]        = []
            jpeg_bufs: list[Any]        = []
            other_idx: list[int]        = []
            other_mvs: list[memoryview] = []

            for i, mv in enumerate(mvs):
                if bytes(mv[:2]) == _JPEG_MAGIC:
                    jpeg_idx.append(i)
                    # Zero-copy: frombuffer references the SHM memoryview directly
                    jpeg_bufs.append(torch.frombuffer(mv, dtype=torch.uint8))
                else:
                    other_idx.append(i)
                    other_mvs.append(mv)

            out: list[Any] = [None] * len(mvs)

            # One decode_jpeg call for all JPEGs in the batch
            if jpeg_bufs:
                decoded = _tvio.decode_jpeg(jpeg_bufs, device=torch_device)
                for i, t in zip(jpeg_idx, decoded):
                    out[i] = t

            # Non-JPEG: imagecodecs per image → CHW RGB tensor
            for i, mv in zip(other_idx, other_mvs):
                img = _ic.imread(bytes(mv))   # RGB HWC uint8 numpy
                out[i] = (
                    torch.from_numpy(np.ascontiguousarray(img))
                    .permute(2, 0, 1)
                    .to(torch_device)
                )

            return out

        return _batch_decode_nvjpeg

    # imagecodecs for everything
    def _batch_decode_imagecodecs(mvs: List[memoryview]) -> List[Any]:
        out = []
        for mv in mvs:
            img = _ic.imread(bytes(mv))   # RGB HWC uint8 numpy
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
            if torch_device.type != "cpu":
                t = t.to(torch_device)
            out.append(t)
        return out

    return _batch_decode_imagecodecs
