"""Image decoder factory.

Returns a callable ``(bytes) -> image`` based on a decoder name.
Used by both DirectBackend and SubprocessBackend to decode compressed
image bytes (JPEG/PNG) into numpy arrays or GPU tensors.

Supported decoders:
  cv2      — OpenCV imdecode (CPU, returns BGR numpy)
  nvjpeg   — torchvision.io.decode_jpeg on GPU (JPEG only, returns (C,H,W) CUDA tensor)
"""
from __future__ import annotations

from typing import Any, Callable


def make_decoder(name: str, device: str = "cuda:0") -> Callable[[bytes], Any]:
    """Create a decoder callable.

    Args:
        name: Decoder name — ``"cv2"`` or ``"nvjpeg"``.
        device: CUDA device string for GPU decoders. Ignored for CPU decoders.

    Returns:
        A callable that takes raw bytes and returns a decoded image
        (numpy array for CPU decoders, CUDA tensor for GPU decoders).
    """
    if name == "cv2":
        import cv2
        import numpy as np

        def _decode_cv2(data: bytes) -> Any:
            return cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR,
            )

        return _decode_cv2

    if name == "nvjpeg":
        import numpy as np
        import torch
        import torchvision.io

        torch_device = torch.device(device)

        def _decode_nvjpeg(data: bytes) -> Any:
            if data[:2] == b"\xff\xd8":
                buf = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                return torchvision.io.decode_jpeg(buf, device=torch_device)
            # Non-JPEG — cv2 decode then upload to GPU as (C,H,W) to match nvjpeg output
            import cv2
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            return torch.from_numpy(img).to(torch_device).permute(2, 0, 1).contiguous()

        return _decode_nvjpeg

    raise ValueError(
        f"Unknown decoder: {name!r}. Supported: 'cv2', 'nvjpeg'"
    )
