"""GPU-native ``VideoFrameProducer`` for NVIDIA Jetson, backed by ``jetson_utils`` (NVDEC).

Decodes through the Jetson hardware decoder and yields each frame as a ``torch.Tensor``
on CUDA. ``jetson_utils`` ships with JetPack and is **not** pip-installable on a generic
image, so every import of it is *local* (inside ``__init__`` / methods) — importing this
module is always safe; only instantiation requires the dependency. Probe availability via
``inference.core.interfaces.camera.discoverability.check_jetson_utils()``.

Experimental scope (per the design discussion — deliberately narrow):
- frames are ``.clone()``-ed out of the decoder surface so the consumer owns the memory
  (the next ``Capture()`` may recycle the underlying surface);
- color order / tensor layout / device placement are left to the consumer to handle.

NOTE: exact ``jetson_utils`` call shapes vary across JetPack versions — verify against
the version installed on your device.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from inference.core.interfaces.camera.entities import (
    FrameImage,
    SourceProperties,
    VideoFrameProducer,
)

if TYPE_CHECKING:
    import torch


# uint8 RGB, HWC — the cudaImage layout we wrap zero-copy into a torch tensor.
_DEFAULT_CAPTURE_FORMAT = "rgb8"
_DEFAULT_CAPTURE_TIMEOUT_MS = 1000


class JetsonVideoFrameProducer(VideoFrameProducer):
    """``VideoFrameProducer`` backed by Jetson NVDEC via ``jetson_utils.videoSource``."""

    def __init__(
        self,
        video: str,
        *,
        capture_format: str = _DEFAULT_CAPTURE_FORMAT,
        capture_timeout_ms: int = _DEFAULT_CAPTURE_TIMEOUT_MS,
        argv: Optional[List[str]] = None,
    ):
        # Local import: jetson_utils only exists on JetPack. Failing here (not at module
        # import time) keeps this module importable on machines without the Jetson stack.
        try:
            from jetson_utils import videoSource
        except Exception as error:  # noqa: BLE001 - any import failure means "unavailable"
            raise ImportError(
                "JetsonVideoFrameProducer requires `jetson_utils` (ships with JetPack; not "
                "pip-installable). Probe via "
                "inference.core.interfaces.camera.discoverability.check_jetson_utils()."
            ) from error

        self._source_ref = video
        self._capture_format = capture_format
        self._capture_timeout_ms = capture_timeout_ms
        # jetson_utils.videoSource auto-selects the right backend from the URI scheme:
        # a file path / file:// -> NVDEC file decode, rtsp:// -> NVDEC RTSP, /dev/video* ->
        # V4L2, csi:// -> CSI camera. argv carries extra CLI-style options (codec, size...).
        self._source = videoSource(video, argv or [])
        # cudaImage captured in grab() and consumed in retrieve() (jetson_utils has no
        # grab/retrieve split — Capture() does both).
        self._last_capture = None

    def isOpened(self) -> bool:
        return bool(self._source.IsStreaming())

    def grab(self) -> bool:
        try:
            image = self._source.Capture(
                format=self._capture_format,
                timeout=self._capture_timeout_ms,
            )
        except Exception:  # noqa: BLE001 - some jetson_utils versions raise on timeout
            self._last_capture = None
            return False
        if image is None:
            self._last_capture = None
            return False
        self._last_capture = image
        return True

    def retrieve(self) -> Tuple[bool, Optional[FrameImage]]:
        import torch

        image = self._last_capture
        self._last_capture = None
        if image is None:
            return False, None
        # cudaImage exposes __cuda_array_interface__ -> zero-copy view on CUDA.
        tensor = torch.as_tensor(image, device="cuda")
        # clone() so the consumer owns the memory independently of the decoder's surface
        # pool (the next Capture() may overwrite/recycle the underlying buffer).
        return True, tensor.clone()

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        # No-op: jetson_utils fixes capture options at videoSource construction (pass them
        # through `argv`). Present to satisfy the VideoFrameProducer contract.
        return None

    def discover_source_properties(self) -> SourceProperties:
        width = int(self._source.GetWidth())
        height = int(self._source.GetHeight())
        fps = float(self._source.GetFrameRate())
        is_file = _looks_like_local_file(self._source_ref)
        return SourceProperties(
            width=width,
            height=height,
            # jetson_utils does not expose a reliable frame count for live sources;
            # 0 == unknown. is_file is derived from the URI rather than the count.
            total_frames=0,
            is_file=is_file,
            fps=fps,
            is_reconnectable=not is_file,
            timestamp_created=None,
        )

    def release(self) -> None:
        self._last_capture = None
        try:
            self._source.Close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


def _looks_like_local_file(uri: str) -> bool:
    import os

    if "://" in uri:
        return uri.startswith("file://")
    return os.path.exists(uri)
