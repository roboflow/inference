"""GPU-native ``VideoFrameProducer`` for discrete NVIDIA GPUs, backed by ``PyNvVideoCodec`` (NVDEC).

Decodes a video **file** through the NVIDIA Video Codec SDK and yields each frame as a
``torch.Tensor`` on CUDA — zero-copy via DLPack, then ``.clone()`` for ownership. Every
import of ``PyNvVideoCodec`` / ``torch`` is *local*, so importing this module is always
safe; only instantiation requires the dependencies. Probe availability via
``inference.core.interfaces.camera.discoverability.check_pynvvideocodec()``.

Limitations (established during the design discussion):
- ``SimpleDecoder`` is **file-only** — no RTSP / network. Live sources require demuxing
  yourself and feeding the low-level packet decoder; out of scope for this experiment.
- ``PyNvVideoCodec`` hard-links ``libnvidia-encode`` (NVENC) even for decode, so it will
  not import on NVENC-less GPUs (e.g. A100) or containers without the ``video`` driver
  capability. The discoverability probe surfaces that as "unavailable".

Experimental scope: frames are ``.clone()``-ed (the NVDEC surface pool is finite and
recycled as decoding advances); color order / layout / device left to the consumer.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from inference.core.interfaces.camera.entities import (
    FrameImage,
    SourceProperties,
    VideoFrameProducer,
)

if TYPE_CHECKING:
    import torch

# Planar CHW RGB — DL-friendly. Switch to "NV12" to isolate raw decode (no GPU colour conv).
_DEFAULT_OUTPUT_COLOR_TYPE = "RGBP"


class PyNvVideoCodecFrameProducer(VideoFrameProducer):
    """``VideoFrameProducer`` backed by dGPU NVDEC via ``PyNvVideoCodec.SimpleDecoder``."""

    def __init__(
        self,
        video: str,
        *,
        gpu_id: int = 0,
        output_color_type: str = _DEFAULT_OUTPUT_COLOR_TYPE,
    ):
        # Local imports keep this module importable without the dGPU decode stack.
        try:
            import PyNvVideoCodec as nvc
            import torch  # noqa: F401 - needed at retrieve() time; imported here to fail fast
        except Exception as error:  # noqa: BLE001
            raise ImportError(
                "PyNvVideoCodecFrameProducer requires `PyNvVideoCodec` + `torch` with a "
                "working CUDA/NVDEC stack (note: PyNvVideoCodec hard-links libnvidia-encode "
                "even for decode). Probe via "
                "inference.core.interfaces.camera.discoverability.check_pynvvideocodec()."
            ) from error

        self._source_ref = video
        self._gpu_id = gpu_id
        # SimpleDecoder is file-only and iterable; use_device_memory keeps frames on GPU.
        self._decoder = nvc.SimpleDecoder(
            video,
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=getattr(nvc.OutputColorType, output_color_type),
        )
        self._iterator = iter(self._decoder)
        # DLPack-capable frame cached between grab() and retrieve().
        self._last_frame = None
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def grab(self) -> bool:
        try:
            self._last_frame = next(self._iterator)
            return True
        except StopIteration:
            self._last_frame = None
            self._opened = False
            return False

    def retrieve(self) -> Tuple[bool, Optional[FrameImage]]:
        import torch

        frame = self._last_frame
        self._last_frame = None
        if frame is None:
            return False, None
        # Decoded frame exposes __dlpack__ -> zero-copy wrap on CUDA.
        tensor = torch.from_dlpack(frame)
        # clone() for ownership: the NVDEC decode-surface pool is finite and recycled as
        # decoding advances, so a held view would be use-after-overwrite.
        return True, tensor.clone()

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        # No-op: decoder options are fixed at construction. Present for contract parity.
        return None

    def discover_source_properties(self) -> SourceProperties:
        # PyNvVideoCodec exposes container/stream metadata; the exact accessor/attribute
        # names vary by version — verify against your installed PyNvVideoCodec.
        width, height, fps, total_frames = self._read_stream_metadata()
        return SourceProperties(
            width=width,
            height=height,
            total_frames=total_frames,
            is_file=True,  # SimpleDecoder is file-only
            fps=fps,
            is_reconnectable=False,
            timestamp_created=None,
        )

    def _read_stream_metadata(self) -> Tuple[int, int, float, int]:
        try:
            meta = self._decoder.get_stream_metadata()
            width = int(getattr(meta, "width", 0) or 0)
            height = int(getattr(meta, "height", 0) or 0)
            fps = float(getattr(meta, "average_fps", 0.0) or 0.0)
            total_frames = int(getattr(meta, "num_frames", 0) or 0)
        except Exception:  # noqa: BLE001 - metadata API differs across versions
            width = height = total_frames = 0
            fps = 0.0
        if not total_frames:
            try:
                total_frames = len(self._decoder)
            except Exception:  # noqa: BLE001
                total_frames = 0
        return width, height, fps, total_frames

    def release(self) -> None:
        self._last_frame = None
        self._iterator = None
        self._decoder = None
        self._opened = False
