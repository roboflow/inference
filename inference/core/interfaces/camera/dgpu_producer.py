"""GPU-native ``VideoFrameProducer`` for discrete NVIDIA GPUs, backed by ``PyNvVideoCodec`` (NVDEC).

Decodes a video file through the NVIDIA Video Codec SDK and yields each frame as a
``torch.Tensor`` on CUDA. DLPack wraps the decoded surface, then ``clone()`` gives the
consumer independent storage. ``SimpleDecoder`` accepts file sources. Its runtime library
closure includes ``libnvidia-encode`` and requires the NVIDIA ``video`` driver capability.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from inference.core.interfaces.camera.entities import (
    FrameImage,
    SourceProperties,
    VideoFrameProducer,
)

if TYPE_CHECKING:
    import torch

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
        self._decoder = nvc.SimpleDecoder(
            video,
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=getattr(nvc.OutputColorType, output_color_type),
        )
        self._iterator = iter(self._decoder)
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
        tensor = torch.from_dlpack(frame)
        return True, tensor.clone()

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        return None

    def discover_source_properties(self) -> SourceProperties:
        width, height, fps, total_frames = self._read_stream_metadata()
        return SourceProperties(
            width=width,
            height=height,
            total_frames=total_frames,
            is_file=True,
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
