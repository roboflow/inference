import ctypes
import ctypes.util
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
from urllib.parse import unquote, urlparse

from inference.core.interfaces.camera.entities import (
    FrameImage,
    SourceProperties,
    VideoFrameProducer,
)

_GST_RANK_PRIMARY = 256
_NVIDIA_DECODER_RANK = _GST_RANK_PRIMARY + 100
# A live/RTSP source that yields no frame within this window is treated as
# stalled: the native grab() raises TimeoutError instead of blocking forever, so
# VideoSource surfaces an error (reconnect / cv2 fallback) rather than deadlock
# on its state-change lock. Applies to first-frame discovery and steady-state
# consumption alike. Tune per deployment via the env var below; the value must
# be validated on-device against real RTSP connect + keyframe latency.
_DEFAULT_GRAB_TIMEOUT_NS = 15_000_000_000
_GRAB_TIMEOUT_ENV_VAR = "ROBOFLOW_GSTREAMER_CUDA_GRAB_TIMEOUT_SECONDS"
_BASE_ELEMENTS = ("appsink", "cudaconvertscale", "queue", "uridecodebin")
_RTSP_ELEMENTS = (
    "h264parse",
    "h265parse",
    "rtph264depay",
    "rtph265depay",
    "rtspsrc",
)
_FILE_DEMUXERS = {
    ".avi": "avidemux",
    ".m4v": "qtdemux",
    ".mkv": "matroskademux",
    ".mov": "qtdemux",
    ".mp4": "qtdemux",
    ".webm": "matroskademux",
}
_DECODER_FACTORY_NAMES = tuple(
    dict.fromkeys(
        [
            "nvh264dec",
            "nvh264sldec",
            "nvh265dec",
            "nvh265sldec",
            "nvjpegdec",
            "nvvp8dec",
            "nvvp9dec",
            "nvav1dec",
        ]
        + [
            f"{decoder}device{device_id}dec"
            for decoder in (
                "nvh264",
                "nvh265",
                "nvjpeg",
                "nvvp8",
                "nvvp9",
                "nvav1",
            )
            for device_id in range(16)
        ]
    )
)
_FORBIDDEN_FACTORY_NAMES = (
    "avdec_h264",
    "avdec_h265",
    "avdec_mjpeg",
    "avdec_av1",
    "avdec_vp8",
    "avdec_vp9",
    "cudadownload",
    "cudaupload",
    "jpegdec",
    "videoconvert",
)


def _resolve_grab_timeout_ns() -> int:
    raw = os.getenv(_GRAB_TIMEOUT_ENV_VAR)
    if raw is None:
        return _DEFAULT_GRAB_TIMEOUT_NS
    try:
        seconds = float(raw)
    except ValueError:
        return _DEFAULT_GRAB_TIMEOUT_NS
    if seconds <= 0:
        return _DEFAULT_GRAB_TIMEOUT_NS
    return int(seconds * 1_000_000_000)


def probe_gstreamer_cuda_elements(
    elements: Iterable[str], *, boost_ranks: bool = False
) -> Tuple[bool, str]:
    """Verify the required GStreamer-CUDA elements exist (including at least one NVIDIA
    decoder). When ``boost_ranks`` is True (only at producer BUILD time, not during
    availability probing) the NVIDIA decoders are ranked above software decoders so
    ``uridecodebin`` auto-selects them; kept OFF during probing so discovery does not
    perturb decoder selection process-wide for unrelated paths (e.g. a later
    cv2-GStreamer decode)."""
    library_name = ctypes.util.find_library("gstreamer-1.0")
    if not library_name:
        library_name = "libgstreamer-1.0.so.0"
    try:
        gst = ctypes.CDLL(library_name)
    except OSError as error:
        return False, f"could not load {library_name}: {error}"

    gst.gst_init_check.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    gst.gst_init_check.restype = ctypes.c_int
    gst.gst_element_factory_find.argtypes = [ctypes.c_char_p]
    gst.gst_element_factory_find.restype = ctypes.c_void_p
    gst.gst_plugin_feature_set_rank.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    gst.gst_plugin_feature_set_rank.restype = None
    gst.gst_object_unref.argtypes = [ctypes.c_void_p]
    gst.gst_object_unref.restype = None

    error = ctypes.c_void_p()
    if not gst.gst_init_check(None, None, ctypes.byref(error)):
        return False, "GStreamer initialisation failed"

    factories = {}
    missing = []
    for name in sorted(set(elements)):
        factory = gst.gst_element_factory_find(name.encode("utf-8"))
        if not factory:
            missing.append(name)
        else:
            factories[name] = factory
    try:
        if missing:
            return False, f"missing GStreamer elements: {', '.join(missing)}"

        decoder_found = False
        for decoder_name in _DECODER_FACTORY_NAMES:
            decoder = gst.gst_element_factory_find(decoder_name.encode("utf-8"))
            if decoder:
                decoder_found = True
                if boost_ranks:
                    gst.gst_plugin_feature_set_rank(decoder, _NVIDIA_DECODER_RANK)
                gst.gst_object_unref(decoder)
        if not decoder_found:
            return False, "no NVIDIA GStreamer decoder factory is available"
        return True, "ok"
    finally:
        for factory in factories.values():
            gst.gst_object_unref(factory)


def required_gstreamer_cuda_elements(
    video: Optional[Union[str, int]] = None,
) -> Sequence[str]:
    elements = list(_BASE_ELEMENTS)
    if video is None:
        return tuple(elements)
    if not isinstance(video, str):
        return tuple(elements)
    if _is_rtsp_source(video):
        elements.extend(_RTSP_ELEMENTS)
    local_file_path = _local_file_path(video)
    if local_file_path is not None:
        demuxer = _FILE_DEMUXERS.get(Path(local_file_path).suffix.lower())
        if demuxer is not None:
            elements.append(demuxer)
    return tuple(elements)


def build_gstreamer_cuda_pipeline(video: str, *, device_id: int = 0) -> str:
    is_live = _local_file_path(video) is None
    queue_options = (
        "max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream"
        if is_live
        else "max-size-buffers=4 max-size-bytes=0 max-size-time=0"
    )
    appsink_options = (
        "max-buffers=1 drop=true sync=false"
        if is_live
        else "max-buffers=4 drop=false sync=false"
    )
    uri = _source_uri(video)
    return (
        f'uridecodebin uri="{_quote_gstreamer_value(uri)}" '
        'caps="video/x-raw(memory:CUDAMemory)" ! '
        f"queue {queue_options} ! "
        f"cudaconvertscale cuda-device-id={device_id} ! "
        "video/x-raw(memory:CUDAMemory),format=RGBP ! "
        f"appsink name=rf_tensor_sink {appsink_options} wait-on-eos=false"
    )


class GstreamerCudaVideoFrameProducer(VideoFrameProducer):
    def __init__(
        self,
        video: str,
        *,
        gpu_id: int = 0,
        output_tensor: bool = True,
    ) -> None:
        if not _supports_uri_source(video):
            raise TypeError("GStreamer CUDA producer requires a URI or file path")

        gst_ok, gst_reason = probe_gstreamer_cuda_elements(
            required_gstreamer_cuda_elements(video),
            boost_ranks=True,
        )
        if not gst_ok:
            raise RuntimeError(gst_reason)

        try:
            import torch
        except Exception as error:  # noqa: BLE001 - optional runtime capability
            raise ImportError("GStreamer CUDA decoding requires torch") from error
        if not torch.cuda.is_available():
            raise RuntimeError("GStreamer CUDA decoding requires CUDA")
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {gpu_id} is unavailable")

        from inference.core.interfaces.camera.gstreamer_cuda_tensor_bridge import (
            NativeGstreamerCudaTensorPipeline,
            gstreamer_cuda_tensor_bridge_available,
        )

        bridge_ok, bridge_reason = gstreamer_cuda_tensor_bridge_available()
        if not bridge_ok:
            raise RuntimeError(bridge_reason)

        self._source_ref = video
        self._output_tensor = output_tensor
        self._pipeline_description = build_gstreamer_cuda_pipeline(
            video, device_id=gpu_id
        )
        self._native_pipeline = NativeGstreamerCudaTensorPipeline(
            self._pipeline_description, device_id=gpu_id
        )
        self._decoder_validated = False
        self._prerolled_frame_pending = False
        self._cached_source_properties: Optional[SourceProperties] = None
        self._grab_timeout_ns = _resolve_grab_timeout_ns()
        self._closed = False
        self._eos = False

    @property
    def pipeline(self) -> str:
        return self._pipeline_description

    def isOpened(self) -> bool:
        return not self._closed and not self._eos

    def grab(self) -> bool:
        if self._closed or self._eos:
            return False
        if self._prerolled_frame_pending:
            self._prerolled_frame_pending = False
            return True
        grabbed = self._native_pipeline.grab(timeout_ns=self._grab_timeout_ns)
        if grabbed and not self._decoder_validated:
            self._validate_pipeline()
        if not grabbed:
            self._eos = True
        return grabbed

    def retrieve(self) -> Tuple[bool, Optional[FrameImage]]:
        if self._closed or self._eos:
            return False, None
        self._prerolled_frame_pending = False
        rgb_tensor = self._native_pipeline.retrieve()
        if self._output_tensor:
            return True, rgb_tensor
        return True, _rgb_tensor_to_bgr_numpy(rgb_tensor)

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        return None

    def discover_source_properties(self) -> SourceProperties:
        if self._cached_source_properties is not None:
            return self._cached_source_properties
        if not self.grab():
            raise RuntimeError(
                "GStreamer CUDA pipeline did not produce source metadata"
            )
        self._prerolled_frame_pending = True
        frame_info = self._native_pipeline.frame_info()
        fps = (
            frame_info.fps_numerator / frame_info.fps_denominator
            if frame_info.fps_denominator > 0
            else 0.0
        )
        total_frames = (
            int(frame_info.duration_ns * fps / 1_000_000_000)
            if frame_info.duration_ns > 0 and fps > 0
            else 0
        )
        local_path = _local_file_path(self._source_ref)
        is_file = local_path is not None
        timestamp_created = None
        if is_file and fps > 0 and total_frames > 0 and os.path.isfile(local_path):
            file_length_seconds = total_frames / fps
            last_modified = datetime.fromtimestamp(os.path.getmtime(local_path))
            timestamp_created = last_modified - timedelta(seconds=file_length_seconds)
        properties = SourceProperties(
            width=frame_info.width,
            height=frame_info.height,
            total_frames=total_frames,
            is_file=is_file,
            fps=fps,
            is_reconnectable=not is_file,
            timestamp_created=timestamp_created,
        )
        self._cached_source_properties = properties
        return properties

    def interrupt(self) -> None:
        if self._closed or self._eos:
            return
        self._eos = True
        self._native_pipeline.interrupt()

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._prerolled_frame_pending = False
        self._native_pipeline.close()

    @property
    def tensor_bridge_stats(self) -> Dict[str, int]:
        return self._native_pipeline.stats()

    def _validate_pipeline(self) -> None:
        if not self._native_pipeline.has_factory("cudaconvertscale"):
            raise RuntimeError(
                "GStreamer pipeline did not instantiate cudaconvertscale"
            )
        forbidden = [
            factory
            for factory in _FORBIDDEN_FACTORY_NAMES
            if self._native_pipeline.has_factory(factory)
        ]
        if forbidden:
            raise RuntimeError(
                "GStreamer CUDA pipeline instantiated forbidden elements: "
                + ", ".join(forbidden)
            )
        if not any(
            self._native_pipeline.has_factory(factory)
            for factory in _DECODER_FACTORY_NAMES
        ):
            raise RuntimeError(
                "GStreamer CUDA pipeline did not instantiate an NVIDIA decoder"
            )
        self._decoder_validated = True


def _source_uri(video: str) -> str:
    local_path = _local_file_path(video)
    if local_path is not None:
        return Path(local_path).resolve().as_uri()
    return video


def _local_file_path(video: str) -> Optional[str]:
    if video.startswith("file://"):
        parsed = urlparse(video)
        return unquote(parsed.path)
    if "://" in video:
        return None
    return video


def _is_rtsp_source(video: str) -> bool:
    return video.lower().startswith(("rtsp://", "rtsps://"))


def _supports_uri_source(video: object) -> bool:
    return (
        isinstance(video, str)
        and not video.startswith("/dev/video")
        and not video.lower().startswith("csi://")
    )


def _rgb_tensor_to_bgr_numpy(rgb_tensor):
    return rgb_tensor.permute(1, 2, 0).flip(-1).contiguous().cpu().numpy()


def _quote_gstreamer_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
