"""Jetson video capture with NVIDIA GStreamer and CUDA tensor output."""

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
from inference.core.interfaces.camera.exceptions import NativeGrabTimeoutError

_GST_RANK_PRIMARY = 256
_NVIDIA_DECODER_RANK = _GST_RANK_PRIMARY + 100
# Bounded grabs keep a stalled source recoverable: the first grab runs under
# VideoSource's state-transition lock (terminate() needs the same lock), and
# a silent steady-state stall must surface as an error so restart logic fires.
_STARTUP_GRAB_TIMEOUT_NS = 10 * 1_000_000_000
_FRAME_GRAB_TIMEOUT_NS = 30 * 1_000_000_000

_COMMON_ELEMENTS = (
    "appsink",
    "nvvidconv",
    "queue",
)
_URI_DECODE_ELEMENTS = (
    "decodebin",
    "h264parse",
    "h265parse",
    "jpegparse",
    "nvjpegdec",
    "nvv4l2decoder",
    "uridecodebin",
)
_RTSP_ELEMENTS = (
    "rtph264depay",
    "rtph265depay",
    "rtspsrc",
)
_SOFTWARE_DECODER_ELEMENTS = (
    "avdec_h264",
    "avdec_h265",
    "avdec_mjpeg",
    "jpegdec",
    "libde265dec",
    "openh264dec",
)
_FILE_DEMUXERS = {
    ".avi": "avidemux",
    ".m4v": "qtdemux",
    ".mkv": "matroskademux",
    ".mov": "qtdemux",
    ".mp4": "qtdemux",
    ".webm": "matroskademux",
}


class _GError(ctypes.Structure):
    _fields_ = [
        ("domain", ctypes.c_uint32),
        ("code", ctypes.c_int),
        ("message", ctypes.c_char_p),
    ]


def _load_gstreamer_library():
    library_name = ctypes.util.find_library("gstreamer-1.0")
    if not library_name:
        library_name = "libgstreamer-1.0.so.0"
    gst = ctypes.CDLL(library_name)
    gst.gst_init_check.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(_GError)),
    ]
    gst.gst_init_check.restype = ctypes.c_int
    gst.gst_element_factory_find.argtypes = [ctypes.c_char_p]
    gst.gst_element_factory_find.restype = ctypes.c_void_p
    gst.gst_plugin_feature_set_rank.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    gst.gst_plugin_feature_set_rank.restype = None
    gst.gst_object_unref.argtypes = [ctypes.c_void_p]
    gst.gst_object_unref.restype = None
    gst.g_error_free.argtypes = [ctypes.c_void_p]
    gst.g_error_free.restype = None
    return gst


def _init_gstreamer(gst) -> Tuple[bool, str]:
    error = ctypes.POINTER(_GError)()
    if not gst.gst_init_check(None, None, ctypes.byref(error)):
        reason = "GStreamer initialisation failed"
        if error:
            if error.contents.message:
                message = error.contents.message.decode("utf-8", errors="replace")
                reason = f"{reason}: {message}"
            gst.g_error_free(error)
        return False, reason
    return True, "ok"


def probe_gstreamer_elements(elements: Iterable[str]) -> Tuple[bool, str]:
    """Check that the required element factories are available.

    Availability check only. It must not mutate process-wide GStreamer
    state: it also runs when producers are merely enumerated for logging.
    """

    try:
        gst = _load_gstreamer_library()
    except OSError as error:
        return False, f"could not load GStreamer: {error}"

    initialised, reason = _init_gstreamer(gst)
    if not initialised:
        return False, reason

    missing = []
    for name in sorted(set(elements)):
        factory = gst.gst_element_factory_find(name.encode("utf-8"))
        if not factory:
            missing.append(name)
        else:
            gst.gst_object_unref(factory)
    if missing:
        return False, f"missing GStreamer elements: {', '.join(missing)}"
    return True, "ok"


def promote_nvidia_decoder_ranks() -> None:
    # Raising NVIDIA decoders above the software ones re-ranks them for every
    # GStreamer consumer in the process, so this runs only when a producer is
    # actually constructed, never from availability probes.
    gst = _load_gstreamer_library()
    initialised, reason = _init_gstreamer(gst)
    if not initialised:
        raise RuntimeError(reason)
    for decoder_name in ("nvv4l2decoder", "nvjpegdec"):
        decoder = gst.gst_element_factory_find(decoder_name.encode("utf-8"))
        if decoder:
            gst.gst_plugin_feature_set_rank(decoder, _NVIDIA_DECODER_RANK)
            gst.gst_object_unref(decoder)


def required_gstreamer_elements(
    video: Optional[Union[str, int]] = None,
) -> Sequence[str]:
    """Return the element set needed by a source, or the common baseline."""

    elements = list(_COMMON_ELEMENTS)
    if video is None:
        return tuple(elements + list(_URI_DECODE_ELEMENTS))
    if _is_csi_source(video):
        return tuple(elements + ["nvarguscamerasrc"])
    if _is_v4l2_source(video):
        return tuple(
            elements
            + [
                "decodebin",
                "h264parse",
                "h265parse",
                "jpegparse",
                "nvjpegdec",
                "nvv4l2decoder",
                "v4l2src",
            ]
        )
    elements.extend(_URI_DECODE_ELEMENTS)
    if _is_rtsp_source(video):
        elements.extend(_RTSP_ELEMENTS)
    local_file_path = _local_file_path(video)
    if local_file_path is not None:
        demuxer = _FILE_DEMUXERS.get(Path(local_file_path).suffix.lower())
        if demuxer is not None:
            elements.append(demuxer)
    return tuple(elements)


def build_gstreamer_pipeline(video: Union[str, int]) -> str:
    """Build a Jetson GStreamer pipeline ending in an NVMM appsink."""

    is_live = _is_live_source(video)
    sink = _build_sink(is_live=is_live)
    if _is_csi_source(video):
        sensor_id = _csi_sensor_id(video)
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            "video/x-raw(memory:NVMM),format=NV12 ! "
            f"{sink}"
        )
    if _is_v4l2_source(video):
        device = _v4l2_device(video)
        return (
            f'v4l2src device="{_quote_gstreamer_value(device)}" ! '
            f"decodebin ! {sink}"
        )

    uri = _source_uri(video)
    return (
        f'uridecodebin uri="{_quote_gstreamer_value(uri)}" '
        'caps="video/x-raw(memory:NVMM)" ! '
        f"{sink}"
    )


def _build_sink(is_live: bool) -> str:
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
    return (
        f"queue {queue_options} ! "
        "nvvidconv ! video/x-raw(memory:NVMM),format=RGBA ! "
        f"appsink name=rf_tensor_sink {appsink_options} wait-on-eos=false"
    )


class JetsonVideoFrameProducer(VideoFrameProducer):
    """Decode Jetson file/camera/RTSP sources with NVIDIA GStreamer elements."""

    def __init__(
        self,
        video: Union[str, int],
        *,
        output_tensor: bool = True,
        tensor_device: str = "cuda",
    ):
        gst_ok, gst_reason = probe_gstreamer_elements(
            required_gstreamer_elements(video)
        )
        if not gst_ok:
            raise RuntimeError(gst_reason)

        self._source_ref = video
        self._output_tensor = output_tensor
        self._pipeline = build_gstreamer_pipeline(video)
        self._decoder_validated = not _source_requires_decoder(video)
        self._prerolled_frame_pending = False
        self._cached_source_properties: Optional[SourceProperties] = None

        try:
            import torch
        except Exception as error:  # noqa: BLE001 - optional runtime capability
            raise ImportError("Jetson tensor decoding requires torch") from error

        from inference.core.interfaces.camera.jetson_tensor_bridge import (
            NativeJetsonTensorPipeline,
            jetson_tensor_bridge_available,
        )

        device = torch.device(tensor_device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Jetson decoding requires an available CUDA device")
        device_id = (
            torch.cuda.current_device() if device.index is None else device.index
        )
        bridge_ok, bridge_reason = jetson_tensor_bridge_available()
        if not bridge_ok:
            raise RuntimeError(bridge_reason)
        promote_nvidia_decoder_ranks()
        self._native_pipeline = NativeJetsonTensorPipeline(
            self._pipeline, device_id=device_id
        )
        self._closed = False
        self._eos = False

    @property
    def pipeline(self) -> str:
        """Pipeline string exposed for diagnostics and on-device tests."""

        return self._pipeline

    def isOpened(self) -> bool:
        return not self._closed and not self._eos

    def grab(self, timeout_ns: Optional[int] = None) -> bool:
        if self._closed or self._eos:
            return False
        if self._prerolled_frame_pending:
            self._prerolled_frame_pending = False
            return True
        try:
            grabbed = self._native_pipeline.grab(
                timeout_ns=_FRAME_GRAB_TIMEOUT_NS if timeout_ns is None else timeout_ns
            )
        except NativeGrabTimeoutError as error:
            raise RuntimeError(
                "Jetson pipeline stalled: no frame arrived within "
                f"{(_FRAME_GRAB_TIMEOUT_NS if timeout_ns is None else timeout_ns) / 1e9:.0f}s"
            ) from error
        if grabbed and not self._decoder_validated:
            self._validate_hardware_decoder()
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
        # Discovery runs under VideoSource's state-transition lock, which
        # terminate() also needs, so this grab must be bounded.
        if not self.grab(timeout_ns=_STARTUP_GRAB_TIMEOUT_NS):
            raise RuntimeError("Jetson pipeline did not produce source metadata")
        self._prerolled_frame_pending = True
        frame_info = self._native_pipeline.frame_info()
        width = frame_info.width
        height = frame_info.height
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
            width=width,
            height=height,
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
        self._native_pipeline.interrupt()
        self._eos = True

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._prerolled_frame_pending = False
        self._native_pipeline.close()

    @property
    def tensor_bridge_stats(self) -> Dict[str, int]:
        return self._native_pipeline.stats()

    def _validate_hardware_decoder(self) -> None:
        if any(
            self._native_pipeline.has_factory(factory)
            for factory in ("nvv4l2decoder", "nvjpegdec")
        ):
            self._decoder_validated = True
            return
        software_decoders = [
            factory
            for factory in _SOFTWARE_DECODER_ELEMENTS
            if self._native_pipeline.has_factory(factory)
        ]
        if software_decoders:
            raise RuntimeError(
                "Jetson pipeline instantiated software decoders: "
                + ", ".join(software_decoders)
            )
        if _is_v4l2_source(self._source_ref):
            self._decoder_validated = True
            return
        raise RuntimeError("Jetson pipeline did not instantiate an NVIDIA decoder")


def _source_uri(video: Union[str, int]) -> str:
    local_path = _local_file_path(video)
    if local_path is not None:
        return Path(local_path).resolve().as_uri()
    return str(video)


def _local_file_path(video: Union[str, int]) -> Optional[str]:
    if not isinstance(video, str):
        return None
    if video.startswith("file://"):
        parsed = urlparse(video)
        return unquote(parsed.path)
    if "://" in video:
        return None
    return video


def _is_rtsp_source(video: Union[str, int]) -> bool:
    return isinstance(video, str) and video.lower().startswith(("rtsp://", "rtsps://"))


def _is_csi_source(video: Union[str, int]) -> bool:
    return isinstance(video, str) and video.lower().startswith("csi://")


def _csi_sensor_id(video: Union[str, int]) -> int:
    try:
        return int(str(video).split("://", 1)[1] or 0)
    except ValueError as error:
        raise ValueError(f"Invalid CSI source reference: {video!r}") from error


def _is_v4l2_source(video: Union[str, int]) -> bool:
    return isinstance(video, int) or (
        isinstance(video, str) and video.startswith("/dev/video")
    )


def _v4l2_device(video: Union[str, int]) -> str:
    return f"/dev/video{video}" if isinstance(video, int) else video


def _is_live_source(video: Union[str, int]) -> bool:
    return (
        _is_csi_source(video)
        or _is_v4l2_source(video)
        or _local_file_path(video) is None
    )


def _source_requires_decoder(video: Union[str, int]) -> bool:
    return not _is_csi_source(video)


def _rgb_tensor_to_bgr_numpy(rgb_tensor):
    return rgb_tensor.permute(1, 2, 0).flip(-1).contiguous().cpu().numpy()


def _quote_gstreamer_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
