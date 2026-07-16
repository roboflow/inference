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

_GST_RANK_PRIMARY = 256
_NVIDIA_DECODER_RANK = _GST_RANK_PRIMARY + 100
# A live/RTSP source that yields no frame within this window is treated as
# stalled: the native grab() raises TimeoutError instead of blocking forever, so
# VideoSource surfaces an error (reconnect / cv2 fallback) rather than deadlock
# on its state-change lock. Applies to first-frame discovery and steady-state
# consumption alike. Tune per deployment via the env var below; the value must
# be validated on-device against real RTSP connect + keyframe latency.
_DEFAULT_GRAB_TIMEOUT_NS = 15_000_000_000
_GRAB_TIMEOUT_ENV_VAR = "ROBOFLOW_JETSON_GRAB_TIMEOUT_SECONDS"


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


_RTSP_CODEC_ENV_VAR = "ROBOFLOW_RTSP_VIDEO_CODEC"
_RTSP_PROTOCOLS_ENV_VAR = "ROBOFLOW_RTSP_PROTOCOLS"
_RTSP_LATENCY_ENV_VAR = "ROBOFLOW_RTSP_LATENCY_MS"
_DEFAULT_RTSP_PROTOCOLS = "tcp"
_DEFAULT_RTSP_LATENCY_MS = 200
_RTSP_VIDEO_CODECS = ("h264", "h265")

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


def probe_gstreamer_elements(
    elements: Iterable[str], *, boost_ranks: bool = False
) -> Tuple[bool, str]:
    """Verify the required GStreamer element factories exist.

    When ``boost_ranks`` is True (only at producer BUILD time, not during availability
    probing) the NVIDIA HW decoders (``nvv4l2decoder``/``nvjpegdec``) are ranked above
    software decoders so ``decodebin``/``uridecodebin`` auto-selects them. Keeping the
    boost OFF during probing means merely discovering this backend no longer perturbs
    decoder selection process-wide for unrelated paths (e.g. a later cv2-GStreamer
    decode) - the boost is applied only when a Jetson pipeline is actually built."""

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

        if boost_ranks:
            for decoder_name in ("nvv4l2decoder", "nvjpegdec"):
                decoder = factories.get(decoder_name)
                if decoder:
                    gst.gst_plugin_feature_set_rank(decoder, _NVIDIA_DECODER_RANK)
        return True, "ok"
    finally:
        for factory in factories.values():
            gst.gst_object_unref(factory)


def required_gstreamer_elements(
    video: Optional[Union[str, int]] = None,
    *,
    output_tensor: bool = False,
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
    if _is_rtsp_source(video):
        # RTSP uses an explicit rtspsrc ! depay ! parse ! nvv4l2decoder chain
        # (no uridecodebin autoplugging), so only those elements are required.
        return tuple(
            elements
            + ["h264parse", "h265parse", "nvv4l2decoder"]
            + list(_RTSP_ELEMENTS)
        )
    elements.extend(_URI_DECODE_ELEMENTS)
    local_file_path = _local_file_path(video)
    if local_file_path is not None:
        demuxer = _FILE_DEMUXERS.get(Path(local_file_path).suffix.lower())
        if demuxer is not None:
            elements.append(demuxer)
    return tuple(elements)


def build_gstreamer_pipeline(
    video: Union[str, int],
    *,
    output_tensor: bool = False,
) -> str:
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

    if _is_rtsp_source(video):
        # Explicit video-only chain (the jetson-utils shape) instead of
        # uridecodebin: autoplugging an RTSP source decodes EVERY track, and a
        # camera that muxes audio (e.g. A-Law) poisons the bus with a
        # missing-decoder error — fatal at startup when it races preroll, and a
        # mid-run grab() failure otherwise. A codec-specific depayloader only
        # ever links the video stream, so the audio track is never plugged.
        # protocols defaults to tcp: RTP-over-UDP needs raised kernel buffers
        # and a NAT-free path that containers typically lack, and a failed UDP
        # SETUP can make cameras drop the whole control connection.
        codec = _rtsp_video_codec()
        return (
            f'rtspsrc location="{_quote_gstreamer_value(str(video))}" '
            f"protocols={_rtsp_protocols()} latency={_rtsp_latency_ms()} ! "
            f"rtp{codec}depay ! {codec}parse ! nvv4l2decoder ! "
            f"{sink}"
        )

    uri = _source_uri(video)
    return (
        f'uridecodebin uri="{_quote_gstreamer_value(uri)}" '
        'caps="video/x-raw(memory:NVMM)" ! '
        f"{sink}"
    )


def _rtsp_video_codec() -> str:
    codec = os.getenv(_RTSP_CODEC_ENV_VAR, _RTSP_VIDEO_CODECS[0]).strip().lower()
    if codec not in _RTSP_VIDEO_CODECS:
        raise ValueError(
            f"Unsupported RTSP video codec {codec!r} in {_RTSP_CODEC_ENV_VAR} "
            f"(supported: {', '.join(_RTSP_VIDEO_CODECS)})"
        )
    return codec


def _rtsp_protocols() -> str:
    return os.getenv(_RTSP_PROTOCOLS_ENV_VAR, _DEFAULT_RTSP_PROTOCOLS).strip() or (
        _DEFAULT_RTSP_PROTOCOLS
    )


def _rtsp_latency_ms() -> int:
    raw = os.getenv(_RTSP_LATENCY_ENV_VAR)
    if raw is None:
        return _DEFAULT_RTSP_LATENCY_MS
    try:
        latency = int(raw)
    except ValueError:
        return _DEFAULT_RTSP_LATENCY_MS
    return latency if latency >= 0 else _DEFAULT_RTSP_LATENCY_MS


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
        pin_host_memory: bool = True,
    ):
        gst_ok, gst_reason = probe_gstreamer_elements(
            required_gstreamer_elements(video, output_tensor=True),
            boost_ranks=True,
        )
        if not gst_ok:
            raise RuntimeError(gst_reason)

        self._source_ref = video
        self._output_tensor = output_tensor
        self._pipeline = build_gstreamer_pipeline(video, output_tensor=True)
        self._decoder_validated = not _source_requires_decoder(video)
        self._prerolled_frame_pending = False
        self._cached_source_properties: Optional[SourceProperties] = None
        self._grab_timeout_ns = _resolve_grab_timeout_ns()
        del pin_host_memory

        import torch

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

    def grab(self) -> bool:
        if self._closed or self._eos:
            return False
        if self._prerolled_frame_pending:
            self._prerolled_frame_pending = False
            return True
        grabbed = self._native_pipeline.grab(timeout_ns=self._grab_timeout_ns)
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
        if not self.grab():
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
    # rtspt:// / rtspst:// are rtspsrc's force-TCP variants of rtsp:// / rtsps://.
    return isinstance(video, str) and video.lower().startswith(
        ("rtsp://", "rtsps://", "rtspt://", "rtspst://")
    )


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
