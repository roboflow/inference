"""Jetson video capture with NVIDIA GStreamer and CUDA tensor output."""

import ctypes
import ctypes.util
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qsl, unquote, urlparse

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
# TLS validation is deliberately opt-in.  Some private camera deployments use
# a self-signed RTSPS certificate; setting this to 0 asks rtspsrc to accept it.
# The secure GStreamer default remains in force when the variable is absent.
_RTSP_TLS_VALIDATION_FLAGS_ENV_VAR = "ROBOFLOW_RTSP_TLS_VALIDATION_FLAGS"
_DEFAULT_RTSP_PROTOCOLS = "tcp"
_DEFAULT_RTSP_LATENCY_MS = 50
_RTSP_VIDEO_CODECS = ("h264", "h265")
_BOUNDED_QUEUE_OPTIONS = "max-size-buffers=64 max-size-bytes=0 max-size-time=50000000"
_LATEST_DECODED_FRAME_QUEUE_OPTIONS = (
    "max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
)

_COMMON_ELEMENTS = (
    "appsink",
    "queue",
)
_NVVIDCONV_ELEMENTS = ("nvvidconv",)
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
    "parsebin",
    "rtph264depay",
    "rtph265depay",
    "rtspsrc",
)
_SRTP_ELEMENTS = ("capssetter", "srtpdec")
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


def probe_gstreamer_elements(
    elements: Iterable[str], *, boost_ranks: bool = False
) -> Tuple[bool, str]:
    """Verify the required GStreamer element factories exist.

    Rank NVIDIA decoders above software decoders only when ``boost_ranks`` is
    explicitly enabled during producer construction. Discovery probes leave
    process-wide decoder selection untouched.
    """

    try:
        gst = _load_gstreamer_library()
    except OSError as error:
        return False, f"could not load GStreamer: {error}"

    initialised, reason = _init_gstreamer(gst)
    if not initialised:
        return False, reason

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
        return tuple(elements + list(_NVVIDCONV_ELEMENTS) + list(_URI_DECODE_ELEMENTS))
    if _is_csi_source(video):
        return tuple(elements + ["nvarguscamerasrc"])
    if _is_v4l2_source(video):
        return tuple(
            elements
            + list(_NVVIDCONV_ELEMENTS)
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
        # The video-only RTP capsfilter prevents an audio track from reaching
        # parsebin. parsebin then selects the H.264 or H.265 depay/parser
        # chain from the RTP caps, without needing a source-specific codec
        # setting. The explicit codec override remains available for cameras
        # with broken SDP metadata.
        codec_override = _rtsp_video_codec_override()
        srtp_elements = list(_SRTP_ELEMENTS) if _rtsp_uses_srtp(video) else []
        if codec_override is not None:
            return tuple(
                elements
                + srtp_elements
                + [
                    f"rtp{codec_override}depay",
                    f"{codec_override}parse",
                    "nvv4l2decoder",
                    "rtspsrc",
                ]
            )
        return tuple(
            elements
            + srtp_elements
            + ["h264parse", "h265parse", "nvv4l2decoder"]
            + list(_RTSP_ELEMENTS)
        )
    elements.extend(_NVVIDCONV_ELEMENTS)
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
            f'v4l2src device="{_quote_gstreamer_value(device)}" do-timestamp=true ! '
            f"queue {_BOUNDED_QUEUE_OPTIONS} ! "
            f"decodebin ! {sink}"
        )

    if _is_rtsp_source(video):
        # A video-only RTP capsfilter keeps audio tracks away from parsebin,
        # while leaving parsebin free to select the H.264 or H.265 depay/parser
        # chain from the camera's RTP caps. This has the codec flexibility of
        # autoplugging without uridecodebin's habit of trying every RTSP track.
        # protocols defaults to tcp: RTP-over-UDP needs raised kernel buffers
        # and a NAT-free path that containers typically lack, and a failed UDP
        # SETUP can make cameras drop the whole control connection.
        #
        # Further jetson-utils parity (v5 bridge):
        # - the bounded queue buffers COMPRESSED data before the depayloader.
        #   It must not leak individual RTP/NAL buffers because that can corrupt
        #   pictures until the next IDR; rtspsrc's jitterbuffer enforces latency,
        #   while decoded frames remain latest-wins in the bridge;
        # - no nvvidconv: the decoder's NV12 NVMM output goes straight to the
        #   appsink and the bridge converts NV12->RGB CHW in CUDA, removing the
        #   per-frame VIC pass and its extra buffer pool;
        # - decoder performance is controlled by the Jetson power mode.  Do
        #   not set nvv4l2decoder's historical ``enable-max-performance``
        #   property here: it is absent on the Thor/JP7.2 plugin and makes a
        #   static pipeline fail to parse before it can receive a frame;
        # - a one-frame leaky queue after NVDEC lets decoding continue while a
        #   full-resolution CUDA conversion is in flight. Surplus complete
        #   decoded frames are dropped before conversion rather than wasting
        #   GPU bandwidth or backpressuring the compressed RTP path;
        # - the appsink never accumulates (the bridge's new-sample callback
        #   drains it on the queue's streaming thread); drop=true explicitly
        #   selects the bridge's latest-frame handoff for this live source.
        codec = _rtsp_video_codec_override()
        tls_validation_flags = _rtsp_tls_validation_flags()
        source = (
            f'rtspsrc location="{_quote_gstreamer_value(str(video))}" '
            f"protocols={_rtsp_protocols()} latency={_rtsp_latency_ms()}"
            " drop-on-latency=true teardown-timeout=0"
            f"{tls_validation_flags} ! "
            "application/x-rtp,media=video ! "
            f"queue {_BOUNDED_QUEUE_OPTIONS} ! "
        )
        if _rtsp_uses_srtp(video):
            # UniFi and other SDES endpoints advertise the master key through
            # the RTP caps' a-crypto field. The native bridge rewrites the
            # named capssetter's CAPS event before srtpdec sees the first
            # packet; key material never crosses Python or appears in the
            # launch string.
            source += (
                "capssetter name=rf_srtp_caps caps=application/x-srtp "
                "join=false replace=false ! srtpdec ! "
            )
        if codec is None:
            decoder = "parsebin name=rf_rtsp_video_parse ! nvv4l2decoder ! "
        else:
            decoder = f"rtp{codec}depay ! {codec}parse ! nvv4l2decoder ! "
        return (
            source + decoder + "video/x-raw(memory:NVMM),format=NV12 ! "
            f"queue {_LATEST_DECODED_FRAME_QUEUE_OPTIONS} ! "
            "appsink name=rf_tensor_sink max-buffers=1 drop=true sync=false "
            "wait-on-eos=false"
        )

    uri = _source_uri(video)
    return (
        f'uridecodebin uri="{_quote_gstreamer_value(uri)}" '
        'caps="video/x-raw(memory:NVMM)" ! '
        f"{sink}"
    )


def _rtsp_video_codec_override() -> Optional[str]:
    """Return a validated explicit RTSP codec override, if configured."""

    raw = os.getenv(_RTSP_CODEC_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    codec = raw.strip().lower()
    if codec not in _RTSP_VIDEO_CODECS:
        raise ValueError(
            f"Unsupported RTSP video codec {codec!r} in {_RTSP_CODEC_ENV_VAR} "
            f"(supported: {', '.join(_RTSP_VIDEO_CODECS)})"
        )
    return codec


def _rtsp_uses_srtp(video: Union[str, int]) -> bool:
    """Return whether an RTSP URL explicitly requests encrypted RTP media."""

    if not isinstance(video, str):
        return False
    for key, value in parse_qsl(urlparse(video).query, keep_blank_values=True):
        if key.lower() != "enablesrtp":
            continue
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return False


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


def _rtsp_tls_validation_flags() -> str:
    """Return an explicit rtspsrc TLS-validation setting when requested.

    ``0`` disables certificate validation for cameras with private/self-signed
    certificates. Keeping this unset by default avoids weakening RTSPS
    validation for normal deployments.
    """

    raw = os.getenv(_RTSP_TLS_VALIDATION_FLAGS_ENV_VAR)
    if raw is None or not raw.strip():
        return ""
    try:
        flags = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{_RTSP_TLS_VALIDATION_FLAGS_ENV_VAR} must be a non-negative integer"
        ) from error
    if flags < 0:
        raise ValueError(
            f"{_RTSP_TLS_VALIDATION_FLAGS_ENV_VAR} must be a non-negative integer"
        )
    return f" tls-validation-flags={flags}"


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
        self._grabbed_frame_info = None
        self._current_frame_timestamp: Optional[datetime] = None
        self._file_timestamp_origin: Optional[datetime] = None
        self._stream_clock_origin_ns: Optional[int] = None
        self._stream_wall_origin_ns: Optional[int] = None
        self._last_stream_clock_ns: Optional[int] = None
        self._grab_timeout_ns = _resolve_grab_timeout_ns()
        del pin_host_memory

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
        self._native_pipeline = NativeJetsonTensorPipeline(
            self._pipeline, device_id=device_id
        )
        self._closed = False
        self._eos = False

    @property
    def pipeline(self) -> str:
        """Pipeline string exposed for diagnostics and on-device tests."""

        return self._pipeline

    @property
    def has_native_latest_frame_handoff(self) -> bool:
        """Report that live native frames use a latest-wins handoff slot."""

        return True

    def isOpened(self) -> bool:
        return not self._closed and not self._eos

    def grab(self) -> bool:
        if self._closed or self._eos:
            return False
        if self._prerolled_frame_pending:
            self._prerolled_frame_pending = False
            return True
        grabbed = self._native_pipeline.grab(timeout_ns=self._grab_timeout_ns)
        if grabbed:
            if not self._decoder_validated:
                self._validate_hardware_decoder()
            self._record_grabbed_frame_info(self._native_pipeline.frame_info())
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
        frame_info = self._grabbed_frame_info or self._native_pipeline.frame_info()
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
        self._file_timestamp_origin = timestamp_created
        self._current_frame_timestamp = self._resolve_frame_timestamp(frame_info)
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

    def frame_timestamp(self) -> Optional[datetime]:
        """Return timing for the frame selected by the latest successful grab."""

        return self._current_frame_timestamp

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

    def _record_grabbed_frame_info(self, frame_info) -> None:
        """Keep metadata aligned with the exact tensor selected by ``grab``."""

        self._grabbed_frame_info = frame_info
        self._current_frame_timestamp = self._resolve_frame_timestamp(frame_info)

    def _resolve_frame_timestamp(self, frame_info) -> datetime:
        """Map GStreamer stream timing onto the wall-clock timestamp API."""

        pts_ns = int(getattr(frame_info, "pts_ns", -1))
        dts_ns = int(getattr(frame_info, "dts_ns", -1))
        stream_clock_ns = pts_ns if pts_ns >= 0 else dts_ns

        if self._file_timestamp_origin is not None and stream_clock_ns >= 0:
            return self._file_timestamp_origin + timedelta(
                microseconds=stream_clock_ns / 1_000
            )

        arrival_wall_ns = int(getattr(frame_info, "arrival_wall_time_ns", -1))
        if stream_clock_ns >= 0 and arrival_wall_ns >= 0:
            clock_reset = (
                self._last_stream_clock_ns is not None
                and stream_clock_ns < self._last_stream_clock_ns
            )
            if self._stream_clock_origin_ns is None or clock_reset:
                self._stream_clock_origin_ns = stream_clock_ns
                self._stream_wall_origin_ns = arrival_wall_ns
            self._last_stream_clock_ns = stream_clock_ns
            wall_origin_ns = self._stream_wall_origin_ns
            if wall_origin_ns is None:
                wall_origin_ns = arrival_wall_ns
            clock_origin_ns = self._stream_clock_origin_ns
            if clock_origin_ns is None:
                clock_origin_ns = stream_clock_ns
            timestamp_ns = wall_origin_ns + (stream_clock_ns - clock_origin_ns)
            return datetime.fromtimestamp(timestamp_ns / 1_000_000_000)

        if arrival_wall_ns >= 0:
            return datetime.fromtimestamp(arrival_wall_ns / 1_000_000_000)
        return datetime.now()


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
        isinstance(video, str)
        and (video.startswith("/dev/video") or video.lower().startswith("v4l2://"))
    )


def _v4l2_device(video: Union[str, int]) -> str:
    """Normalize direct and URI-form V4L2 references to a device path."""

    if isinstance(video, int):
        if video < 0:
            raise ValueError(f"Invalid V4L2 device index: {video}")
        return f"/dev/video{video}"

    if not video.lower().startswith("v4l2://"):
        return video

    parsed = urlparse(video)
    device = unquote(parsed.path or parsed.netloc)
    if device.isdigit():
        return f"/dev/video{device}"
    if device.startswith("video") and device[5:].isdigit():
        return f"/dev/{device}"
    if device.startswith("/dev/video"):
        return device
    raise ValueError(
        "V4L2 URIs must identify a numeric device or /dev/video path, " f"got {video!r}"
    )


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
