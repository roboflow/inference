"""The installed `StreamsConfiguration`, bound as module constants.

The stream runtime's replacement for `inference.core.env`: the same UPPER_CASE
names, types and values, sourced from the configuration the host installed
with `streamvision.stream.configuration.configure_process`, never
from `os.environ`.

The values are bound once, at this module's import, exactly as `env.py` binds
its own. Importing this module therefore freezes the process configuration.
Modules that must observe a value at call time read it as an attribute of this
module rather than importing the name.
"""

from streamvision.stream.configuration import get_configuration

_CONFIGURATION = get_configuration()

# --- camera / buffering ---
DEFAULT_BUFFER_SIZE = _CONFIGURATION.default_buffer_size
DEFAULT_ADAPTIVE_MODE_BACKPRESSURE = _CONFIGURATION.default_adaptive_mode_backpressure
DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE = (
    _CONFIGURATION.default_adaptive_mode_reader_pace_tolerance
)
DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE = (
    _CONFIGURATION.default_adaptive_mode_stream_pace_tolerance
)
DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW = (
    _CONFIGURATION.default_maximum_adaptive_frames_dropped_in_row
)
DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES = (
    _CONFIGURATION.default_minimum_adaptive_mode_samples
)
DISABLE_GSTREAMER_VIDEO_SOURCES = _CONFIGURATION.disable_gstreamer_video_sources
DISABLE_NATIVE_STDERR_CAPTURE = _CONFIGURATION.disable_native_stderr_capture
RESTART_ATTEMPT_DELAY = _CONFIGURATION.restart_attempt_delay
RUNS_ON_JETSON = _CONFIGURATION.runs_on_jetson
VIDEO_SOURCE_ALLOW_CPU_FALLBACK = _CONFIGURATION.video_source_allow_cpu_fallback

# --- runtime / manager ---
ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING = (
    _CONFIGURATION.enable_frame_drop_on_video_file_rate_limiting
)
ENABLE_TENSOR_DATA_REPRESENTATION = _CONFIGURATION.enable_tensor_data_representation
ENABLE_WORKFLOWS_PROFILING = _CONFIGURATION.enable_workflows_profiling
WORKFLOWS_PROFILER_BUFFER_SIZE = _CONFIGURATION.workflows_profiler_buffer_size
PREDICTIONS_QUEUE_SIZE = _CONFIGURATION.predictions_queue_size
PREDICTIONS_QUEUE_SIZE_EXPLICIT = _CONFIGURATION.predictions_queue_size_explicit
STREAM_MANAGER_MAX_ACTIVE_PIPELINES = _CONFIGURATION.stream_manager_max_active_pipelines
STREAM_MANAGER_MAX_RAM_MB = _CONFIGURATION.stream_manager_max_ram_mb
STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE = _CONFIGURATION.stream_manager_ram_usage_queue_size
# No `env.py` counterpart; `None` means "no host override" (resolved late by app.py).
STREAM_MANAGER_HOST = _CONFIGURATION.stream_manager_host
STREAM_MANAGER_PORT = _CONFIGURATION.stream_manager_port
STREAM_MANAGER_SOCKET_TIMEOUT = _CONFIGURATION.stream_manager_socket_timeout

# --- request policy / WebRTC ---
ALLOW_UNSAFE_GSTREAMER_PIPELINES = _CONFIGURATION.allow_unsafe_gstreamer_pipelines
DEBUG_AIORTC_QUEUES = _CONFIGURATION.debug_aiortc_queues
DEBUG_WEBRTC_PROCESSING_LATENCY = _CONFIGURATION.debug_webrtc_processing_latency
OFFLINE_MODE = _CONFIGURATION.offline_mode
WEBRTC_REALTIME_PROCESSING = _CONFIGURATION.webrtc_realtime_processing
WEBRTC_DATA_CHANNEL_ACK_WINDOW = _CONFIGURATION.webrtc_data_channel_ack_window
WEBRTC_DATA_CHANNEL_BUFFER_DRAINING_DELAY = (
    _CONFIGURATION.webrtc_data_channel_buffer_draining_delay
)
WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT = (
    _CONFIGURATION.webrtc_data_channel_buffer_size_limit
)
WEBRTC_GZIP_PREVIEW_FRAME_COMPRESSION = (
    _CONFIGURATION.webrtc_gzip_preview_frame_compression
)
WEBRTC_MJPEG_ALLOW_NON_GLOBAL_ADDRESSES = (
    _CONFIGURATION.webrtc_mjpeg_allow_non_global_addresses
)
WEBRTC_MODAL_FUNCTION_TIME_LIMIT = _CONFIGURATION.webrtc_modal_function_time_limit
WEBRTC_MODAL_PUBLIC_STUN_SERVERS = _CONFIGURATION.webrtc_modal_public_stun_servers
WEBRTC_MODAL_RTSP_PLACEHOLDER = _CONFIGURATION.webrtc_modal_rtsp_placeholder
WEBRTC_MODAL_RTSP_PLACEHOLDER_URL = _CONFIGURATION.webrtc_modal_rtsp_placeholder_url
WEBRTC_MODAL_SHUTDOWN_RESERVE = _CONFIGURATION.webrtc_modal_shutdown_reserve
WEBRTC_PREVIEW_FRAME_JPEG_QUALITY = _CONFIGURATION.webrtc_preview_frame_jpeg_quality
WEBRTC_MODAL_MIN_CPU_CORES = _CONFIGURATION.webrtc_modal_min_cpu_cores
WEBRTC_MODAL_MIN_RAM_MB = _CONFIGURATION.webrtc_modal_min_ram_mb
WEBRTC_MODAL_USAGE_QUOTA_ENABLED = _CONFIGURATION.webrtc_modal_usage_quota_enabled
WEBRTC_MODAL_WATCHDOG_TIMEMOUT = _CONFIGURATION.webrtc_modal_watchdog_timemout
WEBRTC_SESSION_HEARTBEAT_INTERVAL_SECONDS = (
    _CONFIGURATION.webrtc_session_heartbeat_interval_seconds
)
WEBRTC_SESSION_HEARTBEAT_URL = _CONFIGURATION.webrtc_session_heartbeat_url

# --- ModelConfig compatibility defaults and env-variable names ---
CLASS_AGNOSTIC_NMS_ENV = _CONFIGURATION.model_config_defaults.class_agnostic_nms_env
CONFIDENCE_ENV = _CONFIGURATION.model_config_defaults.confidence_env
IOU_THRESHOLD_ENV = _CONFIGURATION.model_config_defaults.iou_threshold_env
MAX_CANDIDATES_ENV = _CONFIGURATION.model_config_defaults.max_candidates_env
MAX_DETECTIONS_ENV = _CONFIGURATION.model_config_defaults.max_detections_env
DEFAULT_CLASS_AGNOSTIC_NMS = _CONFIGURATION.model_config_defaults.class_agnostic_nms
DEFAULT_CONFIDENCE = _CONFIGURATION.model_config_defaults.confidence
DEFAULT_IOU_THRESHOLD = _CONFIGURATION.model_config_defaults.iou_threshold
DEFAULT_MAX_CANDIDATES = _CONFIGURATION.model_config_defaults.max_candidates
DEFAULT_MAX_DETECTIONS = _CONFIGURATION.model_config_defaults.max_detections
