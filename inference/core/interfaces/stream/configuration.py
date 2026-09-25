"""Process-level configuration of the camera, stream and stream-manager runtime.

The host builds one `StreamsConfiguration` and installs it with
`configure_process` before anything imports
`inference.core.interfaces.stream.environment`, the constants facade the
runtime modules read. The lifecycle mirrors the Workflows configuration
(`roboflow_workflows.configuration`): set-once by value, with a sticky
standalone fallback, so a value already frozen into a module constant can
never silently disagree with a later installation.

This module must stay import-light: no model, Workflows-engine, video or
server imports. API keys and model-manager settings are host concerns and are
deliberately absent.
"""

import threading
from dataclasses import dataclass, field, fields
from typing import List, Optional


@dataclass(frozen=True)
class ModelConfigDefaults:
    """Environment variable names and fallbacks read by `ModelConfig.init`.

    `ModelConfig.init` parses these variables when it is called, not when the
    configuration is built, so only the names and fallback values live here.
    """

    class_agnostic_nms_env: str = "CLASS_AGNOSTIC_NMS"
    confidence_env: str = "CONFIDENCE"
    iou_threshold_env: str = "IOU_THRESHOLD"
    max_candidates_env: str = "MAX_CANDIDATES"
    max_detections_env: str = "MAX_DETECTIONS"
    class_agnostic_nms: bool = False
    confidence: float = 0.4
    iou_threshold: float = 0.3
    max_candidates: int = 3000
    max_detections: int = 300


@dataclass(frozen=True)
class StreamsConfiguration:
    """Settings of the camera, stream and stream-manager runtime.

    Defaults are the standalone values: the numpy (non-tensor) data path and
    the restrictive request policy. A host with its own environment handling
    installs a configuration carrying its resolved values instead.
    """

    # camera / buffering
    default_buffer_size: int = 64
    default_adaptive_mode_backpressure: bool = False
    default_adaptive_mode_reader_pace_tolerance: float = 5.0
    default_adaptive_mode_stream_pace_tolerance: float = 0.1
    default_maximum_adaptive_frames_dropped_in_row: int = 16
    default_minimum_adaptive_mode_samples: int = 10
    disable_gstreamer_video_sources: bool = False
    disable_native_stderr_capture: bool = False
    restart_attempt_delay: int = 1
    runs_on_jetson: bool = False
    # runtime / manager
    enable_frame_drop_on_video_file_rate_limiting: bool = False
    enable_tensor_data_representation: bool = False
    enable_workflows_profiling: bool = False
    workflows_profiler_buffer_size: int = 64
    predictions_queue_size: int = 512
    # Whether the host set the predictions-queue size explicitly - the tensor
    # pipeline only caps an omitted size, never an explicit one, even when the
    # explicit value equals the default.
    predictions_queue_size_explicit: bool = False
    stream_manager_max_active_pipelines: int = 8
    stream_manager_max_ram_mb: Optional[float] = None
    stream_manager_ram_usage_queue_size: int = 10
    # Standalone defaults, matching the historical `os.getenv` fallbacks. A
    # host that resolves these itself (see `streams_configuration.py`) passes
    # `None` explicitly to defer resolution to
    # `manager_app/app.py`'s own import instead of overriding it here.
    stream_manager_host: Optional[str] = "127.0.0.1"
    stream_manager_port: Optional[int] = 7070
    stream_manager_socket_timeout: Optional[float] = 5.0
    # request policy / WebRTC
    allow_unsafe_gstreamer_pipelines: bool = False
    debug_aiortc_queues: bool = False
    debug_webrtc_processing_latency: bool = False
    offline_mode: bool = False
    webrtc_realtime_processing: bool = True
    model_config_defaults: ModelConfigDefaults = field(
        default_factory=ModelConfigDefaults
    )


class StreamsConfigurationError(RuntimeError):
    """A differing configuration was installed after one was already in use."""


def describe_configuration_difference(
    installed: StreamsConfiguration,
    candidate: StreamsConfiguration,
) -> List[str]:
    """List the fields in which two configurations differ.

    Args:
        installed: The configuration already in effect.
        candidate: The configuration being compared against it.

    Returns:
        One `'field: <installed> != <candidate>'` entry per differing field,
        empty when the configurations are equal.
    """
    differences = []
    for member in fields(StreamsConfiguration):
        left = getattr(installed, member.name)
        right = getattr(candidate, member.name)
        if left != right:
            differences.append(f"{member.name}: {left!r} != {right!r}")

    return differences


_CONFIGURATION: Optional[StreamsConfiguration] = None
_INSTALL_LOCK = threading.RLock()


def configure_process(configuration: StreamsConfiguration) -> None:
    """Install the process-wide configuration. Set-once, by value.

    Installing the identical or an equal configuration again is a no-op.

    Args:
        configuration: The configuration every stream runtime module reads.

    Raises:
        StreamsConfigurationError: A differing configuration is already
            installed, or the standalone default was already read.
    """
    global _CONFIGURATION
    with _INSTALL_LOCK:
        if _CONFIGURATION is not None and _CONFIGURATION != configuration:
            differences = describe_configuration_difference(
                _CONFIGURATION, configuration
            )
            raise StreamsConfigurationError(
                "A different StreamsConfiguration is already installed in this "
                "process. Stream configuration is process-level and frozen into "
                "module constants at import; install it once, before importing "
                f"any stream runtime module. Differences: {differences}"
            )
        _CONFIGURATION = configuration


def get_configuration() -> StreamsConfiguration:
    """Return the installed configuration, falling back to the default.

    The fallback is sticky: once read, it becomes the process configuration,
    so a later differing `configure_process` raises instead of disagreeing with
    values already frozen into module constants.

    Returns:
        The process configuration.
    """
    global _CONFIGURATION
    with _INSTALL_LOCK:
        if _CONFIGURATION is None:
            _CONFIGURATION = StreamsConfiguration()

        return _CONFIGURATION


def reset_configuration() -> None:
    """Test-only hook: forget the installed configuration."""
    global _CONFIGURATION
    with _INSTALL_LOCK:
        _CONFIGURATION = None
