"""Builds the stream runtime configuration from the server's resolved settings.

Mirrors `inference.core.interfaces.workflows_configuration`: `env.py` does the
parsing (including the tensor-dependent buffer and backpressure defaults,
`env.py:1684` and `:1734`); this module ONLY copies the resolved attributes.
The three stream-manager address settings have no `env.py` counterpart and are
left unset (`None`) here on purpose: parsing `STREAM_MANAGER_PORT` /
`STREAM_MANAGER_SOCKET_TIMEOUT` this early would run on every import of
`inference.core` (including through `inference.core.exceptions`, ahead of
`env.py`'s own bootstrap), and `inference.core.interfaces.stream.environment`
is imported far too early too - by the camera and pipeline modules, for
settings that have nothing to do with the manager - to parse them either.
They keep the historical timing instead: `manager_app/app.py` resolves them
itself, with the same `os.getenv` expressions it always used, at its own
import - the only place that actually needs them.

`install_streams_configuration()` is called from `inference/core/__init__.py`
after `env.py` has completed and before anything imports the constants facade
(`inference.core.interfaces.stream.environment`). It also installs
`LEGACY_PIPELINE_HOST_DESCRIPTOR` as the process default of the stream
manager's pipeline host, so `manager_app.app.start()` called without a
descriptor keeps running pipelines with the `inference` host.
"""

import os
from typing import Optional

from inference.core import env
from inference.core.interfaces.stream.configuration import (
    ModelConfigDefaults,
    StreamsConfiguration,
    configure_process,
)
from inference.core.interfaces.stream_manager.manager_app.host import (
    PipelineHostDescriptor,
    install_default_host_descriptor,
)

# Referenced by path: the host module imports the models stack, which this
# bootstrap-time module must not.
LEGACY_PIPELINE_HOST_DESCRIPTOR = PipelineHostDescriptor(
    factory="inference.core.interfaces.legacy_stream.host:LegacyPipelineHost",
)

_SERVER_CONFIGURATION: Optional[StreamsConfiguration] = None


def build_configuration_from_env() -> StreamsConfiguration:
    return StreamsConfiguration(
        default_buffer_size=env.DEFAULT_BUFFER_SIZE,
        default_adaptive_mode_backpressure=env.DEFAULT_ADAPTIVE_MODE_BACKPRESSURE,
        default_adaptive_mode_reader_pace_tolerance=env.DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE,
        default_adaptive_mode_stream_pace_tolerance=env.DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE,
        default_maximum_adaptive_frames_dropped_in_row=env.DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW,
        default_minimum_adaptive_mode_samples=env.DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES,
        disable_gstreamer_video_sources=env.DISABLE_GSTREAMER_VIDEO_SOURCES,
        disable_native_stderr_capture=env.DISABLE_NATIVE_STDERR_CAPTURE,
        restart_attempt_delay=env.RESTART_ATTEMPT_DELAY,
        runs_on_jetson=env.RUNS_ON_JETSON,
        enable_frame_drop_on_video_file_rate_limiting=env.ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING,
        enable_tensor_data_representation=env.ENABLE_TENSOR_DATA_REPRESENTATION,
        enable_workflows_profiling=env.ENABLE_WORKFLOWS_PROFILING,
        workflows_profiler_buffer_size=env.WORKFLOWS_PROFILER_BUFFER_SIZE,
        predictions_queue_size=env.PREDICTIONS_QUEUE_SIZE,
        predictions_queue_size_explicit=(
            "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE" in os.environ
        ),
        stream_manager_max_active_pipelines=env.STREAM_MANAGER_MAX_ACTIVE_PIPELINES,
        stream_manager_max_ram_mb=env.STREAM_MANAGER_MAX_RAM_MB,
        stream_manager_ram_usage_queue_size=env.STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE,
        # Left unset: see the module docstring. `manager_app/app.py` resolves
        # them itself, at its own import, the first time it needs them.
        stream_manager_host=None,
        stream_manager_port=None,
        stream_manager_socket_timeout=None,
        allow_unsafe_gstreamer_pipelines=env.ALLOW_UNSAFE_GSTREAMER_PIPELINES,
        debug_aiortc_queues=env.DEBUG_AIORTC_QUEUES,
        debug_webrtc_processing_latency=env.DEBUG_WEBRTC_PROCESSING_LATENCY,
        offline_mode=env.OFFLINE_MODE,
        webrtc_realtime_processing=env.WEBRTC_REALTIME_PROCESSING,
        model_config_defaults=ModelConfigDefaults(
            class_agnostic_nms_env=env.CLASS_AGNOSTIC_NMS_ENV,
            confidence_env=env.CONFIDENCE_ENV,
            iou_threshold_env=env.IOU_THRESHOLD_ENV,
            max_candidates_env=env.MAX_CANDIDATES_ENV,
            max_detections_env=env.MAX_DETECTIONS_ENV,
            class_agnostic_nms=env.DEFAULT_CLASS_AGNOSTIC_NMS,
            confidence=env.DEFAULT_CONFIDENCE,
            iou_threshold=env.DEFAULT_IOU_THRESHOLD,
            max_candidates=env.DEFAULT_MAX_CANDIDATES,
            max_detections=env.DEFAULT_MAX_DETECTIONS,
        ),
    )


def server_streams_configuration() -> StreamsConfiguration:
    """The one configuration this process uses. Built once, reused thereafter."""
    global _SERVER_CONFIGURATION
    if _SERVER_CONFIGURATION is None:
        _SERVER_CONFIGURATION = build_configuration_from_env()
    return _SERVER_CONFIGURATION


def install_streams_configuration() -> None:
    """Hand the configuration and pipeline host to the stream runtime.

    Idempotent by value.
    """
    configure_process(server_streams_configuration())
    install_default_host_descriptor(LEGACY_PIPELINE_HOST_DESCRIPTOR)
