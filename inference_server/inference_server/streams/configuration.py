"""Builds the `streamvision` runtime configuration from the server settings.

`inference_server.configuration` reads the stream settings under the legacy
names; this module only maps them onto `StreamsConfiguration`. The two
tensor-dependent defaults (decoding buffer size and adaptive backpressure)
resolve against the Workflows tensor-representation flag the server already
reads. Install the configuration before importing any stream runtime module:
`streamvision.stream.environment` freezes it into module constants on import.
"""

from streamvision.stream.configuration import (
    ModelConfigDefaults,
    StreamsConfiguration,
    configure_process,
)

from inference_server import configuration
from inference_server.workflows import host as workflows_host


def build_streams_configuration() -> StreamsConfiguration:
    """Map the server settings onto a `StreamsConfiguration`.

    Returns:
        The configuration carrying the legacy-named settings and defaults.
    """
    tensor_representation_enabled = (
        workflows_host.SERVER_WORKFLOWS_CONFIGURATION.tensor.representation_enabled
    )
    buffer_size = configuration.VIDEO_SOURCE_BUFFER_SIZE
    if buffer_size is None:
        buffer_size = (
            configuration.VIDEO_SOURCE_BUFFER_SIZE_TENSOR_DEFAULT
            if tensor_representation_enabled
            else configuration.VIDEO_SOURCE_BUFFER_SIZE_DEFAULT
        )
    adaptive_mode_backpressure = configuration.VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE
    if adaptive_mode_backpressure is None:
        adaptive_mode_backpressure = tensor_representation_enabled

    streams_configuration = StreamsConfiguration(
        default_buffer_size=buffer_size,
        default_adaptive_mode_backpressure=adaptive_mode_backpressure,
        default_adaptive_mode_reader_pace_tolerance=configuration.VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE,
        default_adaptive_mode_stream_pace_tolerance=configuration.VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE,
        default_maximum_adaptive_frames_dropped_in_row=configuration.VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW,
        default_minimum_adaptive_mode_samples=configuration.VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES,
        disable_gstreamer_video_sources=configuration.DISABLE_GSTREAMER_VIDEO_SOURCES,
        disable_native_stderr_capture=configuration.DISABLE_NATIVE_STDERR_CAPTURE,
        restart_attempt_delay=configuration.INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY,
        runs_on_jetson=configuration.RUNS_ON_JETSON,
        enable_frame_drop_on_video_file_rate_limiting=configuration.ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING,
        enable_tensor_data_representation=tensor_representation_enabled,
        enable_workflows_profiling=configuration.ENABLE_WORKFLOWS_PROFILING,
        workflows_profiler_buffer_size=configuration.WORKFLOWS_PROFILER_BUFFER_SIZE,
        predictions_queue_size=configuration.INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE,
        predictions_queue_size_explicit=configuration.INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE_EXPLICIT,
        stream_manager_max_active_pipelines=configuration.STREAM_MANAGER_MAX_ACTIVE_PIPELINES,
        stream_manager_max_ram_mb=configuration.STREAM_MANAGER_MAX_RAM_MB,
        stream_manager_ram_usage_queue_size=configuration.STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE,
        stream_manager_host=configuration.STREAM_MANAGER_HOST,
        stream_manager_port=configuration.STREAM_MANAGER_PORT,
        stream_manager_socket_timeout=configuration.STREAM_MANAGER_SOCKET_TIMEOUT,
        allow_unsafe_gstreamer_pipelines=configuration.ALLOW_UNSAFE_GSTREAMER_PIPELINES,
        debug_aiortc_queues=configuration.DEBUG_AIORTC_QUEUES,
        debug_webrtc_processing_latency=configuration.DEBUG_WEBRTC_PROCESSING_LATENCY,
        offline_mode=configuration.OFFLINE_MODE,
        webrtc_realtime_processing=configuration.WEBRTC_REALTIME_PROCESSING,
        model_config_defaults=ModelConfigDefaults(
            class_agnostic_nms=configuration.CLASS_AGNOSTIC_NMS,
            confidence=configuration.DEFAULT_CONFIDENCE,
            iou_threshold=configuration.DEFAULT_IOU_THRESHOLD,
            max_candidates=configuration.DEFAULT_MAX_CANDIDATES,
            max_detections=configuration.DEFAULT_MAX_DETECTIONS,
        ),
    )

    return streams_configuration


def install_streams_configuration() -> StreamsConfiguration:
    """Install the server's stream configuration in this process.

    Idempotent by value: installing an equal configuration again is a no-op.

    Returns:
        The installed configuration.

    Raises:
        StreamsConfigurationError: A differing configuration is already
            installed in this process.
    """
    streams_configuration = build_streams_configuration()
    configure_process(streams_configuration)

    return streams_configuration
