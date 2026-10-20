import json
import os
import subprocess
import sys
from dataclasses import replace
from functools import lru_cache

import pytest
from streamvision.stream.configuration import (
    StreamsConfigurationError,
    configure_process,
    get_configuration,
    reset_configuration,
)

STREAM_ENV_NAMES = (
    "ENABLE_TENSOR_DATA_REPRESENTATION",
    "ENABLE_WORKFLOWS_PROFILING",
    "WORKFLOWS_PROFILER_BUFFER_SIZE",
    "OFFLINE_MODE",
    "STREAM_MANAGER_HOST",
    "STREAM_MANAGER_PORT",
    "STREAM_MANAGER_SOCKET_TIMEOUT",
    "STREAM_MANAGER_MAX_ACTIVE_PIPELINES",
    "STREAM_MANAGER_MAX_RAM_MB",
    "STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE",
    "STREAM_API_PRELOADED_PROCESSES",
    "VIDEO_SOURCE_BUFFER_SIZE",
    "VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE",
    "VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE",
    "VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE",
    "VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW",
    "VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES",
    "DISABLE_GSTREAMER_VIDEO_SOURCES",
    "DISABLE_NATIVE_STDERR_CAPTURE",
    "INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY",
    "RUNS_ON_JETSON",
    "RUNNING_ON_JETSON",
    "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING",
    "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE",
    "ALLOW_UNSAFE_GSTREAMER_PIPELINES",
    "DEBUG_AIORTC_QUEUES",
    "DEBUG_WEBRTC_PROCESSING_LATENCY",
    "WEBRTC_REALTIME_PROCESSING",
    "CLASS_AGNOSTIC_NMS",
)

EXPLICIT_VALUES = {
    "STREAM_MANAGER_HOST": "0.0.0.0",
    "STREAM_MANAGER_PORT": "7171",
    "STREAM_MANAGER_SOCKET_TIMEOUT": "2.5",
    "STREAM_MANAGER_MAX_ACTIVE_PIPELINES": "3",
    "STREAM_MANAGER_MAX_RAM_MB": "-2048",
    "STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE": "7",
    "VIDEO_SOURCE_BUFFER_SIZE": "17",
    "VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE": "True",
    "VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE": "6.5",
    "VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE": "0.25",
    "VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW": "21",
    "VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES": "13",
    "DISABLE_GSTREAMER_VIDEO_SOURCES": "True",
    "DISABLE_NATIVE_STDERR_CAPTURE": "True",
    "INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY": "4",
    "RUNS_ON_JETSON": "True",
    "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING": "True",
    "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE": "99",
    "ALLOW_UNSAFE_GSTREAMER_PIPELINES": "True",
    "DEBUG_AIORTC_QUEUES": "True",
    "DEBUG_WEBRTC_PROCESSING_LATENCY": "True",
    "WEBRTC_REALTIME_PROCESSING": "False",
    "ENABLE_WORKFLOWS_PROFILING": "True",
    "WORKFLOWS_PROFILER_BUFFER_SIZE": "12",
    "CLASS_AGNOSTIC_NMS": "True",
}

EXPECTED_FIELDS = [
    ("STREAM_MANAGER_HOST", "stream_manager_host", "0.0.0.0"),
    ("STREAM_MANAGER_PORT", "stream_manager_port", 7171),
    ("STREAM_MANAGER_SOCKET_TIMEOUT", "stream_manager_socket_timeout", 2.5),
    (
        "STREAM_MANAGER_MAX_ACTIVE_PIPELINES",
        "stream_manager_max_active_pipelines",
        3,
    ),
    ("STREAM_MANAGER_MAX_RAM_MB", "stream_manager_max_ram_mb", 2048.0),
    (
        "STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE",
        "stream_manager_ram_usage_queue_size",
        7,
    ),
    ("VIDEO_SOURCE_BUFFER_SIZE", "default_buffer_size", 17),
    (
        "VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE",
        "default_adaptive_mode_backpressure",
        True,
    ),
    (
        "VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE",
        "default_adaptive_mode_reader_pace_tolerance",
        6.5,
    ),
    (
        "VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE",
        "default_adaptive_mode_stream_pace_tolerance",
        0.25,
    ),
    (
        "VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW",
        "default_maximum_adaptive_frames_dropped_in_row",
        21,
    ),
    (
        "VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES",
        "default_minimum_adaptive_mode_samples",
        13,
    ),
    ("DISABLE_GSTREAMER_VIDEO_SOURCES", "disable_gstreamer_video_sources", True),
    ("DISABLE_NATIVE_STDERR_CAPTURE", "disable_native_stderr_capture", True),
    ("INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY", "restart_attempt_delay", 4),
    ("RUNS_ON_JETSON", "runs_on_jetson", True),
    (
        "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING",
        "enable_frame_drop_on_video_file_rate_limiting",
        True,
    ),
    ("INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE", "predictions_queue_size", 99),
    (
        "ALLOW_UNSAFE_GSTREAMER_PIPELINES",
        "allow_unsafe_gstreamer_pipelines",
        True,
    ),
    ("DEBUG_AIORTC_QUEUES", "debug_aiortc_queues", True),
    ("DEBUG_WEBRTC_PROCESSING_LATENCY", "debug_webrtc_processing_latency", True),
    ("WEBRTC_REALTIME_PROCESSING", "webrtc_realtime_processing", False),
    ("ENABLE_WORKFLOWS_PROFILING", "enable_workflows_profiling", True),
    ("WORKFLOWS_PROFILER_BUFFER_SIZE", "workflows_profiler_buffer_size", 12),
]

_DUMP_CODE = (
    "import json; from dataclasses import asdict; "
    "from inference_server.streams.configuration import build_streams_configuration; "
    "print(json.dumps(asdict(build_streams_configuration())))"
)


@lru_cache(maxsize=None)
def _built_configuration(overrides: tuple) -> dict:
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in STREAM_ENV_NAMES
    }
    env.update(dict(overrides))
    result = subprocess.run(
        [sys.executable, "-c", _DUMP_CODE], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    built = json.loads(result.stdout.strip().splitlines()[-1])

    return built


@pytest.mark.parametrize("name, field, expected", EXPECTED_FIELDS)
def test_build_streams_configuration_maps_the_legacy_names(name, field, expected):
    built = _built_configuration(tuple(sorted(EXPLICIT_VALUES.items())))

    assert built[field] == expected, name


def test_build_streams_configuration_defaults_match_legacy():
    built = _built_configuration(())

    assert built["stream_manager_host"] == "127.0.0.1"
    assert built["stream_manager_port"] == 7070
    assert built["stream_manager_socket_timeout"] == 5.0
    assert built["stream_manager_max_active_pipelines"] == 8
    assert built["stream_manager_max_ram_mb"] is None
    assert built["stream_manager_ram_usage_queue_size"] == 10
    assert built["default_buffer_size"] == 64
    assert built["default_adaptive_mode_backpressure"] is False
    assert built["predictions_queue_size"] == 512
    assert built["predictions_queue_size_explicit"] is False
    assert built["restart_attempt_delay"] == 1
    assert built["webrtc_realtime_processing"] is True
    assert built["enable_tensor_data_representation"] is False
    assert built["offline_mode"] is False
    assert built["model_config_defaults"] == {
        "class_agnostic_nms_env": "CLASS_AGNOSTIC_NMS",
        "confidence_env": "CONFIDENCE",
        "iou_threshold_env": "IOU_THRESHOLD",
        "max_candidates_env": "MAX_CANDIDATES",
        "max_detections_env": "MAX_DETECTIONS",
        "class_agnostic_nms": False,
        "confidence": 0.4,
        "iou_threshold": 0.3,
        "max_candidates": 3000,
        "max_detections": 300,
    }


def test_tensor_representation_changes_the_buffer_and_backpressure_defaults():
    built = _built_configuration((("ENABLE_TENSOR_DATA_REPRESENTATION", "True"),))

    assert built["enable_tensor_data_representation"] is True
    assert built["default_buffer_size"] == 8
    assert built["default_adaptive_mode_backpressure"] is True


def test_explicit_buffer_and_backpressure_win_over_the_tensor_defaults():
    built = _built_configuration(
        (
            ("ENABLE_TENSOR_DATA_REPRESENTATION", "True"),
            ("VIDEO_SOURCE_BUFFER_SIZE", "32"),
            ("VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE", "False"),
        )
    )

    assert built["default_buffer_size"] == 32
    assert built["default_adaptive_mode_backpressure"] is False


def test_predictions_queue_size_is_marked_explicit_only_when_set():
    assert _built_configuration(())["predictions_queue_size_explicit"] is False
    assert (
        _built_configuration((("INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE", "512"),))[
            "predictions_queue_size_explicit"
        ]
        is True
    )


def test_max_active_pipelines_is_never_below_the_preloaded_processes():
    built = _built_configuration(
        (
            ("STREAM_API_PRELOADED_PROCESSES", "5"),
            ("STREAM_MANAGER_MAX_ACTIVE_PIPELINES", "2"),
        )
    )

    assert built["stream_manager_max_active_pipelines"] == 5


def test_offline_mode_and_class_agnostic_nms_are_carried():
    built = _built_configuration(
        (("OFFLINE_MODE", "True"), ("CLASS_AGNOSTIC_NMS", "True"))
    )

    assert built["offline_mode"] is True
    assert built["model_config_defaults"]["class_agnostic_nms"] is True


def test_running_on_jetson_spelling_is_honoured():
    built = _built_configuration((("RUNNING_ON_JETSON", "True"),))

    assert built["runs_on_jetson"] is True


@pytest.fixture
def _fresh_streams_configuration():
    reset_configuration()
    yield
    reset_configuration()


def test_install_streams_configuration_twice_is_a_no_op(
    _fresh_streams_configuration,
):
    from inference_server.streams.configuration import (
        build_streams_configuration,
        install_streams_configuration,
    )

    first = install_streams_configuration()
    second = install_streams_configuration()

    assert first == build_streams_configuration()
    assert second == first
    assert get_configuration() == first


def test_install_streams_configuration_raises_on_a_differing_installed_config(
    _fresh_streams_configuration,
):
    from inference_server.streams.configuration import (
        build_streams_configuration,
        install_streams_configuration,
    )

    built = build_streams_configuration()
    differing = replace(built, default_buffer_size=built.default_buffer_size + 1)
    configure_process(differing)

    with pytest.raises(StreamsConfigurationError):
        install_streams_configuration()
