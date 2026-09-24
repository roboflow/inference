"""WP-A01 characterization of stream/camera/stream_manager configuration.

Section 1 pins what the consumers of `inference.core.env` observe today - the
value, type and binding time of every setting the three trees read - through
the consumer modules themselves, so it holds identically before and after the
reads are rerouted through `inference.core.interfaces.stream.environment`.
Environment-dependent cases run in a fresh interpreter: `env.py`, and every
module-level default derived from it, is bound once per process.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

from inference.core.interfaces.stream.entities import ModelConfig

REPO_ROOT = Path(__file__).resolve().parents[5]

# Every environment variable behind a setting this file pins. The child
# interpreter starts from the parent's environment with all of these removed,
# so an operator's shell cannot leak into an expectation.
_CONTROLLED_VARIABLES = (
    "ALLOW_UNSAFE_GSTREAMER_PIPELINES",
    "DEBUG_AIORTC_QUEUES",
    "DEBUG_WEBRTC_PROCESSING_LATENCY",
    "DISABLE_GSTREAMER_VIDEO_SOURCES",
    "DISABLE_NATIVE_STDERR_CAPTURE",
    "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING",
    "ENABLE_TENSOR_DATA_REPRESENTATION",
    "ENABLE_WORKFLOWS_PROFILING",
    "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE",
    "INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY",
    "OFFLINE_MODE",
    "RUNNING_ON_JETSON",
    "RUNS_ON_JETSON",
    "STREAM_API_PRELOADED_PROCESSES",
    "STREAM_MANAGER_HOST",
    "STREAM_MANAGER_MAX_ACTIVE_PIPELINES",
    "STREAM_MANAGER_MAX_RAM_MB",
    "STREAM_MANAGER_PORT",
    "STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE",
    "STREAM_MANAGER_SOCKET_TIMEOUT",
    "USE_INFERENCE_MODELS",
    "VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE",
    "VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE",
    "VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE",
    "VIDEO_SOURCE_BUFFER_SIZE",
    "VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW",
    "VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES",
    "WEBRTC_REALTIME_PROCESSING",
    "WORKFLOWS_PROFILER_BUFFER_SIZE",
    # inference_models latches OFFLINE_MODE into the environment of the process
    # that imported it (the pytest parent included); a child inheriting the
    # latch would ignore its own OFFLINE_MODE.
    "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START",
    "_ROBOFLOW_INFERENCE_OFFLINE_MODE_STARTUP_ERROR",
)


def _run_child(script: str, overrides: Optional[Dict[str, str]] = None) -> dict:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _CONTROLLED_VARIABLES
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "DISABLE_VERSION_CHECK": "True",
            "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT / "workflows"), str(REPO_ROOT / "inference_models")]
            ),
        }
    )
    environment.update(overrides or {})
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    return result


# --------------------------------------------------------------------------
# Section 1 - consumer-observed behavior
# --------------------------------------------------------------------------

_BUFFER_DEFAULTS_SCRIPT = """
import inspect
import json
import {entry_module}

from inference.core import env
from inference.core.interfaces.camera.video_source import VideoConsumer, VideoSource
from inference.core.interfaces.stream import utils as stream_utils
from inference.core.interfaces.stream_manager.manager_app.entities import (
    InitialisePipelinePayload,
)


def default(function, name):
    return inspect.signature(function).parameters[name].default


print(json.dumps({{
    "env_tensor": env.ENABLE_TENSOR_DATA_REPRESENTATION,
    "video_source_buffer_size": default(VideoSource.init, "buffer_size"),
    "consumer_init_backpressure": default(VideoConsumer.init, "adaptive_backpressure"),
    "consumer_backpressure": default(VideoConsumer.__init__, "adaptive_backpressure"),
    "prepare_video_sources_buffer_size": default(
        stream_utils.prepare_video_sources, "decoding_buffer_size"
    ),
    "initialise_payload_buffer_size": InitialisePipelinePayload.model_fields[
        "decoding_buffer_size"
    ].default,
}}))
"""


@pytest.mark.parametrize(
    "overrides, expected_buffer_size, expected_backpressure",
    [
        ({"ENABLE_TENSOR_DATA_REPRESENTATION": "False"}, 64, False),
        ({"ENABLE_TENSOR_DATA_REPRESENTATION": "True"}, 8, True),
        (
            {
                "ENABLE_TENSOR_DATA_REPRESENTATION": "True",
                "VIDEO_SOURCE_BUFFER_SIZE": "32",
                "VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE": "False",
            },
            32,
            False,
        ),
        (
            {
                "ENABLE_TENSOR_DATA_REPRESENTATION": "False",
                "VIDEO_SOURCE_BUFFER_SIZE": "5",
                "VIDEO_SOURCE_ADAPTIVE_BACKPRESSURE": "True",
            },
            5,
            True,
        ),
    ],
    ids=["numpy-defaults", "tensor-defaults", "tensor-explicit", "numpy-explicit"],
)
@pytest.mark.parametrize("entry_module", ["inference", "inference.core.exceptions"])
def test_buffer_defaults_follow_tensor_mode_and_explicit_overrides(
    overrides: Dict[str, str],
    expected_buffer_size: int,
    expected_backpressure: bool,
    entry_module: str,
) -> None:
    # `inference.core.exceptions` is on env.py's own import path; entering the
    # process through it must still leave the stream defaults bound to the
    # fully resolved env values (tensor-dependent defaults, env.py:1684/:1734).
    result = _run_child(
        _BUFFER_DEFAULTS_SCRIPT.format(entry_module=entry_module),
        {"USE_INFERENCE_MODELS": "True", **overrides},
    )

    expected_tensor = overrides["ENABLE_TENSOR_DATA_REPRESENTATION"] == "True"
    assert result == {
        "env_tensor": expected_tensor,
        "video_source_buffer_size": expected_buffer_size,
        "consumer_init_backpressure": expected_backpressure,
        "consumer_backpressure": expected_backpressure,
        "prepare_video_sources_buffer_size": expected_buffer_size,
        "initialise_payload_buffer_size": expected_buffer_size,
    }


_CONSUMER_BINDINGS_SCRIPT = """
import json

from inference.core.interfaces.camera import stream_error_classifier
from inference.core.interfaces.camera import utils as camera_utils
from inference.core.interfaces.camera import video_source
from inference.core.interfaces.stream import utils as stream_utils
from inference.core.interfaces.stream.model_handlers import workflows as handler
from inference.core.interfaces.stream_manager.manager_app import app, entities, webrtc

bindings = {
    "camera_utils.RESTART_ATTEMPT_DELAY": camera_utils.RESTART_ATTEMPT_DELAY,
    "stream_error_classifier.DISABLE_NATIVE_STDERR_CAPTURE": (
        stream_error_classifier.DISABLE_NATIVE_STDERR_CAPTURE
    ),
    "video_source.DISABLE_GSTREAMER_VIDEO_SOURCES": (
        video_source.DISABLE_GSTREAMER_VIDEO_SOURCES
    ),
    "video_source.ENABLE_TENSOR_DATA_REPRESENTATION": (
        video_source.ENABLE_TENSOR_DATA_REPRESENTATION
    ),
    "video_source.RUNS_ON_JETSON": video_source.RUNS_ON_JETSON,
    "video_source.DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE": (
        video_source.DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE
    ),
    "video_source.DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE": (
        video_source.DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE
    ),
    "video_source.DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES": (
        video_source.DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES
    ),
    "video_source.DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW": (
        video_source.DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW
    ),
    "stream_utils.ENABLE_WORKFLOWS_PROFILING": stream_utils.ENABLE_WORKFLOWS_PROFILING,
    "handler.ENABLE_TENSOR_DATA_REPRESENTATION": (
        handler.ENABLE_TENSOR_DATA_REPRESENTATION
    ),
    "app.STREAM_MANAGER_MAX_ACTIVE_PIPELINES": app.STREAM_MANAGER_MAX_ACTIVE_PIPELINES,
    "app.STREAM_MANAGER_MAX_RAM_MB": app.STREAM_MANAGER_MAX_RAM_MB,
    "app.STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE": app.STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE,
    "app.HOST": app.HOST,
    "app.PORT": app.PORT,
    "app.SOCKET_TIMEOUT": app.SOCKET_TIMEOUT,
    "entities.ALLOW_UNSAFE_GSTREAMER_PIPELINES": (
        entities.ALLOW_UNSAFE_GSTREAMER_PIPELINES
    ),
    "entities.PREDICTIONS_QUEUE_SIZE": entities.PREDICTIONS_QUEUE_SIZE,
    "entities.WEBRTC_REALTIME_PROCESSING": entities.WEBRTC_REALTIME_PROCESSING,
    "webrtc.DEBUG_AIORTC_QUEUES": webrtc.DEBUG_AIORTC_QUEUES,
    "webrtc.DEBUG_WEBRTC_PROCESSING_LATENCY": webrtc.DEBUG_WEBRTC_PROCESSING_LATENCY,
    "webrtc.OFFLINE_MODE": webrtc.OFFLINE_MODE,
}
print(json.dumps({name: [value, type(value).__name__] for name, value in bindings.items()}))
"""

_DEFAULT_BINDINGS = {
    "camera_utils.RESTART_ATTEMPT_DELAY": [1, "int"],
    "stream_error_classifier.DISABLE_NATIVE_STDERR_CAPTURE": [False, "bool"],
    "video_source.DISABLE_GSTREAMER_VIDEO_SOURCES": [False, "bool"],
    "video_source.ENABLE_TENSOR_DATA_REPRESENTATION": [False, "bool"],
    "video_source.RUNS_ON_JETSON": [False, "bool"],
    "video_source.DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE": [0.1, "float"],
    "video_source.DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE": [5.0, "float"],
    "video_source.DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES": [10, "int"],
    "video_source.DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW": [16, "int"],
    "stream_utils.ENABLE_WORKFLOWS_PROFILING": [False, "bool"],
    "handler.ENABLE_TENSOR_DATA_REPRESENTATION": [False, "bool"],
    "app.STREAM_MANAGER_MAX_ACTIVE_PIPELINES": [8, "int"],
    "app.STREAM_MANAGER_MAX_RAM_MB": [None, "NoneType"],
    "app.STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE": [10, "int"],
    "app.HOST": ["127.0.0.1", "str"],
    "app.PORT": [7070, "int"],
    "app.SOCKET_TIMEOUT": [5.0, "float"],
    "entities.ALLOW_UNSAFE_GSTREAMER_PIPELINES": [False, "bool"],
    "entities.PREDICTIONS_QUEUE_SIZE": [512, "int"],
    "entities.WEBRTC_REALTIME_PROCESSING": [True, "bool"],
    "webrtc.DEBUG_AIORTC_QUEUES": [False, "bool"],
    "webrtc.DEBUG_WEBRTC_PROCESSING_LATENCY": [False, "bool"],
    "webrtc.OFFLINE_MODE": [False, "bool"],
}

_OVERRIDDEN_ENVIRONMENT = {
    "INFERENCE_PIPELINE_RESTART_ATTEMPT_DELAY": "3",
    "DISABLE_NATIVE_STDERR_CAPTURE": "True",
    "DISABLE_GSTREAMER_VIDEO_SOURCES": "True",
    "RUNS_ON_JETSON": "True",
    "VIDEO_SOURCE_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE": "0.25",
    "VIDEO_SOURCE_ADAPTIVE_MODE_READER_PACE_TOLERANCE": "2",
    "VIDEO_SOURCE_MINIMUM_ADAPTIVE_MODE_SAMPLES": "4",
    "VIDEO_SOURCE_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW": "7",
    "ENABLE_WORKFLOWS_PROFILING": "True",
    "STREAM_MANAGER_MAX_ACTIVE_PIPELINES": "3",
    "STREAM_MANAGER_MAX_RAM_MB": "-1024",
    "STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE": "-4",
    "STREAM_MANAGER_HOST": "0.0.0.0",
    "STREAM_MANAGER_PORT": "7171",
    "STREAM_MANAGER_SOCKET_TIMEOUT": "2",
    "ALLOW_UNSAFE_GSTREAMER_PIPELINES": "True",
    "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE": "16",
    "WEBRTC_REALTIME_PROCESSING": "False",
    "DEBUG_AIORTC_QUEUES": "True",
    "DEBUG_WEBRTC_PROCESSING_LATENCY": "True",
}

_OVERRIDDEN_BINDINGS = {
    **_DEFAULT_BINDINGS,
    "camera_utils.RESTART_ATTEMPT_DELAY": [3, "int"],
    "stream_error_classifier.DISABLE_NATIVE_STDERR_CAPTURE": [True, "bool"],
    "video_source.DISABLE_GSTREAMER_VIDEO_SOURCES": [True, "bool"],
    "video_source.RUNS_ON_JETSON": [True, "bool"],
    "video_source.DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE": [0.25, "float"],
    "video_source.DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE": [2.0, "float"],
    "video_source.DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES": [4, "int"],
    "video_source.DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW": [7, "int"],
    "stream_utils.ENABLE_WORKFLOWS_PROFILING": [True, "bool"],
    "app.STREAM_MANAGER_MAX_ACTIVE_PIPELINES": [3, "int"],
    "app.STREAM_MANAGER_MAX_RAM_MB": [1024.0, "float"],
    "app.STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE": [4, "int"],
    "app.HOST": ["0.0.0.0", "str"],
    "app.PORT": [7171, "int"],
    "app.SOCKET_TIMEOUT": [2.0, "float"],
    "entities.ALLOW_UNSAFE_GSTREAMER_PIPELINES": [True, "bool"],
    "entities.PREDICTIONS_QUEUE_SIZE": [16, "int"],
    "entities.WEBRTC_REALTIME_PROCESSING": [False, "bool"],
    "webrtc.DEBUG_AIORTC_QUEUES": [True, "bool"],
    "webrtc.DEBUG_WEBRTC_PROCESSING_LATENCY": [True, "bool"],
}


@pytest.mark.parametrize(
    "overrides, expected",
    [({}, _DEFAULT_BINDINGS), (_OVERRIDDEN_ENVIRONMENT, _OVERRIDDEN_BINDINGS)],
    ids=["defaults", "overridden"],
)
def test_consumer_module_bindings_match_environment(
    overrides: Dict[str, str], expected: dict
) -> None:
    result = _run_child(
        _CONSUMER_BINDINGS_SCRIPT,
        {"ENABLE_TENSOR_DATA_REPRESENTATION": "False", **overrides},
    )

    assert result == expected


_UNSAFE_GSTREAMER_SCRIPT = """
import json

from inference.core.interfaces.stream_manager.manager_app.entities import (
    VideoConfiguration,
)

try:
    VideoConfiguration(
        type="VideoConfiguration",
        video_reference="gst-launch-1.0 videotestsrc ! autovideosink",
    )
    accepted = True
except ValueError:
    accepted = False
print(json.dumps({"accepted": accepted}))
"""


@pytest.mark.parametrize("flag, expected", [("False", False), ("True", True)])
def test_unsafe_gstreamer_launch_strings_follow_the_policy_flag(
    flag: str, expected: bool
) -> None:
    result = _run_child(
        _UNSAFE_GSTREAMER_SCRIPT, {"ALLOW_UNSAFE_GSTREAMER_PIPELINES": flag}
    )

    assert result == {"accepted": expected}


_OFFLINE_WEBRTC_SCRIPT = """
import json

from inference.core.exceptions import WebRTCConfigurationError
from inference.core.interfaces.stream_manager.manager_app import webrtc
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCTURNConfig,
)

configuration = webrtc._build_rtc_configuration(webrtc_turn_config=None)
try:
    webrtc._build_rtc_configuration(
        webrtc_turn_config=WebRTCTURNConfig(
            urls="turn:turn.example:3478", username="user", credential="secret"
        )
    )
    turn_rejected = False
except WebRTCConfigurationError:
    turn_rejected = True
print(json.dumps({
    "ice_servers": None if configuration is None else len(configuration.iceServers),
    "turn_rejected": turn_rejected,
}))
"""


@pytest.mark.parametrize(
    "flag, expected",
    [
        ("False", {"ice_servers": None, "turn_rejected": False}),
        ("True", {"ice_servers": 0, "turn_rejected": True}),
    ],
)
def test_offline_mode_keeps_stream_manager_webrtc_local(
    flag: str, expected: dict
) -> None:
    result = _run_child(_OFFLINE_WEBRTC_SCRIPT, {"OFFLINE_MODE": flag})

    assert result == expected


_MODEL_CONFIG_VARIABLES = (
    "CLASS_AGNOSTIC_NMS",
    "CONFIDENCE",
    "IOU_THRESHOLD",
    "MAX_CANDIDATES",
    "MAX_DETECTIONS",
)


@pytest.fixture
def clean_model_config_environment(monkeypatch):
    for name in _MODEL_CONFIG_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    return monkeypatch


def test_model_config_init_falls_back_to_compatibility_defaults(
    clean_model_config_environment,
) -> None:
    config = ModelConfig.init()

    assert (
        config.class_agnostic_nms,
        config.confidence,
        config.iou_threshold,
        config.max_candidates,
        config.max_detections,
    ) == (False, 0.4, 0.3, 3000, 300)


def test_model_config_init_reads_the_environment_at_call_time(
    clean_model_config_environment,
) -> None:
    monkeypatch = clean_model_config_environment
    monkeypatch.setenv("CLASS_AGNOSTIC_NMS", "True")
    monkeypatch.setenv("CONFIDENCE", "0.77")
    monkeypatch.setenv("IOU_THRESHOLD", "0.55")
    monkeypatch.setenv("MAX_CANDIDATES", "11")
    monkeypatch.setenv("MAX_DETECTIONS", "5")

    config = ModelConfig.init()

    assert (
        config.class_agnostic_nms,
        config.confidence,
        config.iou_threshold,
        config.max_candidates,
        config.max_detections,
    ) == (True, 0.77, 0.55, 11, 5)


def test_model_config_init_prefers_explicit_arguments_over_the_environment(
    clean_model_config_environment,
) -> None:
    monkeypatch = clean_model_config_environment
    monkeypatch.setenv("CONFIDENCE", "0.9")
    monkeypatch.setenv("MAX_DETECTIONS", "not-a-number")

    config = ModelConfig.init(confidence=0.1, max_detections=7)

    assert (config.confidence, config.max_detections) == (0.1, 7)


def test_model_config_init_propagates_unparseable_environment_values(
    clean_model_config_environment,
) -> None:
    clean_model_config_environment.setenv("MAX_DETECTIONS", "not-a-number")

    with pytest.raises(ValueError):
        ModelConfig.init()


# --------------------------------------------------------------------------
# Section 2 - the configuration, its facade and the legacy installation
# --------------------------------------------------------------------------

# (facade name, inference.core.env name or None when env.py has no counterpart)
FIELDS = [
    ("DEFAULT_BUFFER_SIZE", "DEFAULT_BUFFER_SIZE"),
    ("DEFAULT_ADAPTIVE_MODE_BACKPRESSURE", "DEFAULT_ADAPTIVE_MODE_BACKPRESSURE"),
    (
        "DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE",
        "DEFAULT_ADAPTIVE_MODE_READER_PACE_TOLERANCE",
    ),
    (
        "DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE",
        "DEFAULT_ADAPTIVE_MODE_STREAM_PACE_TOLERANCE",
    ),
    (
        "DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW",
        "DEFAULT_MAXIMUM_ADAPTIVE_FRAMES_DROPPED_IN_ROW",
    ),
    ("DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES", "DEFAULT_MINIMUM_ADAPTIVE_MODE_SAMPLES"),
    ("DISABLE_GSTREAMER_VIDEO_SOURCES", "DISABLE_GSTREAMER_VIDEO_SOURCES"),
    ("DISABLE_NATIVE_STDERR_CAPTURE", "DISABLE_NATIVE_STDERR_CAPTURE"),
    ("RESTART_ATTEMPT_DELAY", "RESTART_ATTEMPT_DELAY"),
    ("RUNS_ON_JETSON", "RUNS_ON_JETSON"),
    (
        "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING",
        "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING",
    ),
    ("ENABLE_TENSOR_DATA_REPRESENTATION", "ENABLE_TENSOR_DATA_REPRESENTATION"),
    ("ENABLE_WORKFLOWS_PROFILING", "ENABLE_WORKFLOWS_PROFILING"),
    ("WORKFLOWS_PROFILER_BUFFER_SIZE", "WORKFLOWS_PROFILER_BUFFER_SIZE"),
    ("PREDICTIONS_QUEUE_SIZE", "PREDICTIONS_QUEUE_SIZE"),
    ("PREDICTIONS_QUEUE_SIZE_EXPLICIT", None),
    ("STREAM_MANAGER_MAX_ACTIVE_PIPELINES", "STREAM_MANAGER_MAX_ACTIVE_PIPELINES"),
    ("STREAM_MANAGER_MAX_RAM_MB", "STREAM_MANAGER_MAX_RAM_MB"),
    ("STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE", "STREAM_MANAGER_RAM_USAGE_QUEUE_SIZE"),
    ("STREAM_MANAGER_HOST", None),
    ("STREAM_MANAGER_PORT", None),
    ("STREAM_MANAGER_SOCKET_TIMEOUT", None),
    ("ALLOW_UNSAFE_GSTREAMER_PIPELINES", "ALLOW_UNSAFE_GSTREAMER_PIPELINES"),
    ("DEBUG_AIORTC_QUEUES", "DEBUG_AIORTC_QUEUES"),
    ("DEBUG_WEBRTC_PROCESSING_LATENCY", "DEBUG_WEBRTC_PROCESSING_LATENCY"),
    ("OFFLINE_MODE", "OFFLINE_MODE"),
    ("WEBRTC_REALTIME_PROCESSING", "WEBRTC_REALTIME_PROCESSING"),
    ("CLASS_AGNOSTIC_NMS_ENV", "CLASS_AGNOSTIC_NMS_ENV"),
    ("CONFIDENCE_ENV", "CONFIDENCE_ENV"),
    ("IOU_THRESHOLD_ENV", "IOU_THRESHOLD_ENV"),
    ("MAX_CANDIDATES_ENV", "MAX_CANDIDATES_ENV"),
    ("MAX_DETECTIONS_ENV", "MAX_DETECTIONS_ENV"),
    ("DEFAULT_CLASS_AGNOSTIC_NMS", "DEFAULT_CLASS_AGNOSTIC_NMS"),
    ("DEFAULT_CONFIDENCE", "DEFAULT_CONFIDENCE"),
    ("DEFAULT_IOU_THRESHOLD", "DEFAULT_IOU_THRESHOLD"),
    ("DEFAULT_MAX_CANDIDATES", "DEFAULT_MAX_CANDIDATES"),
    ("DEFAULT_MAX_DETECTIONS", "DEFAULT_MAX_DETECTIONS"),
]

# Settings that belong to the host (model construction, credentials, legacy
# Stream-only env) and must never become package configuration.
_HOST_ONLY_NAMES = (
    "ACTIVE_LEARNING_ENABLED",
    "API_KEY",
    "API_KEY_ENV_NAMES",
    "MAX_ACTIVE_MODELS",
    "DISABLE_PREPROC_AUTO_ORIENT",
    "MODEL_ID",
    "STREAM_ID",
    "CONFIDENCE",
    "IOU_THRESHOLD",
    "MAX_CANDIDATES",
    "MAX_DETECTIONS",
    "CLASS_AGNOSTIC_NMS",
    "ENABLE_BYTE_TRACK",
    "ENFORCE_FPS",
    "JSON_RESPONSE",
)


@pytest.fixture
def installed_configuration():
    from inference.core.interfaces.stream import configuration

    previous = configuration._CONFIGURATION
    yield configuration
    configuration._CONFIGURATION = previous


def _facade_exports() -> set:
    from inference.core.interfaces.stream import environment

    return {
        name
        for name in vars(environment)
        if name.isupper() and not name.startswith("_")
    }


def test_the_field_table_matches_the_facade_exports() -> None:
    tabled = {name for name, _ in FIELDS}

    assert tabled == _facade_exports()
    assert len(FIELDS) == 37


def test_host_only_settings_are_not_package_configuration() -> None:
    from dataclasses import fields

    from inference.core.interfaces.stream.configuration import StreamsConfiguration

    field_names = {member.name.upper() for member in fields(StreamsConfiguration)}

    suspicious = {name for name in field_names if "API_KEY" in name or "MODEL" in name}

    assert not (set(_HOST_ONLY_NAMES) & _facade_exports())
    assert suspicious == {"MODEL_CONFIG_DEFAULTS"}


@pytest.mark.parametrize("name, env_name", [f for f in FIELDS if f[1] is not None])
def test_the_facade_equals_env_field_by_field(name: str, env_name: str) -> None:
    from inference.core import env
    from inference.core.interfaces.stream import environment

    expected = getattr(env, env_name)
    actual = getattr(environment, name)

    assert actual == expected
    assert type(actual) is type(expected)


def test_manager_address_settings_match_the_historical_expressions() -> None:
    # The facade itself no longer resolves these - it is imported far too
    # early, by camera and pipeline modules that have nothing to do with the
    # manager. Only `manager_app/app.py`, at its own import, resolves an
    # unset value with the historical expressions, so that is where this
    # must be checked.
    from inference.core.interfaces.stream_manager.manager_app import app

    assert app.HOST == os.getenv("STREAM_MANAGER_HOST", "127.0.0.1")
    assert app.PORT == int(os.getenv("STREAM_MANAGER_PORT", "7070"))
    assert app.SOCKET_TIMEOUT == float(
        os.getenv("STREAM_MANAGER_SOCKET_TIMEOUT", "5.0")
    )


def test_the_bootstrap_installs_the_memoised_server_configuration() -> None:
    from inference.core.interfaces.stream.configuration import get_configuration
    from inference.core.interfaces.streams_configuration import (
        build_configuration_from_env,
        install_streams_configuration,
        server_streams_configuration,
    )

    install_streams_configuration()

    assert server_streams_configuration() is server_streams_configuration()
    assert get_configuration() is server_streams_configuration()
    assert get_configuration() == build_configuration_from_env()


def test_same_value_reconfiguration_is_a_no_op(installed_configuration) -> None:
    import dataclasses

    current = installed_configuration.get_configuration()
    equal_copy = dataclasses.replace(current)

    installed_configuration.configure_process(current)
    installed_configuration.configure_process(equal_copy)

    assert installed_configuration.get_configuration() is equal_copy


def test_conflicting_reconfiguration_raises_and_names_the_field(
    installed_configuration,
) -> None:
    import dataclasses

    current = installed_configuration.get_configuration()
    conflicting = dataclasses.replace(
        current, default_buffer_size=current.default_buffer_size + 1
    )

    with pytest.raises(
        installed_configuration.StreamsConfigurationError,
        match="default_buffer_size",
    ):
        installed_configuration.configure_process(conflicting)
    assert installed_configuration.get_configuration() is current


def test_reading_the_standalone_default_makes_it_sticky(
    installed_configuration,
) -> None:
    import dataclasses

    installed_configuration.reset_configuration()
    default = installed_configuration.get_configuration()

    assert default == installed_configuration.StreamsConfiguration()
    with pytest.raises(installed_configuration.StreamsConfigurationError):
        installed_configuration.configure_process(
            dataclasses.replace(default, enable_tensor_data_representation=True)
        )


_STANDALONE_DEFAULTS_SCRIPT = """
import json
from dataclasses import fields

from inference.core.interfaces.stream.configuration import StreamsConfiguration
from inference.core.interfaces.streams_configuration import build_configuration_from_env

standalone = StreamsConfiguration()
legacy = build_configuration_from_env()
# The manager address settings are deliberately excluded: the legacy builder
# defers them to `manager_app/app.py`'s own import (`None`), while the
# standalone default is the historical `os.getenv` fallback value.
_DEFERRED_FIELDS = {
    "stream_manager_host",
    "stream_manager_port",
    "stream_manager_socket_timeout",
}
pairs = [
    (standalone, legacy),
    (standalone.model_config_defaults, legacy.model_config_defaults),
]
mismatches = []
for left_group, right_group in pairs:
    for member in fields(left_group):
        if member.name in _DEFERRED_FIELDS:
            continue
        left = getattr(left_group, member.name)
        right = getattr(right_group, member.name)
        if left != right or type(left) is not type(right):
            mismatches.append([member.name, repr(left), repr(right)])
print(json.dumps({"mismatches": mismatches}))
"""


def test_standalone_defaults_equal_the_legacy_numpy_defaults() -> None:
    result = _run_child(
        _STANDALONE_DEFAULTS_SCRIPT, {"ENABLE_TENSOR_DATA_REPRESENTATION": "False"}
    )

    assert result == {"mismatches": []}


_PREDICTIONS_QUEUE_SCRIPT = """
import json

from inference.core.interfaces.stream import environment

print(json.dumps([
    environment.PREDICTIONS_QUEUE_SIZE,
    environment.PREDICTIONS_QUEUE_SIZE_EXPLICIT,
]))
"""


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, [512, False]),
        ({"INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE": "512"}, [512, True]),
        ({"INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE": "16"}, [16, True]),
    ],
    ids=["omitted", "explicit-default", "explicit-value"],
)
def test_explicit_predictions_queue_size_is_distinguished_from_the_default(
    overrides: Dict[str, str], expected: list
) -> None:
    result = _run_child(_PREDICTIONS_QUEUE_SCRIPT, overrides)

    assert result == expected


_BOOTSTRAP_ORDER_SCRIPT = """
import json
import sys

observed = {}


class Recorder:
    def find_spec(self, name, path=None, target=None):
        if name == "inference.core.interfaces.stream.configuration":
            env_module = sys.modules.get("inference.core.env")
            observed["env_complete_before_configuration"] = hasattr(
                env_module, "DEFAULT_BUFFER_SIZE"
            )
        if name == "inference.core.interfaces.stream.environment":
            configuration = sys.modules[
                "inference.core.interfaces.stream.configuration"
            ]
            observed["installed_before_facade"] = (
                configuration._CONFIGURATION is not None
            )
        return None


sys.meta_path.insert(0, Recorder())
import inference.core.exceptions  # noqa: E402

from inference.core.interfaces.stream import environment  # noqa: E402

observed["default_buffer_size"] = environment.DEFAULT_BUFFER_SIZE
observed["backpressure"] = environment.DEFAULT_ADAPTIVE_MODE_BACKPRESSURE
print(json.dumps(observed))
"""


def test_entering_through_core_exceptions_cannot_freeze_standalone_defaults() -> None:
    result = _run_child(
        _BOOTSTRAP_ORDER_SCRIPT,
        {"USE_INFERENCE_MODELS": "True", "ENABLE_TENSOR_DATA_REPRESENTATION": "True"},
    )

    assert result == {
        "env_complete_before_configuration": True,
        "installed_before_facade": True,
        "default_buffer_size": 8,
        "backpressure": True,
    }


_MANAGER_ONLY_INVALID_ENV = {"STREAM_MANAGER_PORT": "not-a-port"}

_CORE_EXCEPTIONS_ONLY_SCRIPT = """
import json

import inference.core.exceptions  # noqa: E402

print(json.dumps({"imported": True}))
"""


def test_core_exceptions_import_survives_an_invalid_manager_only_setting() -> None:
    # STREAM_MANAGER_PORT has no `env.py` counterpart and is manager-only: it
    # must not be parsed on the bootstrap path every `inference.core` import
    # runs, so an invalid value here cannot break unrelated imports.
    result = _run_child(_CORE_EXCEPTIONS_ONLY_SCRIPT, _MANAGER_ONLY_INVALID_ENV)

    assert result == {"imported": True}


_MANAGER_APP_IMPORT_SCRIPT = """
import inference.core.interfaces.stream_manager.manager_app.app  # noqa: E402
"""


def test_manager_app_import_still_rejects_an_invalid_manager_only_setting() -> None:
    # The same invalid value must still be rejected once the manager app
    # actually needs it - deferring the parse must not weaken validation.
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in _CONTROLLED_VARIABLES
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "DISABLE_VERSION_CHECK": "True",
            "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT / "workflows"), str(REPO_ROOT / "inference_models")]
            ),
            **_MANAGER_ONLY_INVALID_ENV,
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _MANAGER_APP_IMPORT_SCRIPT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )

    assert completed.returncode != 0
    assert "not-a-port" in completed.stderr


_SET_MANAGER_ENV_AFTER_CORE_IMPORT_SCRIPT = """
import json
import os

import inference.core.exceptions  # noqa: E402

os.environ["STREAM_MANAGER_HOST"] = "0.0.0.0"
os.environ["STREAM_MANAGER_PORT"] = "7171"
os.environ["STREAM_MANAGER_SOCKET_TIMEOUT"] = "2"

from inference.core.interfaces.stream_manager.manager_app import app  # noqa: E402

print(json.dumps({"host": app.HOST, "port": app.PORT, "timeout": app.SOCKET_TIMEOUT}))
"""


def test_manager_env_set_after_core_import_still_reaches_the_manager_app() -> None:
    result = _run_child(_SET_MANAGER_ENV_AFTER_CORE_IMPORT_SCRIPT)

    assert result == {"host": "0.0.0.0", "port": 7171, "timeout": 2.0}


# A01 regression: `inference.core.interfaces.stream.environment` is imported
# by camera and pipeline modules for settings that have nothing to do with
# the manager, well before anything needs the manager's address. It must not
# parse `STREAM_MANAGER_PORT`/`STREAM_MANAGER_SOCKET_TIMEOUT` as a side effect
# of that import - only `manager_app/app.py`, at its own later import, may.
_CAMERA_FIRST_THEN_MANAGER_SETTING_SCRIPT = """
import json
import os

from inference.core.interfaces.camera.video_source import VideoSource  # noqa: E402

camera_import_ok = True

os.environ["STREAM_MANAGER_HOST"] = "{host}"
os.environ["STREAM_MANAGER_PORT"] = "{port}"
os.environ["STREAM_MANAGER_SOCKET_TIMEOUT"] = "{timeout}"

try:
    from inference.core.interfaces.stream_manager.manager_app import app  # noqa: E402
    result = {{
        "camera_import_ok": camera_import_ok,
        "manager_import_ok": True,
        "host": app.HOST,
        "port": app.PORT,
        "timeout": app.SOCKET_TIMEOUT,
    }}
except ValueError:
    result = {{"camera_import_ok": camera_import_ok, "manager_import_ok": False}}

print(json.dumps(result))
"""


def test_camera_import_then_a_valid_manager_setting_still_reaches_the_manager_app() -> (
    None
):
    script = _CAMERA_FIRST_THEN_MANAGER_SETTING_SCRIPT.format(
        host="0.0.0.0", port="7171", timeout="2"
    )

    result = _run_child(script)

    assert result == {
        "camera_import_ok": True,
        "manager_import_ok": True,
        "host": "0.0.0.0",
        "port": 7171,
        "timeout": 2.0,
    }


@pytest.mark.parametrize(
    "host, port, timeout",
    [
        ("127.0.0.1", "not-a-port", "5.0"),
        ("127.0.0.1", "7070", "not-a-timeout"),
    ],
    ids=["invalid-port", "invalid-timeout"],
)
def test_camera_import_survives_an_invalid_manager_setting_that_still_fails_the_manager_app(
    host: str, port: str, timeout: str
) -> None:
    script = _CAMERA_FIRST_THEN_MANAGER_SETTING_SCRIPT.format(
        host=host, port=port, timeout=timeout
    )
    # Pass manager settings via overrides so they exist before camera import,
    # testing that camera doesn't parse invalid manager-only settings.
    result = _run_child(
        script,
        {
            "STREAM_MANAGER_HOST": host,
            "STREAM_MANAGER_PORT": port,
            "STREAM_MANAGER_SOCKET_TIMEOUT": timeout,
        },
    )

    assert result == {"camera_import_ok": True, "manager_import_ok": False}


def test_falsy_explicit_manager_settings_are_honored_over_the_environment(
    monkeypatch,
) -> None:
    # `is not None`, not `or`: an explicitly configured falsy value (port `0`,
    # timeout `0.0`, host `""`) must win over the environment too.
    import importlib

    from inference.core.interfaces.stream import environment
    from inference.core.interfaces.stream_manager.manager_app import app

    monkeypatch.setattr(environment, "STREAM_MANAGER_HOST", "")
    monkeypatch.setattr(environment, "STREAM_MANAGER_PORT", 0)
    monkeypatch.setattr(environment, "STREAM_MANAGER_SOCKET_TIMEOUT", 0.0)
    monkeypatch.setenv("STREAM_MANAGER_HOST", "0.0.0.0")
    monkeypatch.setenv("STREAM_MANAGER_PORT", "7171")
    monkeypatch.setenv("STREAM_MANAGER_SOCKET_TIMEOUT", "2")

    importlib.reload(app)
    resolved = (app.HOST, app.PORT, app.SOCKET_TIMEOUT)

    monkeypatch.undo()
    importlib.reload(app)

    assert resolved == ("", 0, 0.0)


_EXPLICIT_STANDALONE_MANAGER_CONFIGURATION_SCRIPT = """
import json
import sys
import types
from pathlib import Path

# As in the canonical-first script above: bypass `inference/core/__init__.py`
# (the legacy bootstrap) so a host installing its own configuration first -
# the future standalone-package order - is what is under test, not the
# bootstrap's own conflict detection.
root = Path.cwd()
for package in (
    "inference",
    "inference.core",
    "inference.core.interfaces",
    "inference.core.interfaces.stream",
):
    module = types.ModuleType(package)
    module.__path__ = [str(root.joinpath(*package.split(".")))]
    sys.modules[package] = module

from inference.core.interfaces.stream.configuration import (
    StreamsConfiguration,
    configure_process,
)

configure_process(
    StreamsConfiguration(
        stream_manager_host="10.0.0.1",
        stream_manager_port=9999,
        stream_manager_socket_timeout=1.5,
    )
)

from inference.core.interfaces.stream import environment

print(json.dumps({
    "host": environment.STREAM_MANAGER_HOST,
    "port": environment.STREAM_MANAGER_PORT,
    "timeout": environment.STREAM_MANAGER_SOCKET_TIMEOUT,
}))
"""


def test_explicit_standalone_manager_address_configuration_is_not_lost() -> None:
    # STREAM_MANAGER_HOST/PORT/SOCKET_TIMEOUT are set in the environment too,
    # but an explicitly installed configuration must win over them.
    result = _run_child(
        _EXPLICIT_STANDALONE_MANAGER_CONFIGURATION_SCRIPT,
        {
            "STREAM_MANAGER_HOST": "0.0.0.0",
            "STREAM_MANAGER_PORT": "7171",
            "STREAM_MANAGER_SOCKET_TIMEOUT": "2",
        },
    )

    assert result == {"host": "10.0.0.1", "port": 9999, "timeout": 1.5}


# The package modules are loaded from their files under stub parent packages,
# so neither `inference/__init__.py` nor `inference/core/__init__.py` (the
# legacy bootstrap) runs: this is the future standalone-package order, where a
# caller installs its own configuration before anything reads it.
_CANONICAL_FIRST_SCRIPT = """
import json
import sys
import types
from pathlib import Path

root = Path.cwd()
for package in (
    "inference",
    "inference.core",
    "inference.core.interfaces",
    "inference.core.interfaces.stream",
):
    module = types.ModuleType(package)
    module.__path__ = [str(root.joinpath(*package.split(".")))]
    sys.modules[package] = module
baseline = set(sys.modules)

from inference.core.interfaces.stream.configuration import (
    StreamsConfiguration,
    configure_process,
)

configure_process(
    StreamsConfiguration(
        default_buffer_size=3,
        enable_tensor_data_representation=True,
        offline_mode=True,
    )
)
from inference.core.interfaces.stream import environment

print(json.dumps({
    "values": [
        environment.DEFAULT_BUFFER_SIZE,
        environment.ENABLE_TENSOR_DATA_REPRESENTATION,
        environment.OFFLINE_MODE,
    ],
    "new_modules": sorted(
        name
        for name in set(sys.modules) - baseline
        if name.split(".")[0] in {"inference", "cv2", "numpy", "torch", "pydantic"}
    ),
}))
"""


def test_a_canonical_configuration_installed_first_reaches_the_facade() -> None:
    result = _run_child(_CANONICAL_FIRST_SCRIPT)

    assert result == {
        "values": [3, True, True],
        "new_modules": [
            "inference.core.interfaces.stream.configuration",
            "inference.core.interfaces.stream.environment",
        ],
    }


def test_collection_policy_reads_the_tensor_flag_at_call_time(monkeypatch) -> None:
    from inference.core.interfaces.camera.collection_policy import (
        VideoProcessingMode,
        resolve_video_processing_mode,
    )
    from inference.core.interfaces.stream import environment

    monkeypatch.setattr(environment, "ENABLE_TENSOR_DATA_REPRESENTATION", True)
    tensor_mode = resolve_video_processing_mode(explicit_mode=None)
    monkeypatch.setattr(environment, "ENABLE_TENSOR_DATA_REPRESENTATION", False)
    numpy_mode = resolve_video_processing_mode(explicit_mode=None)

    assert tensor_mode is VideoProcessingMode.AUTO
    assert numpy_mode is None


# --------------------------------------------------------------------------
# Buffer strategy enums
# --------------------------------------------------------------------------


def test_buffer_strategy_enums_are_shared_between_old_and_new_paths() -> None:
    from inference.core.interfaces.camera import buffer_strategies, video_source
    from inference.core.interfaces.stream_manager.manager_app import entities

    assert video_source.BufferFillingStrategy is buffer_strategies.BufferFillingStrategy
    assert (
        video_source.BufferConsumptionStrategy
        is buffer_strategies.BufferConsumptionStrategy
    )
    assert entities.BufferFillingStrategy is buffer_strategies.BufferFillingStrategy
    assert (
        entities.BufferConsumptionStrategy
        is buffer_strategies.BufferConsumptionStrategy
    )


def test_buffer_strategy_wire_values_are_frozen() -> None:
    from inference.core.interfaces.camera.buffer_strategies import (
        BufferConsumptionStrategy,
        BufferFillingStrategy,
    )

    assert {member.name: member.value for member in BufferFillingStrategy} == {
        "WAIT": "WAIT",
        "DROP_OLDEST": "DROP_OLDEST",
        "ADAPTIVE_DROP_OLDEST": "ADAPTIVE_DROP_OLDEST",
        "DROP_LATEST": "DROP_LATEST",
        "ADAPTIVE_DROP_LATEST": "ADAPTIVE_DROP_LATEST",
    }
    assert {member.name: member.value for member in BufferConsumptionStrategy} == {
        "LAZY": "LAZY",
        "EAGER": "EAGER",
    }


@pytest.mark.parametrize("protocol", range(2, 6))
def test_buffer_strategy_pickles_keep_the_historical_reference(protocol: int) -> None:
    import pickle

    from inference.core.interfaces.camera.buffer_strategies import (
        BufferConsumptionStrategy,
        BufferFillingStrategy,
    )

    for member in [*BufferFillingStrategy, *BufferConsumptionStrategy]:
        payload = pickle.dumps(member, protocol=protocol)

        assert pickle.loads(payload) is member
        # Readable by a process that only has the pre-extraction module.
        assert b"inference.core.interfaces.camera.video_source" in payload
        assert b"buffer_strategies" not in payload


# --------------------------------------------------------------------------
# Wire entities import weight
# --------------------------------------------------------------------------

_WIRE_IMPORT_SCRIPT = """
import json
import sys

import inference.core

baseline = set(sys.modules)
from inference.core.interfaces.stream_manager.manager_app import entities  # noqa

print(json.dumps(sorted(set(sys.modules) - baseline)))
"""


def test_request_entities_do_not_import_the_decoder_webrtc_or_pipeline() -> None:
    # cv2 itself is already loaded by the bootstrap (inference_models ->
    # supervision), so the check is on what the entities add on top of it.
    new_modules = set(_run_child(_WIRE_IMPORT_SCRIPT))

    assert not {
        name for name in new_modules if name.split(".")[0] in {"cv2", "aiortc", "av"}
    }
    # WP-A03: the (empty) stream_manager and manager_app packages are already
    # loaded by `inference.core`, which installs the default pipeline host
    # descriptor from the import-light manager_app.host.
    assert {name for name in new_modules if name.startswith("inference.")} == {
        "inference.core.interfaces.camera",
        "inference.core.interfaces.camera.buffer_strategies",
        "inference.core.interfaces.camera.source_reference_validation",
        "inference.core.interfaces.stream.environment",
        "inference.core.interfaces.stream_manager.manager_app.entities",
    }


# --------------------------------------------------------------------------
# Logger hierarchy
# --------------------------------------------------------------------------

_STDLIB_LOGGER_MODULES = (
    "inference.core.interfaces.camera.camera",
    "inference.core.interfaces.camera.utils",
    "inference.core.interfaces.camera.video_source",
    # A facade since WP-A02: the legacy module keeps this historical logger
    # name, while the pipeline runtime logs through stream.pipeline.
    "inference.core.interfaces.stream.inference_pipeline",
    "inference.core.interfaces.stream.pipeline",
    "inference.core.interfaces.stream.sinks",
    "inference.core.interfaces.stream.stream",
    "inference.core.interfaces.stream.utils",
    "inference.core.interfaces.stream_manager.api.stream_manager_client",
    "inference.core.interfaces.stream_manager.manager_app.app",
    "inference.core.interfaces.stream_manager.manager_app.communication",
    "inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager",
    "inference.core.interfaces.stream_manager.manager_app.webrtc",
)


class _ProbeHandler:
    def __init__(self) -> None:
        import logging

        self.records = []
        self.handler = logging.Handler(level=logging.DEBUG)
        self.handler.emit = self.records.append


@pytest.mark.parametrize("module_name", _STDLIB_LOGGER_MODULES)
def test_module_logger_reaches_the_inference_logger_handler(module_name: str) -> None:
    # caplog listens on the root logger, which `inference` never propagates to
    # (inference/core/logger.py sets `propagate = False`), so the probe is
    # attached to the `inference` logger itself - where the real handler is.
    import importlib
    import logging

    module = importlib.import_module(module_name)
    inference_logger = logging.getLogger("inference")
    probe = _ProbeHandler()
    previous_level = inference_logger.level
    inference_logger.addHandler(probe.handler)
    inference_logger.setLevel(logging.DEBUG)
    try:
        module.logger.warning("probe from %s", module_name)
    finally:
        inference_logger.removeHandler(probe.handler)
        inference_logger.setLevel(previous_level)

    assert module.logger.name == module_name
    assert module.logger.propagate is True
    assert not module.logger.handlers
    assert [record.getMessage() for record in probe.records] == [
        f"probe from {module_name}"
    ]


def test_inference_logger_keeps_its_handler_and_isolation() -> None:
    import logging

    import inference.core.logger  # noqa: F401 - configures the hierarchy

    inference_logger = logging.getLogger("inference")

    assert inference_logger.handlers
    assert inference_logger.propagate is False
