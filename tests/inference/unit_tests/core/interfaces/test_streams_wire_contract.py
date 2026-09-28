"""Wire characterization for the client <-> manager TCP boundary.

Framing/codec mechanics (fragmented reads, malformed header, overflow
recovery, JSON decode errors) already have direct unit coverage in
tests/interfaces/stream_manager/{manager_app/test_communucation.py,
manager_app/test_serialisation.py,api/test_stream_manager_client.py}. This
file does not repeat that; it freezes what those files leave implicit:

- the exact wire vocabulary (CommandType/ErrorType string values, envelope
  key names) a moved package must keep byte-for-byte compatible with an
  unmoved legacy manager during a rolling upgrade;
- payload schema defaults/required-ness, and that `exclude_none=True`
  (the client's actual `.dict(...)` call, api/stream_manager_client.py:112,130)
  drops None-valued optional fields but keeps explicit falsy values;
- `inner_error_type`, which nothing else exercises directly;
- the real normal-vs-WebRTC init error-handling asymmetry, end to end
  through the actual `InferencePipelineManager`, not just read off the
  source.
"""

import asyncio
import json
import threading
from datetime import datetime
from multiprocessing import Queue
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference.core.env import (
    ALLOW_UNSAFE_GSTREAMER_PIPELINES,
    DEFAULT_BUFFER_SIZE,
    PREDICTIONS_QUEUE_SIZE,
    WEBRTC_REALTIME_PROCESSING,
)
from inference.core.exceptions import (
    RoboflowAPIConnectionError,
    RoboflowAPITimeoutError,
)
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream_manager.api import stream_manager_client
from inference.core.interfaces.stream_manager.api.entities import (
    CommandContext,
    CommandResponse,
)
from inference.core.interfaces.stream_manager.api.errors import (
    ProcessesManagerClientError,
)
from inference.core.interfaces.stream_manager.api.stream_manager_client import (
    StreamManagerClient,
)
from inference.core.interfaces.stream_manager.manager_app import (
    app as manager_app_module,
)
from inference.core.interfaces.stream_manager.manager_app import (
    inference_pipeline_manager,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    COMMAND_KEY,
    ENCODING,
    ERROR_TYPE_KEY,
    PIPELINE_ID_KEY,
    REPORT_KEY,
    REQUEST_ID_KEY,
    RESPONSE_KEY,
    SOURCES_METADATA_KEY,
    STATE_KEY,
    STATUS_KEY,
    TYPE_KEY,
    VIDEO_SOURCE_STATUS_UPDATES_KEY,
    CommandType,
    ConsumeResultsPayload,
    ErrorType,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
    MemorySinkConfiguration,
    OperationStatus,
    VideoConfiguration,
    WebRTCData,
    WebRTCOffer,
    WebRTCTURNConfig,
    WorkflowConfiguration,
)
from inference.core.interfaces.stream_manager.manager_app.errors import (
    CommunicationProtocolError,
)
from inference.core.interfaces.stream_manager.manager_app.inference_pipeline_manager import (
    InferencePipelineManager,
)
from tests.inference.unit_tests.core.interfaces.stream_manager.api.test_stream_manager_client import (
    DummyStreamWriter,
    assembly_socket_reader,
    assert_correct_command_sent,
)

# Enum wire vocabulary


def test_command_type_wire_values_are_frozen() -> None:
    assert {member.name: member.value for member in CommandType} == {
        "INIT": "init",
        "WEBRTC": "webrtc",
        "MUTE": "mute",
        "RESUME": "resume",
        "STATUS": "status",
        "TERMINATE": "terminate",
        "LIST_PIPELINES": "list_pipelines",
        "CONSUME_RESULT": "consume_result",
    }


def test_error_type_wire_values_are_frozen() -> None:
    assert {member.name: member.value for member in ErrorType} == {
        "INTERNAL_ERROR": "internal_error",
        "INVALID_PAYLOAD": "invalid_payload",
        "NOT_FOUND": "not_found",
        "OPERATION_ERROR": "operation_error",
        "AUTHORISATION_ERROR": "authorisation_error",
    }


def test_error_type_to_client_exception_mapping_covers_every_value() -> None:
    # a value missing here silently falls back to ProcessesManagerClientError
    assert set(stream_manager_client.ERRORS_MAPPING) == {e.value for e in ErrorType}
    assert len(set(stream_manager_client.ERRORS_MAPPING.values())) == len(ErrorType)


def test_envelope_key_names_are_frozen() -> None:
    # interop needs the literal key spelling to match, not just the constant name
    assert (
        STATUS_KEY,
        STATE_KEY,
        SOURCES_METADATA_KEY,
        VIDEO_SOURCE_STATUS_UPDATES_KEY,
        REPORT_KEY,
        TYPE_KEY,
        ERROR_TYPE_KEY,
        REQUEST_ID_KEY,
        PIPELINE_ID_KEY,
        COMMAND_KEY,
        RESPONSE_KEY,
        ENCODING,
    ) == (
        "status",
        "state",
        "sources_metadata",
        "video_source_status_updates",
        "report",
        "type",
        "error_type",
        "request_id",
        "pipeline_id",
        "command",
        "response",
        "utf-8",
    )


# Framing: both sides must agree on header size independently of each other


def test_client_and_server_default_framing_constants_match() -> None:
    assert stream_manager_client.HEADER_SIZE == manager_app_module.HEADER_SIZE == 4
    assert (
        stream_manager_client.BUFFER_SIZE
        == manager_app_module.SOCKET_BUFFER_SIZE
        == 16384
    )


# Payload schemas: required fields, defaults, and exclude_none=True omission


def _valid_video_configuration() -> VideoConfiguration:
    return VideoConfiguration(type="VideoConfiguration", video_reference="rtsp://x")


def test_video_configuration_defaults() -> None:
    config = _valid_video_configuration()
    assert config.max_fps is None
    assert config.source_buffer_filling_strategy == BufferFillingStrategy.DROP_OLDEST
    assert config.source_buffer_consumption_strategy == BufferConsumptionStrategy.EAGER
    assert config.video_source_properties is None
    assert config.batch_collection_timeout is None


def test_memory_sink_configuration_default_buffer_size() -> None:
    assert (
        MemorySinkConfiguration(type="MemorySinkConfiguration").results_buffer_size
        == 64
    )


def test_workflow_configuration_defaults() -> None:
    config = WorkflowConfiguration(type="WorkflowConfiguration")
    assert config.workflow_specification is None
    assert config.workspace_name is None
    assert config.workflow_id is None
    assert config.workflow_version_id is None
    assert config.image_input_name == "image"
    assert config.workflows_parameters is None
    assert config.disable_sinks is False
    assert config.workflows_thread_pool_workers == 4
    assert config.execution_engine_thread_pool_workers == 4
    assert config.cancel_thread_pool_tasks_on_exit is True
    assert config.video_metadata_input_name == "video_metadata"


def test_initialise_pipeline_payload_defaults_bind_to_current_env() -> None:
    payload = InitialisePipelinePayload(
        video_configuration=_valid_video_configuration(),
        processing_configuration=WorkflowConfiguration(type="WorkflowConfiguration"),
    )
    assert payload.sink_configuration == MemorySinkConfiguration(
        type="MemorySinkConfiguration"
    )
    assert payload.consumption_timeout is None
    assert payload.api_key is None
    # bound at class-definition/import time from core.env, not at call time
    assert payload.predictions_queue_size == PREDICTIONS_QUEUE_SIZE
    assert payload.decoding_buffer_size == DEFAULT_BUFFER_SIZE


def test_webrtc_turn_config_normalizes_single_url_to_list() -> None:
    single = WebRTCTURNConfig(urls="turn:example", username="u", credential="c")
    many = WebRTCTURNConfig(urls=["turn:a", "turn:b"], username="u", credential="c")
    assert single.urls == ["turn:example"]
    assert many.urls == ["turn:a", "turn:b"]


def test_initialise_webrtc_pipeline_payload_defaults() -> None:
    payload = InitialiseWebRTCPipelinePayload(
        video_configuration=_valid_video_configuration(),
        processing_configuration=WorkflowConfiguration(type="WorkflowConfiguration"),
        webrtc_offer=WebRTCOffer(type="offer", sdp="v=0"),
    )
    assert payload.webrtc_peer_timeout == 1
    assert payload.webrtc_realtime_processing == WEBRTC_REALTIME_PROCESSING
    assert payload.webrtc_turn_config is None
    assert payload.stream_output == []
    assert payload.data_output == []
    assert payload.webcam_fps is None
    assert payload.processing_timeout == 0.005
    assert payload.fps_probe_frames == 10
    assert payload.max_consecutive_timeouts == 30
    assert payload.min_consecutive_on_time == 5


def test_webrtc_data_defaults() -> None:
    data = WebRTCData()
    assert data.stream_output is None
    assert data.data_output is None
    assert data.ack is None


def test_consume_results_payload_default_excluded_fields_is_empty_list() -> None:
    assert ConsumeResultsPayload().excluded_fields == []


def test_video_configuration_rejects_unsafe_gstreamer_by_default() -> None:
    # validator (entities.py) is wired to core.env; off by default in this env
    if ALLOW_UNSAFE_GSTREAMER_PIPELINES:
        pytest.skip("ALLOW_UNSAFE_GSTREAMER_PIPELINES is enabled in this environment")
    with pytest.raises(Exception):
        VideoConfiguration(
            type="VideoConfiguration",
            video_reference="gst-launch-1.0 videotestsrc ! autovideosink",
        )


def test_client_command_dict_omits_none_but_keeps_explicit_falsy_values() -> None:
    # reproduces api/stream_manager_client.py:112,130 - the real wire command dict
    payload = InitialisePipelinePayload(
        video_configuration=_valid_video_configuration(),
        processing_configuration=WorkflowConfiguration(type="WorkflowConfiguration"),
    )
    command = payload.dict(exclude_none=True)
    command[TYPE_KEY] = CommandType.INIT

    # None-valued optional fields, including nested ones, are dropped...
    assert "api_key" not in command
    assert "consumption_timeout" not in command
    assert "workflow_specification" not in command["processing_configuration"]
    assert "max_fps" not in command["video_configuration"]
    # ...but explicit non-None values, including falsy ones, survive.
    assert command["processing_configuration"]["disable_sinks"] is False
    assert command["predictions_queue_size"] == PREDICTIONS_QUEUE_SIZE

    # the command as actually JSON-encoded on the wire (send_message's own serializer)
    wire_bytes = json.dumps(
        command, default=stream_manager_client._json_serializer
    ).encode(ENCODING)
    decoded = json.loads(wire_bytes)
    assert decoded[TYPE_KEY] == CommandType.INIT.value
    assert "api_key" not in decoded


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_client_initialise_pipeline_omits_none_fields_on_the_real_wire(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # drives the real client chain and inspects the actual wire bytes, not a fake dict
    payload = InitialisePipelinePayload(
        video_configuration=_valid_video_configuration(),
        processing_configuration=WorkflowConfiguration(type="WorkflowConfiguration"),
    )
    reader = assembly_socket_reader(
        message={
            "request_id": "my_request",
            "pipeline_id": "new_pipeline",
            "response": {"status": "success"},
        },
        header_size=stream_manager_client.HEADER_SIZE,
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    client = StreamManagerClient.init(host="127.0.0.1", port=7070)

    result = await client.initialise_pipeline(initialisation_request=payload)

    assert result == CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="new_pipeline"),
    )
    expected_command = payload.dict(exclude_none=True)
    expected_command[TYPE_KEY] = CommandType.INIT
    assert_correct_command_sent(
        writer=writer,
        command=expected_command,
        header_size=stream_manager_client.HEADER_SIZE,
        message="Expected exclude_none=True fields to actually be missing from the wire bytes",
    )


# inner_error_type: same shape on client- and server-side error base classes


@pytest.mark.parametrize(
    "error_cls", [ProcessesManagerClientError, CommunicationProtocolError]
)
def test_inner_error_type_is_none_without_a_wrapped_error(error_cls) -> None:
    error = error_cls(private_message="boom")
    assert error.inner_error is None
    assert error.inner_error_type is None
    assert error.public_message is None


@pytest.mark.parametrize(
    "error_cls", [ProcessesManagerClientError, CommunicationProtocolError]
)
def test_inner_error_type_reports_the_wrapped_exceptions_class_name(error_cls) -> None:
    inner = ValueError("nope")
    error = error_cls(
        private_message="boom", public_message="public boom", inner_error=inner
    )
    assert error.inner_error is inner
    assert error.inner_error_type == "ValueError"
    assert error.public_message == "public boom"


# Normal vs WebRTC init error asymmetry, through the real manager


def _assembly_init_payload() -> dict:
    specification = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [],
        "outputs": [],
    }
    payload = InitialisePipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration", video_reference="rtsp://128.0.0.1"
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration", workflow_specification=specification
        ),
        api_key="<MY-API-KEY>",
    ).dict()
    payload[TYPE_KEY] = CommandType.INIT
    return payload


def _assembly_webrtc_payload() -> dict:
    specification = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [],
        "outputs": [],
    }
    payload = InitialiseWebRTCPipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration", video_reference="rtsp://128.0.0.1"
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration", workflow_specification=specification
        ),
        webrtc_offer=WebRTCOffer(type="offer", sdp="v=0"),
        api_key="<MY-API-KEY>",
    ).dict()
    payload[TYPE_KEY] = CommandType.WEBRTC
    return payload


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_normal_init_maps_roboflow_timeout_to_operation_error(
    pipeline_init_mock: MagicMock,
) -> None:
    pipeline_init_mock.side_effect = RoboflowAPITimeoutError("timed out")
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="p", command_queue=command_queue, responses_queue=responses_queue
    )
    command_queue.put(("1", _assembly_init_payload()))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    request_id, response = responses_queue.get()
    assert request_id == "1"
    assert response[STATUS_KEY] == "failure"
    assert response[ERROR_TYPE_KEY] == ErrorType.OPERATION_ERROR
    assert response["error_class"] == "RoboflowAPITimeoutError"


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_normal_init_maps_roboflow_connection_error_to_operation_error(
    pipeline_init_mock: MagicMock,
) -> None:
    pipeline_init_mock.side_effect = RoboflowAPIConnectionError("no network")
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="p", command_queue=command_queue, responses_queue=responses_queue
    )
    command_queue.put(("1", _assembly_init_payload()))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    request_id, response = responses_queue.get()
    assert request_id == "1"
    assert response[ERROR_TYPE_KEY] == ErrorType.OPERATION_ERROR
    assert response["error_class"] == "RoboflowAPIConnectionError"


class _FakeLocalDescription:
    sdp = "v=0 fake-answer"
    type = "answer"


class _FakePeerConnection:
    localDescription = _FakeLocalDescription()
    video_transform_track = None
    data_output = None
    data_channel = None
    stream_output = None
    _consumers_signalled = True

    async def close(self) -> None:
        return None


@pytest.mark.timeout(30)
@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
@mock.patch.object(
    inference_pipeline_manager,
    "init_rtc_peer_connection",
    new_callable=AsyncMock,
)
def test_webrtc_init_maps_the_same_roboflow_timeout_to_internal_error(
    init_rtc_peer_connection_mock: AsyncMock,
    pipeline_init_mock: MagicMock,
) -> None:
    # unlike _initialise_pipeline, _start_webrtc doesn't catch this; asymmetry kept
    init_rtc_peer_connection_mock.return_value = _FakePeerConnection()
    pipeline_init_mock.side_effect = RoboflowAPITimeoutError("timed out")
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="p", command_queue=command_queue, responses_queue=responses_queue
    )
    command_queue.put(("1", _assembly_webrtc_payload()))
    command_queue.put(("2", {"type": CommandType.TERMINATE}))

    manager.run()

    # WebRTC reports success (SDP) once the peer is up, before the pipeline exists
    first_request_id, first_response = responses_queue.get()
    assert first_request_id == "1"
    assert first_response[STATUS_KEY] == OperationStatus.SUCCESS
    assert first_response["sdp"] == _FakePeerConnection.localDescription.sdp

    second_request_id, second_response = responses_queue.get()
    assert second_request_id == "1"
    assert second_response[STATUS_KEY] == "failure"
    assert second_response[ERROR_TYPE_KEY] == ErrorType.INTERNAL_ERROR
    assert second_response["error_class"] == "RoboflowAPITimeoutError"
    assert (
        "Unknown internal error" in second_response["public_error_message"]
    ), "WebRTC path loses the specific timeout message the normal path gives"


# CONSUME_RESULT through the real manager and its real (unmocked) InMemoryBufferSink


@mock.patch.object(inference_pipeline_manager.InferencePipeline, "init_with_workflow")
def test_consume_results_serialises_a_really_buffered_prediction(
    pipeline_init_mock: MagicMock,
) -> None:
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="p", command_queue=command_queue, responses_queue=responses_queue
    )
    manager._handle_command(request_id="1", payload=_assembly_init_payload())
    init_request_id, init_response = responses_queue.get()
    assert init_request_id == "1"
    assert init_response[STATUS_KEY] == OperationStatus.SUCCESS

    # _initialise_pipeline still builds a real sink; seed it like the pipeline would
    frame = VideoFrame(
        image=None,
        frame_id=7,
        frame_timestamp=datetime(2026, 1, 1, 12, 0, 0),
        source_id=0,
    )
    manager._buffer_sink.on_prediction({"predictions": [], "top": "cat"}, frame)

    manager._handle_command(
        request_id="2", payload={TYPE_KEY: CommandType.CONSUME_RESULT}
    )

    consume_request_id, consume_response = responses_queue.get()
    assert consume_request_id == "2"
    assert consume_response[STATUS_KEY] == OperationStatus.SUCCESS
    assert consume_response["outputs"] == [{"predictions": [], "top": "cat"}]
    assert consume_response["frames_metadata"] == [
        {"frame_timestamp": "2026-01-01T12:00:00", "frame_id": 7, "source_id": 0}
    ]


def test_consume_results_reports_empty_buffer_without_touching_the_sink() -> None:
    command_queue, responses_queue = Queue(), Queue()
    manager = InferencePipelineManager(
        pipeline_id="p", command_queue=command_queue, responses_queue=responses_queue
    )
    manager._buffer_sink = inference_pipeline_manager.InMemoryBufferSink.init(
        queue_size=4
    )

    manager._handle_command(
        request_id="1", payload={TYPE_KEY: CommandType.CONSUME_RESULT}
    )

    request_id, response = responses_queue.get()
    assert request_id == "1"
    assert response == {
        STATUS_KEY: OperationStatus.SUCCESS,
        "outputs": [],
        "frames_metadata": [],
    }
