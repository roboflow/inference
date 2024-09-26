import asyncio
import json
from typing import Type
from unittest import mock
from unittest.mock import AsyncMock

import pytest

from inference.core.interfaces.stream_manager.api import stream_manager_client
from inference.core.interfaces.stream_manager.api.entities import (
    CommandContext,
    CommandResponse,
    InferencePipelineStatusResponse,
    ListPipelinesResponse,
)
from inference.core.interfaces.stream_manager.api.errors import (
    ConnectivityError,
    ProcessesManagerAuthorisationError,
    ProcessesManagerClientError,
    ProcessesManagerInternalError,
    ProcessesManagerInvalidPayload,
    ProcessesManagerNotFoundError,
    ProcessesManagerOperationError,
)
from inference.core.interfaces.stream_manager.api.stream_manager_client import (
    StreamManagerClient,
    build_response,
    dispatch_error,
    is_request_unsuccessful,
    receive_message,
    send_command,
    send_message,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    CommandType,
    InitialisePipelinePayload,
    VideoConfiguration,
    WorkflowConfiguration,
)
from inference.core.interfaces.stream_manager.manager_app.errors import (
    CommunicationProtocolError,
    MalformedHeaderError,
    MalformedPayloadError,
    MessageToBigError,
    TransmissionChannelClosed,
)


def test_build_response_when_all_optional_fields_are_filled() -> None:
    # given
    response = {
        "response": {"status": "failure"},
        "request_id": "my_request",
        "pipeline_id": "my_pipeline",
    }

    # when
    result = build_response(response=response)

    # then
    assert result == CommandResponse(
        status="failure",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    ), "Assembled response must indicate failure and context with request id and pipeline id denoted"


def test_build_response_when_all_optional_fields_are_missing() -> None:
    # given
    response = {
        "response": {"status": "failure"},
    }

    # when
    result = build_response(response=response)

    # then
    assert result == CommandResponse(
        status="failure",
        context=CommandContext(request_id=None, pipeline_id=None),
    ), "Assembled response must indicate failure and empty context"


@pytest.mark.parametrize(
    "error_type, expected_error",
    [
        ("internal_error", ProcessesManagerInternalError),
        ("invalid_payload", ProcessesManagerInvalidPayload),
        ("not_found", ProcessesManagerNotFoundError),
        ("operation_error", ProcessesManagerOperationError),
        ("authorisation_error", ProcessesManagerAuthorisationError),
    ],
)
def test_dispatch_error_when_known_error_is_detected(
    error_type: str, expected_error: Type[Exception]
) -> None:
    # given
    error_response = {
        "response": {
            "status": "failure",
            "error_type": error_type,
        }
    }

    # when
    with pytest.raises(expected_error):
        dispatch_error(error_response=error_response)


def test_dispatch_error_when_unknown_error_is_detected() -> None:
    # given
    error_response = {
        "response": {
            "status": "failure",
            "error_type": "unknown",
        }
    }

    # when
    with pytest.raises(ProcessesManagerClientError):
        dispatch_error(error_response=error_response)


def test_dispatch_error_when_malformed_payload_is_detected() -> None:
    # given
    error_response = {"response": {"status": "failure"}}

    # when
    with pytest.raises(ProcessesManagerClientError):
        dispatch_error(error_response=error_response)


def test_is_request_unsuccessful_when_successful_response_given() -> None:
    # given
    response = {"response": {"status": "success"}}

    # when
    result = is_request_unsuccessful(response=response)

    # then
    assert (
        result is False
    ), "Success status denoted should be assumed as sign of request success"


def test_is_request_unsuccessful_when_unsuccessful_response_given() -> None:
    # given
    error_response = {
        "response": {
            "status": "failure",
            "error_type": "not_found",
        }
    }

    # when
    result = is_request_unsuccessful(response=error_response)

    # then
    assert result is True, "Explicitly failed response is indication of failed response"


def test_is_request_unsuccessful_when_malformed_response_given() -> None:
    # given
    response = {"response": {"some": "data"}}

    # when
    result = is_request_unsuccessful(response=response)

    # then
    assert (
        result is True
    ), "When success is not clearly demonstrated - failure is to be assumed"


class DummyStreamReader:
    def __init__(self, read_buffer_content: bytes):
        self._read_buffer_content = read_buffer_content

    async def read(self, n: int = -1) -> bytes:
        if n == -1:
            n = len(self._read_buffer_content)
        to_return = self._read_buffer_content[:n]
        self._read_buffer_content = self._read_buffer_content[n:]
        return to_return


@pytest.mark.asyncio
async def test_receive_message_when_malformed_header_sent() -> None:
    # given
    header = 3
    reader = DummyStreamReader(
        read_buffer_content=header.to_bytes(length=1, byteorder="big")
    )

    # when
    with pytest.raises(MalformedHeaderError):
        _ = await receive_message(reader=reader, header_size=4, buffer_size=512)


@pytest.mark.asyncio
async def test_receive_message_when_payload_to_be_read_in_single_piece() -> None:
    # given
    data = b"DO OR NOT DO, THERE IS NO TRY"
    payload = len(data).to_bytes(length=4, byteorder="big") + data
    reader = DummyStreamReader(read_buffer_content=payload)

    # when
    result = await receive_message(
        reader=reader, header_size=4, buffer_size=len(payload)
    )

    # then
    assert (
        result == b"DO OR NOT DO, THERE IS NO TRY"
    ), "Result must be exact to the data in payload"


@pytest.mark.asyncio
async def test_receive_message_when_payload_to_be_read_in_multiple_pieces() -> None:
    # given
    data = b"DO OR NOT DO, THERE IS NO TRY"
    payload = len(data).to_bytes(length=4, byteorder="big") + data
    reader = DummyStreamReader(read_buffer_content=payload)

    # when
    result = await receive_message(reader=reader, header_size=4, buffer_size=1)

    # then
    assert (
        result == b"DO OR NOT DO, THERE IS NO TRY"
    ), "Result must be exact to the data in payload"


@pytest.mark.asyncio
async def test_receive_message_when_not_all_declared_bytes_received() -> None:
    # given
    data = b"DO OR NOT DO, THERE IS NO TRY"
    payload = len(data).to_bytes(length=4, byteorder="big") + data[:5]
    reader = DummyStreamReader(read_buffer_content=payload)

    # when
    with pytest.raises(TransmissionChannelClosed):
        _ = await receive_message(reader=reader, header_size=4, buffer_size=1)


@pytest.mark.asyncio
async def test_send_message_when_content_cannot_be_serialised() -> None:
    # given
    writer = AsyncMock()

    # when
    with pytest.raises(MalformedPayloadError):
        await send_message(writer=writer, message=set([1, 2, 3]), header_size=4)


@pytest.mark.asyncio
async def test_send_message_when_message_is_to_long_up_to_header_length() -> None:
    # given
    writer = AsyncMock()
    message = {"data": [i for i in range(1024)]}

    # when
    with pytest.raises(MessageToBigError):
        await send_message(writer=writer, message=message, header_size=1)


@pytest.mark.asyncio
async def test_send_message_when_communication_problem_arises() -> None:
    # given
    writer = AsyncMock()
    writer.drain.side_effect = IOError()
    message = {"data": "some"}

    # when
    with pytest.raises(CommunicationProtocolError):
        await send_message(writer=writer, message=message, header_size=4)


@pytest.mark.asyncio
async def test_send_message_when_communication_succeeds() -> None:
    # given
    writer = AsyncMock()
    message = {"data": "some"}
    serialised_message = json.dumps(message).encode("utf-8")
    expected_payload = (
        len(serialised_message).to_bytes(length=4, byteorder="big") + serialised_message
    )

    # when
    await send_message(writer=writer, message=message, header_size=4)

    # then
    writer.write.assert_called_once_with(expected_payload)


class DummyStreamWriter:
    def __init__(self, operation_delay: float = 0.0):
        self._write_buffer_content = b""
        self._operation_delay = operation_delay

    def get_content(self) -> bytes:
        return self._write_buffer_content

    def write(self, payload: bytes) -> None:
        self._write_buffer_content += payload

    async def drain(self) -> None:
        await asyncio.sleep(self._operation_delay)

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        await asyncio.sleep(self._operation_delay)


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_send_command_when_connectivity_problem_arises(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    establish_socket_connection_mock.side_effect = ConnectionError()

    # when
    with pytest.raises(ConnectivityError):
        _ = await send_command(
            host="127.0.0.1",
            port=7070,
            command={},
            header_size=4,
            buffer_size=16438,
            timeout=0.1,
        )


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_send_command_when_timeout_is_raised(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = DummyStreamReader(read_buffer_content=b"")
    establish_socket_connection_mock.return_value = (
        reader,
        DummyStreamWriter(operation_delay=1.0),
    )

    # when
    with pytest.raises(ConnectivityError):
        _ = await send_command(
            host="127.0.0.1",
            port=7070,
            command={},
            header_size=4,
            buffer_size=16438,
            timeout=0.1,
        )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_send_command_when_communication_successful(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = assembly_socket_reader(
        message={"response": {"status": "success"}}, header_size=4
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    command = {
        "type": CommandType.TERMINATE,
        "pipeline_id": "my_pipeline",
    }

    # when
    result = await send_command(
        host="127.0.0.1", port=7070, command=command, header_size=4, buffer_size=16438
    )

    # then
    assert result == {"response": {"status": "success"}}
    assert_correct_command_sent(
        writer=writer,
        command=command,
        header_size=4,
        message="Expected to send termination command successfully",
    )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_send_command_when_response_payload_could_not_be_decoded(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    response_message = b"FOR SURE NOT A JSON"
    response_payload = (
        len(response_message).to_bytes(length=4, byteorder="big") + response_message
    )
    reader = DummyStreamReader(read_buffer_content=response_payload)
    establish_socket_connection_mock.return_value = (
        reader,
        DummyStreamWriter(operation_delay=1.0),
    )
    command = {
        "type": CommandType.TERMINATE,
        "pipeline_id": "my_pipeline",
    }

    # when
    with pytest.raises(MalformedPayloadError):
        _ = await send_command(
            host="127.0.0.1",
            port=7070,
            command=command,
            header_size=4,
            buffer_size=16438,
        )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_stream_manager_client_can_successfully_list_pipelines(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = assembly_socket_reader(
        message={
            "request_id": "my_request",
            "response": {"status": "success", "pipelines": ["a", "b", "c"]},
        },
        header_size=4,
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    expected_command = {"type": CommandType.LIST_PIPELINES}
    client = StreamManagerClient.init(
        host="127.0.0.1",
        port=7070,
        operations_timeout=1.0,
        header_size=4,
        buffer_size=16438,
    )

    # when
    result = await client.list_pipelines()

    # then
    assert result == ListPipelinesResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id=None),
        pipelines=["a", "b", "c"],
    )
    assert_correct_command_sent(
        writer=writer,
        command=expected_command,
        header_size=4,
        message="Expected list pipelines command to be sent",
    )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_stream_manager_client_can_successfully_initialise_pipeline(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = assembly_socket_reader(
        message={
            "request_id": "my_request",
            "pipeline_id": "new_pipeline",
            "response": {"status": "success"},
        },
        header_size=4,
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    initialisation_request = InitialisePipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration",
            video_reference="rtsp://128.0.0.1",
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration",
            workspace_name="some",
            workflow_id="other",
        ),
        api_key="<MY-API-KEY>",
    )
    client = StreamManagerClient.init(
        host="127.0.0.1",
        port=7070,
        operations_timeout=1.0,
        header_size=4,
        buffer_size=16438,
    )

    # when
    result = await client.initialise_pipeline(
        initialisation_request=initialisation_request
    )

    # then
    assert result == CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="new_pipeline"),
    )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_stream_manager_client_can_successfully_terminate_pipeline(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = assembly_socket_reader(
        message={
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
            "response": {"status": "success"},
        },
        header_size=4,
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    expected_command = {"type": CommandType.TERMINATE, "pipeline_id": "my_pipeline"}
    client = StreamManagerClient.init(
        host="127.0.0.1",
        port=7070,
        operations_timeout=1.0,
        header_size=4,
        buffer_size=16438,
    )

    # when
    result = await client.terminate_pipeline(pipeline_id="my_pipeline")

    # then
    assert result == CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )
    assert_correct_command_sent(
        writer=writer,
        command=expected_command,
        header_size=4,
        message="Expected termination command to be sent",
    )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_stream_manager_client_can_successfully_pause_pipeline(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = assembly_socket_reader(
        message={
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
            "response": {"status": "success"},
        },
        header_size=4,
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    expected_command = {"type": CommandType.MUTE, "pipeline_id": "my_pipeline"}
    client = StreamManagerClient.init(
        host="127.0.0.1",
        port=7070,
        operations_timeout=1.0,
        header_size=4,
        buffer_size=16438,
    )

    # when
    result = await client.pause_pipeline(pipeline_id="my_pipeline")

    # then
    assert result == CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )
    assert_correct_command_sent(
        writer=writer,
        command=expected_command,
        header_size=4,
        message="Expected pause command to be sent",
    )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_stream_manager_client_can_successfully_resume_pipeline(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = assembly_socket_reader(
        message={
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
            "response": {"status": "success"},
        },
        header_size=4,
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    expected_command = {"type": CommandType.RESUME, "pipeline_id": "my_pipeline"}
    client = StreamManagerClient.init(
        host="127.0.0.1",
        port=7070,
        operations_timeout=1.0,
        header_size=4,
        buffer_size=16438,
    )

    # when
    result = await client.resume_pipeline(pipeline_id="my_pipeline")

    # then
    assert result == CommandResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
    )
    assert_correct_command_sent(
        writer=writer,
        command=expected_command,
        header_size=4,
        message="Expected resume command to be sent",
    )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_stream_manager_client_can_successfully_get_pipeline_status(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = assembly_socket_reader(
        message={
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
            "response": {"status": "success", "report": {"my": "report"}},
        },
        header_size=4,
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    expected_command = {"type": CommandType.STATUS, "pipeline_id": "my_pipeline"}
    client = StreamManagerClient.init(
        host="127.0.0.1",
        port=7070,
        operations_timeout=1.0,
        header_size=4,
        buffer_size=16438,
    )

    # when
    result = await client.get_status(pipeline_id="my_pipeline")

    # then
    assert result == InferencePipelineStatusResponse(
        status="success",
        context=CommandContext(request_id="my_request", pipeline_id="my_pipeline"),
        report={"my": "report"},  # this is mock data
    )
    assert_correct_command_sent(
        writer=writer,
        command=expected_command,
        header_size=4,
        message="Expected get info command to be sent",
    )


@pytest.mark.asyncio
@mock.patch.object(stream_manager_client, "establish_socket_connection")
async def test_stream_manager_client_can_dispatch_error_response(
    establish_socket_connection_mock: AsyncMock,
) -> None:
    # given
    reader = assembly_socket_reader(
        message={
            "request_id": "my_request",
            "pipeline_id": "my_pipeline",
            "response": {"status": "failure", "error_type": "not_found"},
        },
        header_size=4,
    )
    writer = DummyStreamWriter()
    establish_socket_connection_mock.return_value = (reader, writer)
    expected_command = {"type": CommandType.RESUME, "pipeline_id": "my_pipeline"}
    client = StreamManagerClient.init(
        host="127.0.0.1",
        port=7070,
        operations_timeout=1.0,
        header_size=4,
        buffer_size=16438,
    )

    # when
    with pytest.raises(ProcessesManagerNotFoundError):
        _ = await client.resume_pipeline(pipeline_id="my_pipeline")

    # then

    assert_correct_command_sent(
        writer=writer,
        command=expected_command,
        header_size=4,
        message="Expected resume command to be sent",
    )


def assembly_socket_reader(message: dict, header_size: int) -> DummyStreamReader:
    serialised = json.dumps(message).encode("utf-8")
    response_payload = (
        len(serialised).to_bytes(length=header_size, byteorder="big") + serialised
    )
    return DummyStreamReader(read_buffer_content=response_payload)


def assert_correct_command_sent(
    writer: DummyStreamWriter, command: dict, header_size: int, message: str
) -> None:
    serialised_command = json.dumps(command).encode("utf-8")
    payload = (
        len(serialised_command).to_bytes(length=header_size, byteorder="big")
        + serialised_command
    )
    assert writer.get_content() == payload, message
