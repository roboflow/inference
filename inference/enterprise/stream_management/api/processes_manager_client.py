import asyncio
import json
from asyncio import StreamReader, StreamWriter

from inference.core import logger
from inference.enterprise.stream_management.api.entities import (
    CommandContext,
    CommandResponse,
    InferencePipelineStatusResponse,
    ListPipelinesResponse,
    PipelineInitialisationRequest,
)
from inference.enterprise.stream_management.api.errors import (
    ConnectivityError,
    ProcessesManagerAuthorisationError,
    ProcessesManagerClientError,
    ProcessesManagerInternalError,
    ProcessesManagerInvalidPayload,
    ProcessesManagerNotFound,
    ProcessesManagerOperationError,
)
from inference.enterprise.stream_management.manager.entities import (
    ERROR_TYPE_KEY,
    PIPELINE_ID_KEY,
    REQUEST_ID_KEY,
    RESPONSE_KEY,
    STATUS_KEY,
    TYPE_KEY,
    CommandType,
    ErrorType,
    OperationStatus,
)
from inference.enterprise.stream_management.manager.errors import (
    CommunicationProtocolError,
    MalformedHeaderError,
    MalformedPayloadError,
    MessageToBigError,
    TransmissionChannelClosed,
)

BUFFER_SIZE = 16384
HEADER_SIZE = 4

ERRORS_MAPPING = {
    ErrorType.INTERNAL_ERROR.value: ProcessesManagerInternalError,
    ErrorType.INVALID_PAYLOAD.value: ProcessesManagerInvalidPayload,
    ErrorType.NOT_FOUND.value: ProcessesManagerNotFound,
    ErrorType.OPERATION_ERROR.value: ProcessesManagerOperationError,
    ErrorType.AUTHORISATION_ERROR.value: ProcessesManagerAuthorisationError,
}


class ProcessesManagerClient:
    def __init__(
        self,
        host: str,
        port: int,
        header_size: int = HEADER_SIZE,
        buffer_size: int = BUFFER_SIZE,
    ):
        self._host = host
        self._port = port
        self._header_size = header_size
        self._buffer_size = buffer_size

    async def list_pipelines(self) -> ListPipelinesResponse:
        command = {
            TYPE_KEY: CommandType.LIST_PIPELINES,
        }
        response = await self._handle_command(command=command)
        status = response[RESPONSE_KEY][STATUS_KEY]
        context = CommandContext(
            request_id=response.get(REQUEST_ID_KEY),
            pipeline_id=response.get(PIPELINE_ID_KEY),
        )
        pipelines = response[RESPONSE_KEY]["pipelines"]
        return ListPipelinesResponse(
            status=status,
            context=context,
            pipelines=pipelines,
        )

    async def initialise_pipeline(
        self, initialisation_request: PipelineInitialisationRequest
    ) -> CommandResponse:
        command = initialisation_request.dict(exclude_none=True)
        command[TYPE_KEY] = CommandType.INIT
        response = await self._handle_command(command=command)
        return build_response(response=response)

    async def terminate_pipeline(self, pipeline_id: str) -> CommandResponse:
        command = {
            TYPE_KEY: CommandType.TERMINATE,
            PIPELINE_ID_KEY: pipeline_id,
        }
        response = await self._handle_command(command=command)
        return build_response(response=response)

    async def pause_pipeline(self, pipeline_id: str) -> CommandResponse:
        command = {
            TYPE_KEY: CommandType.MUTE,
            PIPELINE_ID_KEY: pipeline_id,
        }
        response = await self._handle_command(command=command)
        return build_response(response=response)

    async def resume_pipeline(self, pipeline_id: str) -> CommandResponse:
        command = {
            TYPE_KEY: CommandType.RESUME,
            PIPELINE_ID_KEY: pipeline_id,
        }
        response = await self._handle_command(command=command)
        return build_response(response=response)

    async def get_status(self, pipeline_id: str) -> InferencePipelineStatusResponse:
        command = {
            TYPE_KEY: CommandType.STATUS,
            PIPELINE_ID_KEY: pipeline_id,
        }
        response = await self._handle_command(command=command)
        status = response[RESPONSE_KEY][STATUS_KEY]
        context = CommandContext(
            request_id=response.get(REQUEST_ID_KEY),
            pipeline_id=response.get(PIPELINE_ID_KEY),
        )
        report = response[RESPONSE_KEY]["report"]
        return InferencePipelineStatusResponse(
            status=status,
            context=context,
            report=report,
        )

    async def _handle_command(self, command: dict) -> dict:
        response = await send_command(
            host=self._host,
            port=self._port,
            command=command,
            header_size=self._header_size,
            buffer_size=self._buffer_size,
        )
        if (
            response.get(RESPONSE_KEY, {}).get(
                STATUS_KEY, OperationStatus.FAILURE.value
            )
            == OperationStatus.FAILURE.value
        ):
            dispatch_error(error_response=response)
        return response


async def send_command(
    host: str, port: int, command: dict, header_size: int, buffer_size: int
) -> dict:
    try:
        print(f"Connecting to: {host}, {port}")
        reader, writer = await asyncio.open_connection(host, port)
        await send_message(writer=writer, message=command)
        data = await receive_message(
            reader, header_size=header_size, buffer_size=buffer_size
        )
        writer.close()
        await writer.wait_closed()
        return json.loads(data)
    except (ConnectionError, TimeoutError) as errors:
        raise ConnectivityError(
            f"Could not communicate with Process Manager"
        ) from errors


async def send_message(writer: StreamWriter, message: dict) -> None:
    try:
        body = json.dumps(message).encode("utf-8")
        header = len(body).to_bytes(length=4, byteorder="big")
        payload = header + body
        writer.write(payload)
        await writer.drain()
    except TypeError as error:
        raise MalformedPayloadError(f"Could not serialise message. Details: {error}")
    except OverflowError as error:
        raise MessageToBigError(
            f"Could not send message due to size overflow. Details: {error}"
        )
    except Exception as error:
        raise CommunicationProtocolError(
            f"Could not send message. Cause: {error}"
        ) from error


async def receive_message(
    reader: StreamReader, header_size: int, buffer_size: int
) -> bytes:
    header = await reader.read(header_size)
    if len(header) != header_size:
        raise MalformedHeaderError("Header size missmatch")
    payload_size = int.from_bytes(bytes=header, byteorder="big")
    received = b""
    while len(received) < payload_size:
        chunk = await reader.read(buffer_size)
        if len(chunk) == 0:
            raise TransmissionChannelClosed(
                "Socket was closed to read before payload was decoded."
            )
        received += chunk
    return received


def dispatch_error(error_response: dict) -> None:
    response_payload = error_response.get(RESPONSE_KEY, {})
    error_type = response_payload.get(ERROR_TYPE_KEY)
    error_class = response_payload.get("error_class")
    error_message = response_payload.get("error_message")
    logger.error(
        f"Error in ProcessesManagerClient. error_type={error_type} error_class={error_class} "
        f"error_message={error_message}"
    )
    if error_type in ERRORS_MAPPING:
        raise ERRORS_MAPPING[error_type](
            f"Error in ProcessesManagerClient. Error type: {error_type}. Details: {error_message}"
        )
    raise ProcessesManagerClientError(
        f"Error in ProcessesManagerClient. Error type: {error_type}. Details: {error_message}"
    )


def build_response(response: dict) -> CommandResponse:
    status = response[RESPONSE_KEY][STATUS_KEY]
    context = CommandContext(
        request_id=response.get(REQUEST_ID_KEY),
        pipeline_id=response.get(PIPELINE_ID_KEY),
    )
    return CommandResponse(
        status=status,
        context=context,
    )
