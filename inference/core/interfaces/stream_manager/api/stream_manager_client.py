import asyncio
import json
from asyncio import StreamReader, StreamWriter
from enum import Enum
from json import JSONDecodeError
from typing import List, Optional, Tuple, Union

from inference.core import logger
from inference.core.interfaces.stream_manager.api.entities import (
    CommandContext,
    CommandResponse,
    ConsumePipelineResponse,
    FrameMetadata,
    InferencePipelineStatusResponse,
    InitializeWebRTCPipelineResponse,
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
from inference.core.interfaces.stream_manager.manager_app.entities import (
    ERROR_TYPE_KEY,
    PIPELINE_ID_KEY,
    REQUEST_ID_KEY,
    RESPONSE_KEY,
    STATUS_KEY,
    TYPE_KEY,
    CommandType,
    ErrorType,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
    OperationStatus,
)
from inference.core.interfaces.stream_manager.manager_app.errors import (
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
    ErrorType.NOT_FOUND.value: ProcessesManagerNotFoundError,
    ErrorType.OPERATION_ERROR.value: ProcessesManagerOperationError,
    ErrorType.AUTHORISATION_ERROR.value: ProcessesManagerAuthorisationError,
}


class StreamManagerClient:
    @classmethod
    def init(
        cls,
        host: str,
        port: int,
        operations_timeout: Optional[float] = None,
        header_size: int = HEADER_SIZE,
        buffer_size: int = BUFFER_SIZE,
    ) -> "StreamManagerClient":
        return cls(
            host=host,
            port=port,
            operations_timeout=operations_timeout,
            header_size=header_size,
            buffer_size=buffer_size,
        )

    def __init__(
        self,
        host: str,
        port: int,
        operations_timeout: Optional[float],
        header_size: int,
        buffer_size: int,
    ):
        self._host = host
        self._port = port
        self._operations_timeout = operations_timeout
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

    async def initialise_webrtc_pipeline(
        self, initialisation_request: InitialiseWebRTCPipelinePayload
    ) -> InitializeWebRTCPipelineResponse:
        command = initialisation_request.dict(exclude_none=True)
        command[TYPE_KEY] = CommandType.WEBRTC
        response = await self._handle_command(command=command)
        status = response[RESPONSE_KEY][STATUS_KEY]
        context = CommandContext(
            request_id=response.get(REQUEST_ID_KEY),
            pipeline_id=response.get(PIPELINE_ID_KEY),
        )
        return InitializeWebRTCPipelineResponse(
            status=status,
            context=context,
            sdp=response["response"]["sdp"],
            type=response["response"]["type"],
        )

    async def initialise_pipeline(
        self, initialisation_request: InitialisePipelinePayload
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

    async def consume_pipeline_result(
        self,
        pipeline_id: str,
        excluded_fields: List[str],
    ) -> ConsumePipelineResponse:
        command = {
            TYPE_KEY: CommandType.CONSUME_RESULT,
            PIPELINE_ID_KEY: pipeline_id,
            "excluded_fields": excluded_fields,
        }
        response = await self._handle_command(command=command)
        status = response[RESPONSE_KEY][STATUS_KEY]
        context = CommandContext(
            request_id=response.get(REQUEST_ID_KEY),
            pipeline_id=response.get(PIPELINE_ID_KEY),
        )
        return ConsumePipelineResponse(
            status=status,
            context=context,
            outputs=response[RESPONSE_KEY]["outputs"],
            frames_metadata=[
                FrameMetadata.model_validate(f)
                for f in response[RESPONSE_KEY]["frames_metadata"]
            ],
        )

    async def _handle_command(self, command: dict) -> dict:
        response = await send_command(
            host=self._host,
            port=self._port,
            command=command,
            header_size=self._header_size,
            buffer_size=self._buffer_size,
            timeout=self._operations_timeout,
        )
        if is_request_unsuccessful(response=response):
            dispatch_error(error_response=response)
        return response


async def send_command(
    host: str,
    port: int,
    command: dict,
    header_size: int,
    buffer_size: int,
    timeout: Optional[float] = None,
) -> dict:
    try:
        reader, writer = await establish_socket_connection(
            host=host, port=port, timeout=timeout
        )
        await send_message(
            writer=writer, message=command, header_size=header_size, timeout=timeout
        )
        data = await receive_message(
            reader, header_size=header_size, buffer_size=buffer_size, timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return json.loads(data)
    except JSONDecodeError as error:
        raise MalformedPayloadError(
            private_message=f"Could not decode response. Cause: {error}",
            public_message=f"Could not decode response from InferencePipeline Manager",
            inner_error=error,
        ) from error
    except (OSError, asyncio.TimeoutError) as error:
        raise ConnectivityError(
            private_message=f"Could not communicate with InferencePipeline Manager",
            public_message="Could not establish communication with InferencePipeline Manager",
            inner_error=error,
        ) from error


async def establish_socket_connection(
    host: str, port: int, timeout: Optional[float] = None
) -> Tuple[StreamReader, StreamWriter]:
    return await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)


async def send_message(
    writer: StreamWriter,
    message: dict,
    header_size: int,
    timeout: Optional[float] = None,
) -> None:
    try:
        body = json.dumps(message, default=_json_serializer).encode("utf-8")
        header = len(body).to_bytes(length=header_size, byteorder="big")
        payload = header + body
        writer.write(payload)
        await asyncio.wait_for(writer.drain(), timeout=timeout)
    except (TypeError, ValueError) as error:
        raise MalformedPayloadError(
            private_message=f"Could not serialise message. Details: {error}",
            public_message="Could not serialise payload of command that should be sent to InferencePipeline Manager",
            inner_error=error,
        ) from error
    except OverflowError as error:
        raise MessageToBigError(
            private_message=f"Could not send message due to size overflow. Details: {error}",
            public_message="InferencePipeline Manager command payload to big.",
            inner_error=error,
        ) from error
    except asyncio.TimeoutError as error:
        raise ConnectivityError(
            private_message=f"Could not communicate with InferencePipeline Manager. Error: {error}",
            public_message="Could not communicate with InferencePipeline Manager.",
            inner_error=error,
        ) from error
    except Exception as error:
        raise CommunicationProtocolError(
            private_message=f"Could not send message to InferencePipeline Manager. Cause: {error}",
            public_message="Unknown communication error while sending message to InferencePipeline Manager.",
            inner_error=error,
        ) from error


def _json_serializer(o: object) -> str:
    if isinstance(o, Enum):
        return o.value
    raise ValueError(f"Could not serialise object: {o}")


async def receive_message(
    reader: StreamReader,
    header_size: int,
    buffer_size: int,
    timeout: Optional[float] = None,
) -> bytes:
    header = await asyncio.wait_for(reader.read(header_size), timeout=timeout)
    if len(header) != header_size:
        raise MalformedHeaderError(
            private_message="Header size missmatch",
            public_message="Internal error in communication with InferencePipeline Manager. Violation of "
            "communication protocol - malformed header of message.",
        )
    payload_size = int.from_bytes(bytes=header, byteorder="big")
    received = b""
    while len(received) < payload_size:
        chunk = await asyncio.wait_for(reader.read(buffer_size), timeout=timeout)
        if len(chunk) == 0:
            raise TransmissionChannelClosed(
                private_message="Socket was closed to read before payload was decoded.",
                public_message="Internal error in communication with InferencePipeline Manager. Could not receive full "
                "message.",
            )
        received += chunk
    return received


def is_request_unsuccessful(response: dict) -> bool:
    return (
        response.get(RESPONSE_KEY, {}).get(STATUS_KEY, OperationStatus.FAILURE.value)
        != OperationStatus.SUCCESS.value
    )


def dispatch_error(error_response: dict) -> None:
    response_payload = error_response.get(RESPONSE_KEY, {})
    error_type = response_payload.get(ERROR_TYPE_KEY)
    error_class = response_payload.get("error_class", "N/A")
    error_message = response_payload.get("error_message", "N/A")
    public_error_message = response_payload.get("public_error_message", "N/A")
    logger.error(
        f"Error with command handling raised by InferencePipeline Manager. "
        f"error_type={error_type} error_class={error_class} "
        f"error_message={error_message}"
    )
    if error_type in ERRORS_MAPPING:
        raise ERRORS_MAPPING[error_type](
            private_message=f"Error with command handling raised by InferencePipeline Manager. "
            f"Error type: {error_type}. Details: {error_message}",
            public_message=f"Error with command handling raised by InferencePipeline Manager. "
            f"Error type: {error_type}. Details: {public_error_message}",
        )
    raise ProcessesManagerClientError(
        private_message=f"Unknown error with command handling raised by InferencePipeline Manager. "
        f"Error type: {error_type}. Details: {error_message}",
        public_message=f"Unknown error with command handling raised by InferencePipeline Manager. "
        f"Raised error type: {error_type}. Details: {public_error_message}",
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
