import json
import socket
from typing import Optional

from inference.core import logger
from inference.enterprise.stream_management.manager.entities import ErrorType
from inference.enterprise.stream_management.manager.errors import (
    MalformedHeaderError,
    MalformedPayloadError,
    TransmissionChannelClosed,
)
from inference.enterprise.stream_management.manager.serialisation import (
    prepare_error_response,
)


def receive_socket_data(
    source: socket.socket, header_size: int, buffer_size: int
) -> dict:
    header = source.recv(header_size)
    if len(header) != header_size:
        raise MalformedHeaderError(
            f"Expected header size: {header_size}, received: {header}"
        )
    payload_size = int.from_bytes(bytes=header, byteorder="big")
    if payload_size <= 0:
        raise MalformedHeaderError(
            f"Header is indicating non positive payload size: {payload_size}"
        )
    received = b""
    while len(received) < payload_size:
        chunk = source.recv(buffer_size)
        if len(chunk) == 0:
            raise TransmissionChannelClosed(
                "Socket was closed to read before payload was decoded."
            )
        received += chunk
    try:
        return json.loads(received)
    except ValueError:
        raise MalformedPayloadError("Received payload that is not in a JSON format")


def send_data_trough_socket(
    target: socket.socket,
    header_size: int,
    data: bytes,
    request_id: str,
    recover_from_overflow: bool = True,
    pipeline_id: Optional[str] = None,
) -> None:
    try:
        data_size = len(data)
        header = data_size.to_bytes(length=header_size, byteorder="big")
        payload = header + data
        target.sendall(payload)
    except OverflowError as error:
        if not recover_from_overflow:
            logger.error(f"OverflowError was suppressed. {error}")
            return None
        error_response = prepare_error_response(
            request_id=request_id,
            error=error,
            error_type=ErrorType.INTERNAL_ERROR,
            pipeline_id=pipeline_id,
        )
        send_data_trough_socket(
            target=target,
            header_size=header_size,
            data=error_response,
            request_id=request_id,
            recover_from_overflow=False,
            pipeline_id=pipeline_id,
        )
    except Exception as error:
        logger.error(f"Could not send the response through socket. Error: {error}")
