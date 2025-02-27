import json
import socket
from typing import Optional

from inference.core import logger
from inference.core.interfaces.stream_manager.manager_app.entities import ErrorType
from inference.core.interfaces.stream_manager.manager_app.errors import (
    MalformedHeaderError,
    MalformedPayloadError,
    TransmissionChannelClosed,
)
from inference.core.interfaces.stream_manager.manager_app.serialisation import (
    prepare_error_response,
)


def receive_socket_data(
    source: socket.socket, header_size: int, buffer_size: int
) -> dict:
    # Read the header part
    header = source.recv(header_size)
    if len(header) != header_size:
        raise MalformedHeaderError(
            private_message=f"Expected header size: {header_size}, received: {header}",
            public_message=f"Expected header size: {header_size}, received: {header}",
        )

    # Determine payload size from the header
    try:
        payload_size = int.from_bytes(header, byteorder="big")
    except ValueError as e:
        raise MalformedHeaderError(
            private_message="Header size could not convert to int: " + str(e),
            public_message="Header size could not convert to int",
        )

    if payload_size <= 0:
        raise MalformedHeaderError(
            private_message=f"Header is indicating a non-positive payload size: {payload_size}",
            public_message=f"Header is indicating a non-positive payload size: {payload_size}",
        )

    # Efficiently read the payload
    received = bytearray()
    remaining_payload_size = payload_size

    while remaining_payload_size > 0:
        chunk = source.recv(min(buffer_size, remaining_payload_size))
        if not chunk:
            raise TransmissionChannelClosed(
                private_message="Socket was closed before the payload was fully received.",
                public_message="Socket was closed before the payload was fully received.",
            )
        received.extend(chunk)
        remaining_payload_size -= len(
            chunk
        )  # Reduce the remaining size by the chunk size

    # Parse and return the JSON data
    try:
        return json.loads(received.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise MalformedPayloadError(
            public_message="Received payload that is not in a JSON format",
            private_message="Received payload that is not in a JSON format",
            inner_error=error,
        )


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
