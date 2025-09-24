"""
In this module, tests are written to mock, avoiding issues with trying to find empty socket in tests
"""

import json
import random
from unittest.mock import MagicMock

import pytest

from inference.enterprise.stream_management.manager.communication import (
    receive_socket_data,
    send_data_trough_socket,
)
from inference.enterprise.stream_management.manager.errors import (
    MalformedHeaderError,
    MalformedPayloadError,
    TransmissionChannelClosed,
)


def test_receive_socket_data_when_header_is_malformed() -> None:
    # given
    socket = MagicMock()
    socket.recv.side_effect = [b"A"]

    # when
    with pytest.raises(MalformedHeaderError):
        _ = receive_socket_data(
            source=socket,
            header_size=4,
            buffer_size=512,
        )


def test_receive_socket_data_when_header_cannot_be_decoded_as_valid_value() -> None:
    # given
    socket = MagicMock()
    zero = 0
    socket.recv.side_effect = [zero.to_bytes(length=4, byteorder="big")]

    # when
    with pytest.raises(MalformedHeaderError):
        _ = receive_socket_data(
            source=socket,
            header_size=4,
            buffer_size=512,
        )


def test_receive_socket_data_when_header_indicated_invalid_payload_length() -> None:
    # given
    socket = MagicMock()
    data = json.dumps({"some": "data"}).encode("utf-8")
    header = len(data) + 32
    socket.recv.side_effect = [header.to_bytes(length=4, byteorder="big"), data, b""]

    # when
    with pytest.raises(TransmissionChannelClosed):
        _ = receive_socket_data(
            source=socket,
            header_size=4,
            buffer_size=len(data),
        )


def test_receive_socket_data_when_malformed_payload_given() -> None:
    # given
    socket = MagicMock()
    data = "FOR SURE NOT A JSON :)".encode("utf-8")
    header = len(data)
    socket.recv.side_effect = [header.to_bytes(length=4, byteorder="big"), data]

    # when
    with pytest.raises(MalformedPayloadError):
        _ = receive_socket_data(
            source=socket,
            header_size=4,
            buffer_size=len(data),
        )


def test_receive_socket_data_complete_successfully_despite_fragmented_message() -> None:
    # given
    socket = MagicMock()
    data = json.dumps({"some": "data"}).encode("utf-8")
    header = len(data)
    socket.recv.side_effect = [
        header.to_bytes(length=4, byteorder="big"),
        data[:-3],
        data[-3:],
    ]

    # when
    result = receive_socket_data(
        source=socket,
        header_size=4,
        buffer_size=len(data) - 3,
    )

    # then
    assert result == {"some": "data"}, "Decoded date must be equal to input payload"


def test_receive_socket_data_when_timeout_error_should_be_reraised() -> None:
    # given
    socket = MagicMock()
    data = json.dumps({"some": "data"}).encode("utf-8")
    header = len(data)
    socket.recv.side_effect = [header.to_bytes(length=4, byteorder="big"), TimeoutError]

    # when
    with pytest.raises(TimeoutError):
        _ = receive_socket_data(
            source=socket,
            header_size=4,
            buffer_size=len(data),
        )


def test_send_data_trough_socket_when_operation_succeeds() -> None:
    # given
    socket = MagicMock()
    payload = json.dumps({"my": "data"}).encode("utf-8")

    # when
    send_data_trough_socket(
        target=socket,
        header_size=4,
        data=payload,
        request_id="my_request",
        pipeline_id="my_pipeline",
    )

    # then
    socket.sendall.assert_called_once_with(
        len(payload).to_bytes(length=4, byteorder="big") + payload
    )


def test_send_data_trough_socket_when_payload_overflow_happens() -> None:
    # given
    socket = MagicMock()
    payload = json.dumps(
        {"my": "data", "list": [random.randint(0, 100) for _ in range(128)]}
    ).encode("utf-8")
    expected_error_payload = json.dumps(
        {
            "request_id": "my_request",
            "response": {
                "status": "failure",
                "error_type": "internal_error",
                "error_class": "OverflowError",
                "error_message": "int too big to convert",
            },
            "pipeline_id": "my_pipeline",
        }
    ).encode("utf-8")

    # when
    send_data_trough_socket(
        target=socket,
        header_size=1,
        data=payload,
        request_id="my_request",
        pipeline_id="my_pipeline",
    )

    # then
    socket.sendall.assert_called_once_with(
        len(expected_error_payload).to_bytes(length=1, byteorder="big")
        + expected_error_payload
    )


def test_send_data_trough_socket_when_connection_error_occurs() -> None:
    # given
    socket = MagicMock()
    payload = json.dumps({"my": "data"}).encode("utf-8")

    # when
    send_data_trough_socket(
        target=socket,
        header_size=4,
        data=payload,
        request_id="my_request",
        pipeline_id="my_pipeline",
    )

    # then: Nothing happens - error just logged
