from unittest.mock import MagicMock

from inference.core.interfaces.stream_manager.manager_app.tcp_server import (
    RoboflowTCPServer,
)


def test_roboflow_server_applies_connection_timeout() -> None:
    # given
    server = RoboflowTCPServer(
        server_address=("127.0.0.1", 7070),
        handler_class=MagicMock,
        socket_operations_timeout=1.5,
    )
    connection, address = MagicMock(), MagicMock()
    server.socket = MagicMock()
    server.socket.accept.return_value = (connection, address)

    # when
    result = server.get_request()

    # then
    connection.settimeout.assert_called_once_with(1.5)
    assert result == (
        connection,
        address,
    ), "Method must return accepted connection and address, as per TCPServer interface requirement"
