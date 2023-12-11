import socket
from socketserver import TCPServer, BaseRequestHandler
from typing import Tuple, Type, Optional, Any


class RoboflowTCPServer(TCPServer):
    def __init__(
        self,
        server_address: Tuple[str, int],
        handler_class: Type[BaseRequestHandler],
        socket_operations_timeout: Optional[float] = None,
    ):
        TCPServer.__init__(self, server_address, handler_class)
        self._socket_operations_timeout = socket_operations_timeout

    def get_request(self) -> Tuple[socket.socket, Any]:
        connection, address = self.socket.accept()
        connection.settimeout(self._socket_operations_timeout)
        return connection, address
