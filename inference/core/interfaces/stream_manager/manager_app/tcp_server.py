import socket
from socketserver import BaseRequestHandler, TCPServer
from typing import Any, Optional, Tuple, Type


class RoboflowTCPServer(TCPServer):
    # Matches `http.server.HTTPServer`: lets the manager rebind while a socket from a
    # previous instance lingers in TIME_WAIT. This does not permit two live listeners
    # on the same address, so a second manager still fails to bind, as intended.
    allow_reuse_address = True

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
