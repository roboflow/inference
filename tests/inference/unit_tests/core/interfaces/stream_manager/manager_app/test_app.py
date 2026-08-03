import errno
import socket
from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.stream_manager.manager_app import app as manager_app


def test_start_exits_gracefully_when_another_manager_already_bound_the_port() -> None:
    # given - a manager started by another HTTP worker already owns the address
    occupied_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied_socket.bind(("127.0.0.1", 0))
    occupied_socket.listen(1)
    host, port = occupied_socket.getsockname()
    warm_up_pipelines = MagicMock()
    original_host, original_port = manager_app.HOST, manager_app.PORT
    original_warm_up = manager_app.ensure_idle_pipelines_warmed_up
    manager_app.HOST, manager_app.PORT = host, port
    manager_app.ensure_idle_pipelines_warmed_up = warm_up_pipelines

    # when
    try:
        manager_app.start(expected_warmed_up_pipelines=1)
    finally:
        manager_app.HOST, manager_app.PORT = original_host, original_port
        manager_app.ensure_idle_pipelines_warmed_up = original_warm_up
        occupied_socket.close()

    # then - the instance that lost the race must exit before warming up pipelines,
    # otherwise every worker would preload models
    warm_up_pipelines.assert_not_called()


def test_start_reraises_os_errors_other_than_address_in_use() -> None:
    # given
    def _raise_permission_denied(**kwargs) -> None:
        raise OSError(errno.EACCES, "Permission denied")

    original_server = manager_app.RoboflowTCPServer
    manager_app.RoboflowTCPServer = _raise_permission_denied

    # when
    try:
        with pytest.raises(OSError) as error:
            manager_app.start()
    finally:
        manager_app.RoboflowTCPServer = original_server

    # then
    assert (
        error.value.errno == errno.EACCES
    ), "Only EADDRINUSE is expected to be handled, other socket errors must surface"
