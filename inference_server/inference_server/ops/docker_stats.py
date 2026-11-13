import http.client
import json
import logging
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

DOCKER_STATS_TIMEOUT_S = 10.0
DOCKER_STATS_MAX_RESPONSE_BYTES = 1024 * 1024
CONTAINER_STATS_ERROR_MESSAGE = "An error occurred while fetching container stats."


class ContainerStatsError(Exception):
    pass


def is_docker_socket_mounted(docker_socket_path: str) -> bool:
    mounted = Path(docker_socket_path).is_socket()

    return mounted


def get_container_stats(
    docker_socket_path: str, *, timeout: float = DOCKER_STATS_TIMEOUT_S
) -> dict:
    status_code = None
    try:
        container_id = socket.gethostname()
        connection = http.client.HTTPConnection("localhost", timeout=timeout)
        connection.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.sock.settimeout(timeout)
            connection.sock.connect(docker_socket_path)
            connection.request(
                "GET",
                f"/containers/{container_id}/stats?stream=false",
                headers={"Host": "localhost"},
            )
            response = connection.getresponse()
            data = response.read(DOCKER_STATS_MAX_RESPONSE_BYTES + 1)
        finally:
            connection.close()
        if len(data) > DOCKER_STATS_MAX_RESPONSE_BYTES:
            raise ValueError("Docker answer exceeds the size limit")
        if response.status != 200:
            status_code = response.status
            raise RuntimeError("Docker answered with a non-200 status")

        stats = json.loads(data.decode())
    except Exception as error:
        logger.error(
            "Could not fetch container stats (error_type=%s, status_code=%s)",
            type(error).__name__,
            status_code,
        )
        raise ContainerStatsError(CONTAINER_STATS_ERROR_MESSAGE) from error

    return {"stats": stats}
