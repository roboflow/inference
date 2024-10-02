import json
import socket
import subprocess

from inference.core.logger import logger


def get_container_id():
    try:
        return socket.gethostname()
    except Exception as e:
        logger.warn(f"Error getting container ID: {e}")
        return None


def get_container_stats():
    container_id = get_container_id()
    if container_id:
        result = subprocess.run(
            [
                "curl",
                "--unix-socket",
                "/var/run/docker.sock",
                f"http://localhost/containers/{container_id}/stats?stream=false",
            ],
            capture_output=True,
            text=True,
        )
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warn(f"Error parsing container stats: {result.stdout}")
            return None
    return None
