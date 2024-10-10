import os
import stat


def is_docker_socket_mounted(docker_socket_path: str) -> bool:
    """
    Check if the given path is a mounted Docker socket.

    Args:
        docker_socket_path (str): The path to the socket file.

    Returns:
        bool: True if the path is a Unix socket, False otherwise.
    """
    if os.path.exists(docker_socket_path):
        socket_stat = os.stat(docker_socket_path)
        if stat.S_ISSOCK(socket_stat.st_mode):
            return True
    return False
