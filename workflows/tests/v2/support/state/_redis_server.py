"""Owned, throwaway ``redis-server`` processes for managed state tests.

Each server listens only on a Unix socket in its own directory (``--port 0``)
with persistence off. ``stop`` ends exactly that process.
"""

import shutil
import subprocess
import time
from pathlib import Path

_START_TIMEOUT_SECONDS = 10.0


class OwnedRedisServer:
    """A ``redis-server`` process owned by the test session."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.socket_path = directory / "redis.sock"
        self.url = f"unix://{self.socket_path}"
        self._process = subprocess.Popen(
            [
                shutil.which("redis-server"),
                "--port",
                "0",
                "--unixsocket",
                str(self.socket_path),
                "--unixsocketperm",
                "700",
                "--save",
                "",
                "--appendonly",
                "no",
                "--dir",
                str(directory),
                "--logfile",
                str(directory / "redis.log"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def wait_ready(self) -> None:
        import redis

        deadline = time.monotonic() + _START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError("owned redis-server exited during start")
            try:
                client = redis.Redis(unix_socket_path=str(self.socket_path))
                client.ping()
                client.close()
                return
            except redis.exceptions.ConnectionError:
                time.sleep(0.02)

        raise RuntimeError("owned redis-server did not become ready")

    def admin(self):
        import redis

        client = redis.Redis(unix_socket_path=str(self.socket_path))

        return client

    def stop(self) -> None:
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
