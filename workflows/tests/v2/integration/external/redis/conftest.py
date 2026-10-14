"""Fixtures for the external Redis suites: an owned, throwaway Redis server.

The server listens only on a Unix socket in a fresh ``/tmp`` directory
(``--port 0``), with persistence off. Tests start and stop exactly this
process; no other Redis server, port or data is touched. Redis tests skip
when ``redis-server`` or the ``redis`` package is unavailable.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from tests.v2.support.state._redis_server import OwnedRedisServer
from tests.v2.support.state.fixtures import namespace  # noqa: F401


@pytest.fixture(scope="session")
def redis_server() -> Iterator[OwnedRedisServer]:
    if shutil.which("redis-server") is None:
        pytest.skip("redis-server is not installed")
    pytest.importorskip("redis")

    # Short /tmp path: Unix socket paths are limited to about 100 bytes.
    directory = Path(tempfile.mkdtemp(prefix="wf2s-", dir="/tmp"))
    server = OwnedRedisServer(directory)
    try:
        server.wait_ready()
        yield server
    finally:
        server.stop()
        shutil.rmtree(directory, ignore_errors=True)
