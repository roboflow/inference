import asyncio
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inference_server import configuration
from inference_server.ops import memory_logs
from inference_server.ops.router import include_ops_routers
from tests.unit_tests.test_app_routes import reloaded_app  # noqa: F401

OPS_FLAGS = {
    "LAMBDA": False,
    "GCP_SERVERLESS": False,
    "DOCKER_SOCKET_PATH": None,
    "SECURE_GATEWAY_HEALTH_ENDPOINT_ENABLED": False,
    "SECURE_GATEWAY_HEALTH_CHECK_TIMEOUT": 5.0,
    "ROBOFLOW_API_VERIFY_SSL": True,
    "NOTEBOOK_ENABLED": False,
    "NOTEBOOK_PORT": 9002,
    "NOTEBOOK_PASSWORD": None,
    "ENABLE_IN_MEMORY_LOGS": False,
}


@pytest.fixture
def ops_client(monkeypatch):
    def _build(**overrides) -> TestClient:
        for name, value in {**OPS_FLAGS, **overrides}.items():
            monkeypatch.setattr(configuration, name, value)
        app = FastAPI()
        include_ops_routers(app)

        return TestClient(app, follow_redirects=False)

    return _build


@pytest.fixture
def log_buffer():
    memory_logs._log_entries.clear()
    root = logging.getLogger("inference_server.ops")
    root.addHandler(memory_logs._memory_handler)
    yield lambda: "\n".join(
        entry["message"] for entry in memory_logs.get_recent_logs(0)
    )
    root.removeHandler(memory_logs._memory_handler)
    memory_logs._log_entries.clear()


def running_on_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False

    return True
