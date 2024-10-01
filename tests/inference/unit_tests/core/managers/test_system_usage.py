import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from inference.core.interfaces.http.http_api import HttpInterface
from inference.core.managers.base import ModelManager
import json


@pytest.fixture
def client():
    model_registry = MagicMock()
    model_manager = ModelManager(model_registry=model_registry)
    interface = HttpInterface(model_manager)
    return TestClient(interface.app)


@patch("inference.core.utils.container.subprocess.run")
@patch("inference.core.utils.container.socket.gethostname")
def test_device_stats(mock_gethostname, mock_subprocess_run, client):
    mock_gethostname.return_value = "mock_container_id"
    mock_subprocess_run.return_value.stdout = json.dumps(
        {
            "cpu_stats": {"cpu_usage": {"total_usage": 123456}},
            "memory_stats": {"usage": 654321},
        }
    )

    response = client.get("/device/stats")
    assert response.status_code == 200
    assert response.json() == {
        "status": 200,
        "message": "done",
        "container_stats": {
            "cpu_stats": {"cpu_usage": {"total_usage": 123456}},
            "memory_stats": {"usage": 654321},
        },
    }


@patch("inference.core.utils.container.subprocess.run")
@patch("inference.core.utils.container.socket.gethostname")
def test_device_stats_error(mock_gethostname, mock_subprocess_run, client):
    mock_gethostname.return_value = "mock_container_id"
    mock_subprocess_run.return_value.stdout = ""

    response = client.get("/device/stats")
    assert response.status_code == 200
    assert response.json() == {
        "status": 200,
        "message": "done",
        "container_stats": None,
    }
