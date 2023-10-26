import os

from fastapi.testclient import TestClient
import pytest

from inference.enterprise.device_manager.device_manager import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_environment_variables(mocker):
    mocker.patch.dict(
        os.environ,
        {
            "DEVICE_MANAGER_USERNAME": "username",
            "DEVICE_MANAGER_PASSWORD": "password",
            "DEVICE_MANAGER_PUBSUB_HOST": "foo",
        },
    )


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Roboflow Device Manager"


def test_adds_jobs_for_metrics_and_inferences():
    # Test the scheduler jobs
    from inference.enterprise.device_manager.device_manager import (
        scheduler,
        send_metrics,
        send_latest_inferences,
    )

    # Check if the scheduler jobs are added
    assert len(scheduler.get_jobs()) == 2

    # Check if the job functions are callable
    for job in scheduler.get_jobs():
        assert job.func == send_metrics or job.func == send_latest_inferences
