from unittest import mock
from unittest.mock import AsyncMock, MagicMock

from starlette.testclient import TestClient

from inference.core.workflows.core_steps.sinks.webhook import v1 as webhook_v1


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _build_test_client(monkeypatch) -> TestClient:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    interface = http_api.HttpInterface(model_manager=model_manager)
    return TestClient(interface.app)


WEBHOOK_WORKFLOW = {
    "version": "1.0",
    "inputs": [],
    "steps": [
        {
            "type": "roboflow_core/webhook_sink@v1",
            "name": "webhook",
            "url": "https://example.com",
            "method": "POST",
            "fire_and_forget": False,
            "disable_sink": False,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.webhook.message",
        }
    ],
}


@mock.patch.object(webhook_v1, "execute_request")
def test_workflow_run_injects_sink_disabling_policy(
    execute_request_mock,
    monkeypatch,
) -> None:
    client = _build_test_client(monkeypatch)

    response = client.post(
        "/workflows/run",
        json={
            "specification": WEBHOOK_WORKFLOW,
            "inputs": {},
            "disable_sinks": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["outputs"] == [
        {"message": "Sink was disabled by workflow execution policy"}
    ]
    execute_request_mock.assert_not_called()


@mock.patch.object(webhook_v1, "execute_request")
def test_workflow_run_keeps_sinks_enabled_by_default(
    execute_request_mock,
    monkeypatch,
) -> None:
    execute_request_mock.return_value = (False, "Notification sent successfully")
    client = _build_test_client(monkeypatch)

    response = client.post(
        "/workflows/run",
        json={
            "specification": WEBHOOK_WORKFLOW,
            "inputs": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["outputs"] == [{"message": "Notification sent successfully"}]
    execute_request_mock.assert_called_once()
