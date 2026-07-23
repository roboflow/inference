from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from requests import Response
from starlette.testclient import TestClient

from inference.core.workflows.core_steps.sinks.webhook import v1 as webhook_v1


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


def _build_test_client(monkeypatch, disable_sinks_by_environment: bool) -> TestClient:
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector,
        "async_push_usage_payloads",
        AsyncMock(),
    )
    monkeypatch.setattr(
        http_api,
        "WORKFLOWS_DISABLE_SINKS",
        disable_sinks_by_environment,
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


@pytest.mark.parametrize(
    "request_disables_sinks,environment_disables_sinks",
    [(True, False), (False, True)],
)
@mock.patch.dict(webhook_v1.METHOD_TO_HANDLER, {"POST": MagicMock()}, clear=True)
def test_workflow_run_injects_sink_disabling_policy(
    monkeypatch,
    request_disables_sinks: bool,
    environment_disables_sinks: bool,
) -> None:
    client = _build_test_client(
        monkeypatch,
        disable_sinks_by_environment=environment_disables_sinks,
    )

    response = client.post(
        "/workflows/run",
        json={
            "specification": WEBHOOK_WORKFLOW,
            "inputs": {},
            "disable_sinks": request_disables_sinks,
        },
    )

    assert response.status_code == 200
    assert response.json()["outputs"] == [
        {"message": "Sink was disabled by parameter `disable_sink`"}
    ]
    webhook_v1.METHOD_TO_HANDLER["POST"].assert_not_called()


@mock.patch.dict(webhook_v1.METHOD_TO_HANDLER, {"POST": MagicMock()}, clear=True)
def test_workflow_run_keeps_sinks_enabled_by_default(monkeypatch) -> None:
    response_from_webhook = Response()
    response_from_webhook.status_code = 200
    webhook_v1.METHOD_TO_HANDLER["POST"].return_value = response_from_webhook
    client = _build_test_client(monkeypatch, disable_sinks_by_environment=False)

    response = client.post(
        "/workflows/run",
        json={
            "specification": WEBHOOK_WORKFLOW,
            "inputs": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["outputs"] == [{"message": "Notification sent successfully"}]
    webhook_v1.METHOD_TO_HANDLER["POST"].assert_called_once()
