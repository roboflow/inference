"""Proxied third-party calls carry the workflow and device they run for.

The platform bills `apiproxy/*` calls to the presented key; when that key is
workspace-wide these fields let it bill the workflow's (or device's) folder.
"""

from unittest.mock import MagicMock

from inference_server.middlewares.model_load import REQUEST_WORKFLOW_ID
from inference_server.workflows import host
from inference_server.workflows.host import (
    PLATFORM_CLIENT,
    billing_attribution_fields,
)


def test_fields_empty_without_workflow_or_device(monkeypatch) -> None:
    monkeypatch.setattr(host.configuration, "DEVICE_ID", None)
    token = REQUEST_WORKFLOW_ID.set(None)
    try:
        assert billing_attribution_fields() == {}
    finally:
        REQUEST_WORKFLOW_ID.reset(token)


def test_fields_carry_request_workflow_and_device(monkeypatch) -> None:
    monkeypatch.setattr(host.configuration, "DEVICE_ID", "device-1")
    token = REQUEST_WORKFLOW_ID.set("my-workflow")
    try:
        assert billing_attribution_fields() == {
            "workflow_id": "my-workflow",
            "device_id": "device-1",
        }
    finally:
        REQUEST_WORKFLOW_ID.reset(token)


def test_post_adds_fields_to_proxy_calls_only(monkeypatch) -> None:
    response = MagicMock(status_code=200)
    response.json.return_value = {}
    posted = MagicMock(return_value=response)
    monkeypatch.setattr(host, "_platform_request", posted)
    monkeypatch.setattr(host, "_refuse_when_offline", lambda operation: None)
    monkeypatch.setattr(host.configuration, "DEVICE_ID", "device-1")
    token = REQUEST_WORKFLOW_ID.set("my-workflow")
    try:
        PLATFORM_CLIENT.post(
            endpoint="apiproxy/email", api_key="k", payload={"subject": "s"}
        )
        PLATFORM_CLIENT.post(
            endpoint="apiproxy/twilio",
            api_key="k",
            payload={"workflow_id": "explicit", "message": "m"},
        )
        PLATFORM_CLIENT.post(endpoint="other/route", api_key="k", payload={"a": 1})
    finally:
        REQUEST_WORKFLOW_ID.reset(token)

    email, twilio, other = [call.kwargs["json"] for call in posted.call_args_list]
    assert email == {
        "subject": "s",
        "workflow_id": "my-workflow",
        "device_id": "device-1",
    }
    assert twilio["workflow_id"] == "explicit"
    assert twilio["device_id"] == "device-1"
    assert other == {"a": 1}
