"""Proxied third-party calls carry the workflow and device they run for.

The platform bills `apiproxy/*` calls to the presented key; when that key is
workspace-wide these fields let it bill the workflow's (or device's) folder.
"""

from unittest.mock import MagicMock

import inference.core.interfaces.roboflow_platform_client as client_module
from inference.core.interfaces.roboflow_platform_client import (
    SERVER_PLATFORM_CLIENT,
    billing_attribution_fields,
)
from inference.core.managers.model_load_collector import request_workflow_id


def test_fields_empty_without_workflow_or_device(monkeypatch) -> None:
    monkeypatch.setattr(client_module.env, "DEVICE_ID", None)
    token = request_workflow_id.set(None)
    try:
        assert billing_attribution_fields() == {}
    finally:
        request_workflow_id.reset(token)


def test_fields_carry_request_workflow_and_device(monkeypatch) -> None:
    monkeypatch.setattr(client_module.env, "DEVICE_ID", "device-1")
    token = request_workflow_id.set("my-workflow")
    try:
        assert billing_attribution_fields() == {
            "workflow_id": "my-workflow",
            "device_id": "device-1",
        }
    finally:
        request_workflow_id.reset(token)


def test_post_adds_fields_to_proxy_calls_only(monkeypatch) -> None:
    posted = MagicMock(return_value={})
    monkeypatch.setattr(client_module.roboflow_api, "post_to_roboflow_api", posted)
    monkeypatch.setattr(client_module.env, "DEVICE_ID", "device-1")
    token = request_workflow_id.set("my-workflow")
    try:
        SERVER_PLATFORM_CLIENT.post(
            endpoint="apiproxy/email", api_key="k", payload={"subject": "s"}
        )
        SERVER_PLATFORM_CLIENT.post(
            endpoint="apiproxy/twilio",
            api_key="k",
            payload={"workflow_id": "explicit", "message": "m"},
        )
        SERVER_PLATFORM_CLIENT.post(
            endpoint="other/route", api_key="k", payload={"a": 1}
        )
    finally:
        request_workflow_id.reset(token)

    email, twilio, other = [call.kwargs["payload"] for call in posted.call_args_list]
    assert email == {
        "subject": "s",
        "workflow_id": "my-workflow",
        "device_id": "device-1",
    }
    assert twilio["workflow_id"] == "explicit"
    assert twilio["device_id"] == "device-1"
    assert other == {"a": 1}
