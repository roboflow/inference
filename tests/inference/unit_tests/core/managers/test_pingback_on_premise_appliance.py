import logging
from unittest.mock import MagicMock

import pytest

from inference.core.managers import pingback as pingback_module
from inference.core.managers.pingback import PingbackInfo


@pytest.fixture
def posted(monkeypatch):
    """Capture the payloads the pingback posts."""
    payloads = []

    def fake_post(url, json, timeout, verify):
        payloads.append(json)
        response = MagicMock()
        response.status_code = 200
        return response

    monkeypatch.setattr(pingback_module.requests, "post", fake_post)
    monkeypatch.setattr(pingback_module, "OFFLINE_MODE", False)
    monkeypatch.setattr(pingback_module, "get_system_info", dict)
    monkeypatch.setattr(
        pingback_module, "api_key_safe_raise_for_status", lambda response: None
    )

    return payloads


def _pingback() -> PingbackInfo:
    manager = MagicMock()
    manager.uuid = "server-uuid"
    manager.models.return_value = []

    return PingbackInfo(manager)


def test_appliance_pingback_sends_the_metrics_key_and_never_borrows(
    monkeypatch, posted
) -> None:
    monkeypatch.setattr(pingback_module, "ON_PREMISE_APPLIANCE", True)
    monkeypatch.setattr(pingback_module, "METRICS_API_KEY", "pingback-key")
    monkeypatch.setattr(pingback_module, "API_KEY", None)
    pingback = _pingback()
    pingback.fallback_api_key = "tenant-request-key"

    pingback.post_data(pingback.model_manager)

    assert [payload["api_key"] for payload in posted] == ["pingback-key"]


def test_appliance_pingback_without_a_key_sends_nothing_and_warns_once(
    monkeypatch, posted, caplog
) -> None:
    monkeypatch.setattr(pingback_module, "ON_PREMISE_APPLIANCE", True)
    monkeypatch.setattr(pingback_module, "METRICS_API_KEY", None)
    monkeypatch.setattr(pingback_module, "API_KEY", None)
    caplog.set_level(logging.WARNING)
    pingback = _pingback()
    pingback.fallback_api_key = "tenant-request-key"

    pingback.post_data(pingback.model_manager)
    pingback.post_data(pingback.model_manager)

    assert posted == []
    warnings = [r for r in caplog.records if "pingback is not sent" in r.getMessage()]
    assert len(warnings) == 1


def test_pingback_outside_appliance_mode_still_borrows_the_request_key(
    monkeypatch, posted
) -> None:
    monkeypatch.setattr(pingback_module, "ON_PREMISE_APPLIANCE", False)
    monkeypatch.setattr(pingback_module, "METRICS_API_KEY", "pingback-key")
    monkeypatch.setattr(pingback_module, "API_KEY", None)
    pingback = _pingback()
    pingback.fallback_api_key = "tenant-request-key"

    pingback.post_data(pingback.model_manager)

    assert [payload["api_key"] for payload in posted] == ["tenant-request-key"]


@pytest.mark.parametrize("appliance", [True, False])
def test_pingback_logs_never_contain_the_api_key(
    monkeypatch, posted, caplog, appliance
) -> None:
    monkeypatch.setattr(pingback_module, "ON_PREMISE_APPLIANCE", appliance)
    monkeypatch.setattr(pingback_module, "METRICS_API_KEY", "pingback-key")
    monkeypatch.setattr(pingback_module, "API_KEY", "pingback-key")
    caplog.set_level(logging.DEBUG, logger=pingback_module.logger.name)
    pingback = _pingback()

    pingback.post_data(pingback.model_manager)
    monkeypatch.setattr(
        pingback_module.requests,
        "post",
        MagicMock(side_effect=ConnectionError("metrics endpoint down")),
    )
    pingback.post_data(pingback.model_manager)

    assert "Sent metrics" in caplog.text
    assert "Error was:" in caplog.text
    assert "pingback-key" not in caplog.text
