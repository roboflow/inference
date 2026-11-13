import logging
from unittest.mock import MagicMock

import pytest
import requests

from inference_server import configuration, platform_http
from inference_server.workflows import host


def _capture(monkeypatch, method):
    seen = {}

    def _call(*args, **kwargs):
        seen["kwargs"] = kwargs
        response = MagicMock(status_code=200)
        response.json.return_value = {"workspace": "ws-1"}
        return response

    monkeypatch.setattr(requests, method, _call)
    return seen


@pytest.fixture
def app_log():
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Handler(level=logging.WARNING)
    app_logger = logging.getLogger("inference_server.app")
    app_logger.addHandler(handler)
    yield records
    app_logger.removeHandler(handler)


@pytest.fixture
def verification_off(monkeypatch):
    monkeypatch.setattr(configuration, "ROBOFLOW_API_VERIFY_SSL", False)


def test_platform_request_passes_no_verify_argument_by_default(monkeypatch):
    seen = _capture(monkeypatch, "get")

    platform_http._platform_request("get", "https://x", headers={"h": "v"}, timeout=1)

    assert "verify" not in seen["kwargs"]


def test_platform_request_disables_verification_when_switch_is_off(
    monkeypatch, verification_off
):
    seen = _capture(monkeypatch, "get")

    platform_http._platform_request("get", "https://x", headers={"h": "v"}, timeout=1)

    assert seen["kwargs"]["verify"] is False


def test_platform_request_keeps_a_caller_supplied_verify(monkeypatch, verification_off):
    seen = _capture(monkeypatch, "post")

    platform_http._platform_request("post", "https://x", verify="/ca.pem")

    assert seen["kwargs"]["verify"] == "/ca.pem"


def test_post_to_api_passes_no_verify_argument_by_default(monkeypatch):
    seen = _capture(monkeypatch, "post")

    host.PLATFORM_CLIENT._post_to_api("https://x/y", json={})

    assert "verify" not in seen["kwargs"]


def test_post_to_api_disables_verification_when_switch_is_off(
    monkeypatch, verification_off
):
    seen = _capture(monkeypatch, "post")

    host.PLATFORM_CLIENT._post_to_api("https://x/y", json={})

    assert seen["kwargs"]["verify"] is False


def test_workspace_fetch_passes_no_verify_argument_by_default(monkeypatch):
    seen = _capture(monkeypatch, "get")

    host.PLATFORM_CLIENT._fetch_roboflow_workspace(api_key="k")

    assert "verify" not in seen["kwargs"]


def test_workspace_fetch_disables_verification_when_switch_is_off(
    monkeypatch, verification_off
):
    seen = _capture(monkeypatch, "get")

    host.PLATFORM_CLIENT._fetch_roboflow_workspace(api_key="k")

    assert seen["kwargs"]["verify"] is False


def test_workflow_definition_fetch_disables_verification_when_switch_is_off(
    monkeypatch, verification_off
):
    seen = _capture(monkeypatch, "get")

    host._fetch_workflow_response(
        api_key="k", workspace_id="ws", workflow_id="wf", workflow_version_id=None
    )

    assert seen["kwargs"]["verify"] is False


@pytest.mark.asyncio
async def test_lifespan_warns_once_when_verification_is_off(
    monkeypatch, verification_off, app_log
):
    import inference_server.app as app_mod

    _stub_lifespan(monkeypatch)

    async with app_mod._lifespan(app_mod.app):
        pass

    assert [m for m in app_log if "TLS certificate" in m] == [
        "TLS certificate verification is disabled for Roboflow platform requests"
    ]


@pytest.mark.asyncio
async def test_lifespan_does_not_warn_when_verification_is_on(monkeypatch, app_log):
    import inference_server.app as app_mod

    _stub_lifespan(monkeypatch)

    async with app_mod._lifespan(app_mod.app):
        pass

    assert not [m for m in app_log if "TLS certificate" in m]


def _stub_lifespan(monkeypatch):
    class _StubProxy:
        async def start(self):
            pass

        async def shutdown(self):
            pass

    monkeypatch.delenv("INFERENCE_PRELOAD_MODELS", raising=False)
    monkeypatch.setattr(
        "inference_model_manager.watchdogs.start_enabled_watchdogs", lambda: []
    )
    monkeypatch.setattr(
        "inference_server.gateway_resolver.resolve_gateway", lambda: _StubProxy()
    )
