"""Tests for region enforcement in the WebRTC worker spawn funnel.

The Modal spawn is mocked so no Modal app deploy or network call happens.
"""

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

import inference.core.interfaces.webrtc_worker as webrtc_worker
from inference.core.exceptions import WebRTCConfigurationError
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCOffer,
    WorkflowConfiguration,
)
from inference.core.interfaces.webrtc_worker import start_worker
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCWorkerResult,
    WebRTCWorkerRequest,
)


def _build_request(
    requested_region=None,
    workspace_name="workspace-1",
    workspace_id=None,
) -> WebRTCWorkerRequest:
    return WebRTCWorkerRequest(
        api_key="fake-api-key",
        workflow_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration",
            workspace_name=workspace_name,
        ),
        webrtc_offer=WebRTCOffer(type="offer", sdp="fake-sdp"),
        requested_region=requested_region,
        workspace_id=workspace_id,
    )


def _install_modal_stubs(monkeypatch, spawn_mock):
    """Stub the modules lazily imported inside the Modal branch of start_worker."""
    modal_stub = types.ModuleType("inference.core.interfaces.webrtc_worker.modal")
    modal_stub.spawn_rtc_peer_connection_modal = spawn_mock
    monkeypatch.setitem(
        sys.modules,
        "inference.core.interfaces.webrtc_worker.modal",
        modal_stub,
    )

    utils_stub = types.ModuleType("inference.core.interfaces.webrtc_worker.utils")
    utils_stub.get_total_concurrent_sessions = MagicMock(return_value=0)
    utils_stub.is_over_quota = MagicMock(return_value=False)
    utils_stub.is_over_workspace_session_quota = MagicMock(return_value=False)
    utils_stub.register_webrtc_session = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "inference.core.interfaces.webrtc_worker.utils",
        utils_stub,
    )
    return utils_stub


def _enter_modal_branch(monkeypatch):
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_TOKEN_ID", "token-id")
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_TOKEN_SECRET", "token-secret")
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_USAGE_QUOTA_ENABLED", False)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_WORKSPACE_STREAM_QUOTA_ENABLED", False)
    monkeypatch.setattr(
        "inference.core.interfaces.webrtc_worker.request_utils.get_roboflow_workspace",
        lambda api_key: "workspace-from-api-key",
    )


def test_start_worker_preserves_client_region_when_enforce_disabled(
    monkeypatch,
) -> None:
    # given
    captured = {}

    def spawn(webrtc_request: WebRTCWorkerRequest) -> WebRTCWorkerResult:
        captured["requested_region"] = webrtc_request.requested_region
        return WebRTCWorkerResult()

    _enter_modal_branch(monkeypatch)
    _install_modal_stubs(monkeypatch, spawn)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_ENFORCE_REGION", False)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_REQUIRED_REGION", "eu")

    request = _build_request(requested_region="us-east")

    # when
    result = asyncio.run(start_worker(request))

    # then
    assert isinstance(result, WebRTCWorkerResult)
    assert captured["requested_region"] == "us-east"


def test_start_worker_overrides_region_when_enforce_enabled(monkeypatch) -> None:
    # given
    captured = {}

    def spawn(webrtc_request: WebRTCWorkerRequest) -> WebRTCWorkerResult:
        captured["requested_region"] = webrtc_request.requested_region
        return WebRTCWorkerResult()

    _enter_modal_branch(monkeypatch)
    _install_modal_stubs(monkeypatch, spawn)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_ENFORCE_REGION", True)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_REQUIRED_REGION", "eu")

    request = _build_request(requested_region="us-east")

    # when
    result = asyncio.run(start_worker(request))

    # then
    assert isinstance(result, WebRTCWorkerResult)
    assert captured["requested_region"] == "eu"


def test_start_worker_raises_when_enforced_region_is_not_eu(monkeypatch) -> None:
    # given
    spawn = MagicMock()
    _enter_modal_branch(monkeypatch)
    _install_modal_stubs(monkeypatch, spawn)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_ENFORCE_REGION", True)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_REQUIRED_REGION", "us-east")

    request = _build_request()

    # when / then
    with pytest.raises(WebRTCConfigurationError):
        asyncio.run(start_worker(request))
    spawn.assert_not_called()


def test_start_worker_uses_api_workspace_for_workspace_quota(monkeypatch) -> None:
    # given
    captured = {}

    def spawn(webrtc_request: WebRTCWorkerRequest) -> WebRTCWorkerResult:
        captured["workspace_id"] = webrtc_request.workspace_id
        captured["workspace_name"] = (
            webrtc_request.workflow_configuration.workspace_name
        )
        return WebRTCWorkerResult()

    _enter_modal_branch(monkeypatch)
    utils_stub = _install_modal_stubs(monkeypatch, spawn)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_WORKSPACE_STREAM_QUOTA_ENABLED", True)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_WORKSPACE_STREAM_QUOTA", 1)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_WORKSPACE_STREAM_TTL_SECONDS", 60)

    request = _build_request(
        workspace_name="workspace-from-request",
        workspace_id="workspace-id-from-request",
    )

    # when
    result = asyncio.run(start_worker(request))

    # then
    assert isinstance(result, WebRTCWorkerResult)
    assert captured["workspace_id"] == "workspace-from-api-key"
    assert captured["workspace_name"] == "workspace-from-api-key"
    utils_stub.is_over_workspace_session_quota.assert_called_once_with(
        workspace_id="workspace-from-api-key",
        quota=1,
        ttl_seconds=60,
    )
    utils_stub.register_webrtc_session.assert_called_once()
    assert (
        utils_stub.register_webrtc_session.call_args.kwargs["workspace_id"]
        == "workspace-from-api-key"
    )


def test_start_worker_raises_when_enforced_region_is_none(monkeypatch) -> None:
    # given
    spawn = MagicMock()
    _enter_modal_branch(monkeypatch)
    _install_modal_stubs(monkeypatch, spawn)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_ENFORCE_REGION", True)
    monkeypatch.setattr(webrtc_worker, "WEBRTC_MODAL_REQUIRED_REGION", None)

    request = _build_request()

    # when / then
    with pytest.raises(WebRTCConfigurationError):
        asyncio.run(start_worker(request))
    spawn.assert_not_called()
