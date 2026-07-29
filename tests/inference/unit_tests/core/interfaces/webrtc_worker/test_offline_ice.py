import pytest

from inference.core.exceptions import WebRTCConfigurationError
from inference.core.interfaces.stream_manager.manager_app import (
    webrtc as stream_manager_webrtc,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCTURNConfig,
)
from inference.core.interfaces.webrtc_worker import webrtc as worker_webrtc
from inference.core.interfaces.webrtc_worker.entities import RTCIceServer, WebRTCConfig


def test_worker_explicitly_disables_aiortc_default_stun(
    monkeypatch,
) -> None:
    monkeypatch.setattr(worker_webrtc, "OFFLINE_MODE", True)
    monkeypatch.setattr(worker_webrtc, "WEBRTC_MODAL_PUBLIC_STUN_SERVERS", "")

    configuration = worker_webrtc._build_rtc_configuration(webrtc_config=None)

    assert configuration is not None
    assert configuration.iceServers == []


def test_worker_retains_online_aiortc_default(monkeypatch) -> None:
    monkeypatch.setattr(worker_webrtc, "OFFLINE_MODE", False)

    configuration = worker_webrtc._build_rtc_configuration(webrtc_config=None)

    assert configuration is None


def test_worker_rejects_explicit_ice_servers_offline(monkeypatch) -> None:
    monkeypatch.setattr(worker_webrtc, "OFFLINE_MODE", True)
    configuration = WebRTCConfig(
        iceServers=[RTCIceServer(urls="stun:attacker.example:3478")]
    )

    with pytest.raises(WebRTCConfigurationError, match="OFFLINE_MODE"):
        worker_webrtc._build_rtc_configuration(webrtc_config=configuration)


def test_worker_allows_explicit_empty_ice_configuration_offline(monkeypatch) -> None:
    monkeypatch.setattr(worker_webrtc, "OFFLINE_MODE", True)

    configuration = worker_webrtc._build_rtc_configuration(
        webrtc_config=WebRTCConfig(iceServers=[])
    )

    assert configuration is not None
    assert configuration.iceServers == []


def test_stream_manager_disables_aiortc_default_stun_offline(
    monkeypatch,
) -> None:
    monkeypatch.setattr(stream_manager_webrtc, "OFFLINE_MODE", True)

    configuration = stream_manager_webrtc._build_rtc_configuration(
        webrtc_turn_config=None
    )

    assert configuration is not None
    assert configuration.iceServers == []


def test_stream_manager_retains_online_aiortc_default(
    monkeypatch,
) -> None:
    monkeypatch.setattr(stream_manager_webrtc, "OFFLINE_MODE", False)

    configuration = stream_manager_webrtc._build_rtc_configuration(
        webrtc_turn_config=None
    )

    assert configuration is None


def test_stream_manager_rejects_explicit_turn_server_offline(
    monkeypatch,
) -> None:
    monkeypatch.setattr(stream_manager_webrtc, "OFFLINE_MODE", True)
    turn_config = WebRTCTURNConfig(
        urls="turn:attacker.example:3478",
        username="user",
        credential="secret",
    )

    with pytest.raises(WebRTCConfigurationError, match="OFFLINE_MODE"):
        stream_manager_webrtc._build_rtc_configuration(webrtc_turn_config=turn_config)
