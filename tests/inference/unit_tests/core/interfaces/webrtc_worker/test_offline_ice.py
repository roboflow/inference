from inference.core.interfaces.stream_manager.manager_app import (
    webrtc as stream_manager_webrtc,
)
from inference.core.interfaces.webrtc_worker import webrtc as worker_webrtc


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
