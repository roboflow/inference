from unittest import mock

from inference.core.interfaces.webrtc_worker import watchdog
from inference.core.interfaces.webrtc_worker.watchdog import Watchdog


@mock.patch.object(watchdog.requests, "post")
@mock.patch.object(watchdog, "OFFLINE_MODE", True)
def test_watchdog_does_not_send_session_heartbeats_offline(post_mock) -> None:
    instance = Watchdog(
        api_key="api-key",
        timeout_seconds=30,
        workspace_id="workspace",
        session_id="session",
        heartbeat_url="https://example.com/webrtc/session/heartbeat",
    )

    instance._send_session_heartbeat()
    instance._send_session_heartbeat_stop()

    post_mock.assert_not_called()
