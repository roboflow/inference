import hashlib
import uuid

import pytest

from inference.core.interfaces.sam3_video_session.entities import (
    Sam3VideoArtifactTarget,
    Sam3VideoSessionRequest,
)
from inference.core.interfaces.sam3_video_session.service import (
    publish_internal_event,
    require_owner,
    resolve_events_callback_base,
    session_snapshot,
    start_session,
)
from inference.core.interfaces.sam3_video_session.session_store import (
    create_session,
    list_events,
)


def test_publish_internal_event_appends_and_replays_after_seq() -> None:
    session_id = f"sam3-video-session-test-{uuid.uuid4()}"
    create_session(
        session_id,
        workspace_id="ws-1",
        publish_token="publish-me",
        owner_api_key_hash="abc",
    )

    publish_internal_event(
        session_id,
        publish_token="publish-me",
        event={"type": "downloading"},
    )
    publish_internal_event(
        session_id,
        publish_token="publish-me",
        event={"type": "frame", "frame_id": 4},
    )

    replayed = list_events(session_id, after_seq=1)
    assert [event["type"] for event in replayed] == ["frame"]
    assert replayed[0]["frame_id"] == 4


def test_publish_internal_event_rejects_bad_token() -> None:
    session_id = f"sam3-video-session-test-{uuid.uuid4()}"
    create_session(
        session_id,
        workspace_id="ws-1",
        publish_token="publish-me",
        owner_api_key_hash="abc",
    )
    with pytest.raises(PermissionError):
        publish_internal_event(
            session_id,
            publish_token="wrong-token-value",
            event={"type": "frame"},
        )


def test_require_owner_rejects_other_api_key() -> None:
    session_id = f"sam3-video-session-test-{uuid.uuid4()}"
    create_session(
        session_id,
        workspace_id="ws-1",
        publish_token="token",
        owner_api_key_hash=hashlib.sha256(b"owner-key").hexdigest(),
    )
    with pytest.raises(PermissionError):
        require_owner(session_id, "other-key")
    with pytest.raises(PermissionError):
        session_snapshot(session_id, api_key="other-key")


def test_resolve_events_callback_base_rejects_foreign_host() -> None:
    with pytest.raises(ValueError, match="must match this inference server"):
        resolve_events_callback_base(
            "https://evil.example/callback",
            "https://serverless.roboflow.com",
        )


def test_resolve_events_callback_base_allows_same_host() -> None:
    resolved = resolve_events_callback_base(
        "https://serverless.roboflow.com",
        "https://serverless.roboflow.com/",
    )
    assert resolved == "https://serverless.roboflow.com/"


def test_start_session_spawns_local_without_modal(monkeypatch) -> None:
    spawned = {}
    monkeypatch.setattr(
        "inference.core.interfaces.sam3_video_session.service.WEBRTC_MODAL_TOKEN_ID",
        "",
    )
    monkeypatch.setattr(
        "inference.core.interfaces.sam3_video_session.service.WEBRTC_MODAL_TOKEN_SECRET",
        "",
    )
    monkeypatch.setattr(
        "inference.core.interfaces.sam3_video_session.service.WEBRTC_MODAL_USAGE_QUOTA_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "inference.core.interfaces.sam3_video_session.service.WEBRTC_WORKSPACE_STREAM_QUOTA_ENABLED",
        False,
    )
    monkeypatch.setattr(
        "inference.core.interfaces.sam3_video_session.service.get_roboflow_workspace",
        lambda api_key: "ws-1",
    )

    def fake_spawn(payload) -> None:
        spawned["session_id"] = payload.session_id
        spawned["video_url"] = payload.video_url
        spawned["revision_id"] = payload.artifact.revision_id

    monkeypatch.setattr(
        "inference.core.interfaces.sam3_video_session.service._spawn_local",
        fake_spawn,
    )

    session_id = start_session(
        Sam3VideoSessionRequest(
            video_url="https://storage.example/clip.mp4",
            class_names=["forklift"],
            artifact=Sam3VideoArtifactTarget(
                app_base_url="https://app.roboflow.com",
                video_id="video-1",
                workspace_id="ws-1",
                dataset_id="ds-1",
                revision_id="rev-1",
            ),
        ),
        api_key="rf_key",
        events_callback_base="https://serverless.example",
    )

    assert spawned["session_id"] == session_id
    assert spawned["video_url"] == "https://storage.example/clip.mp4"
    assert spawned["revision_id"] == "rev-1"
    snap = session_snapshot(session_id, api_key="rf_key")
    assert snap is not None
    assert snap["session_id"] == session_id
