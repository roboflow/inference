import uuid

from inference.core.interfaces.sam3_video_session.session_store import (
    append_event,
    create_session,
    is_stop_requested,
    list_events,
    request_stop,
    snapshot,
)


def test_list_events_replays_after_seq() -> None:
    session_id = f"sam3-video-session-test-{uuid.uuid4()}"
    create_session(
        session_id,
        workspace_id="ws-1",
        publish_token="token",
        owner_api_key_hash="abc",
    )
    append_event(session_id, "downloading")
    append_event(session_id, "frame", {"frame_id": 0})
    append_event(session_id, "frame", {"frame_id": 1})
    append_event(session_id, "done", {"frame_count": 2})

    replayed = list_events(session_id, after_seq=1)

    assert [event["type"] for event in replayed] == ["frame", "frame", "done"]
    assert replayed[0]["seq"] == 2
    assert replayed[0]["frame_id"] == 0
    snap = snapshot(session_id)
    assert snap is not None
    assert snap["status"] == "completed"
    assert snap["last_seq"] == 4
    assert snap["last_frame_id"] == 1


def test_append_event_caps_retained_events() -> None:
    session_id = f"sam3-video-session-test-{uuid.uuid4()}"
    create_session(
        session_id,
        workspace_id="ws-1",
        publish_token="token",
        owner_api_key_hash="abc",
    )
    for index in range(40):
        append_event(session_id, "frame", {"frame_id": index})

    replayed = list_events(session_id, after_seq=0)

    assert len(replayed) <= 32
    assert replayed[0]["frame_id"] == 8
    assert replayed[-1]["frame_id"] == 39


def test_list_events_pages_backlog() -> None:
    session_id = f"sam3-video-session-test-{uuid.uuid4()}"
    create_session(
        session_id,
        workspace_id="ws-1",
        publish_token="token",
        owner_api_key_hash="abc",
    )
    for index in range(10):
        append_event(session_id, "frame", {"frame_id": index})

    first_page = list_events(session_id, after_seq=0, limit=3)
    assert [event["frame_id"] for event in first_page] == [0, 1, 2]
    next_page = list_events(session_id, after_seq=first_page[-1]["seq"], limit=3)
    assert [event["frame_id"] for event in next_page] == [3, 4, 5]


def test_stop_requested_flag() -> None:
    session_id = f"sam3-video-session-test-{uuid.uuid4()}"
    create_session(
        session_id,
        workspace_id="ws-1",
        publish_token="token",
        owner_api_key_hash="abc",
    )

    assert is_stop_requested(session_id, client_ttl_seconds=60) is False
    request_stop(session_id)
    assert is_stop_requested(session_id, client_ttl_seconds=60) is True
