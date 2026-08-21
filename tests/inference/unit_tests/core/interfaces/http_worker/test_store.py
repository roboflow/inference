import uuid

from inference.core.interfaces.http_worker.store import (
    append_event,
    create_session,
    is_stop_requested,
    list_events,
    request_stop,
    snapshot,
)


def test_list_events_replays_after_seq() -> None:
    session_id = f"http-worker-test-{uuid.uuid4()}"
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


def test_stop_requested_flag() -> None:
    session_id = f"http-worker-test-{uuid.uuid4()}"
    create_session(
        session_id,
        workspace_id="ws-1",
        publish_token="token",
        owner_api_key_hash="abc",
    )

    assert is_stop_requested(session_id, client_ttl_seconds=60) is False
    request_stop(session_id)
    assert is_stop_requested(session_id, client_ttl_seconds=60) is True
