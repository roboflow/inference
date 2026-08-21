from typing import AsyncIterator

from fastapi import FastAPI
from starlette.testclient import TestClient

from inference.core.interfaces.http.handlers.http_worker import (
    register_http_worker_routes,
)


def test_create_requires_api_key() -> None:
    client = TestClient(FastAPI())
    register_http_worker_routes(client.app)
    response = client.post(
        "/sam3/video/sessions",
        json={
            "video_url": "https://storage.example/clip.mp4",
            "class_names": ["forklift"],
            "artifact": {
                "app_base_url": "https://app.roboflow.com",
                "video_id": "video-1",
                "workspace_id": "ws-1",
                "dataset_id": "ds-1",
                "revision_id": "rev-1",
            },
        },
    )
    assert response.status_code == 401


def test_create_starts_worker(monkeypatch) -> None:
    started = {}

    def fake_start(request, *, api_key, events_callback_base):
        started["api_key"] = api_key
        started["callback"] = events_callback_base
        started["video_url"] = request.video_url
        return "sess-1"

    monkeypatch.setattr(
        "inference.core.interfaces.http.handlers.http_worker.start_worker",
        fake_start,
    )
    app = FastAPI()
    register_http_worker_routes(app)
    client = TestClient(app)
    response = client.post(
        "/sam3/video/sessions",
        json={
            "video_url": "https://storage.example/clip.mp4",
            "class_names": ["forklift"],
            "api_key": "rf_key",
            "artifact": {
                "app_base_url": "https://app.roboflow.com",
                "video_id": "video-1",
                "workspace_id": "ws-1",
                "dataset_id": "ds-1",
                "revision_id": "rev-1",
            },
        },
    )
    assert response.status_code == 200
    assert response.json() == {"session_id": "sess-1"}
    assert started["api_key"] == "rf_key"
    assert started["video_url"] == "https://storage.example/clip.mp4"


def test_create_rejects_foreign_events_callback_base(monkeypatch) -> None:
    monkeypatch.setattr(
        "inference.core.interfaces.http.handlers.http_worker.start_worker",
        lambda *args, **kwargs: "sess-1",
    )
    app = FastAPI()
    register_http_worker_routes(app)
    client = TestClient(app)
    response = client.post(
        "/sam3/video/sessions",
        json={
            "video_url": "https://storage.example/clip.mp4",
            "class_names": ["forklift"],
            "api_key": "rf_key",
            "events_callback_base": "https://evil.example",
            "artifact": {
                "app_base_url": "https://app.roboflow.com",
                "video_id": "video-1",
                "workspace_id": "ws-1",
                "dataset_id": "ds-1",
                "revision_id": "rev-1",
            },
        },
    )
    assert response.status_code == 400


def test_snapshot_404_unknown_session(monkeypatch) -> None:
    def fake_snapshot(session_id, *, api_key):
        raise KeyError(session_id)

    monkeypatch.setattr(
        "inference.core.interfaces.http.handlers.http_worker.worker_snapshot",
        fake_snapshot,
    )
    app = FastAPI()
    register_http_worker_routes(app)
    client = TestClient(app)
    response = client.get("/sam3/video/sessions/missing", params={"api_key": "rf_key"})
    assert response.status_code == 404


def test_snapshot_401_wrong_owner(monkeypatch) -> None:
    def fake_snapshot(session_id, *, api_key):
        raise PermissionError("nope")

    monkeypatch.setattr(
        "inference.core.interfaces.http.handlers.http_worker.worker_snapshot",
        fake_snapshot,
    )
    app = FastAPI()
    register_http_worker_routes(app)
    client = TestClient(app)
    response = client.get("/sam3/video/sessions/sess-1", params={"api_key": "other"})
    assert response.status_code == 401


def test_internal_events_401_bad_token(monkeypatch) -> None:
    def fake_publish(session_id, *, publish_token, event):
        raise PermissionError("bad token")

    monkeypatch.setattr(
        "inference.core.interfaces.http.handlers.http_worker.publish_internal_event",
        fake_publish,
    )
    app = FastAPI()
    register_http_worker_routes(app)
    client = TestClient(app)
    response = client.post(
        "/sam3/video/sessions/sess-1/internal/events",
        json={"publish_token": "wrong", "event": {"type": "frame"}},
        params={"api_key": "rf_key"},
    )
    assert response.status_code == 401


def test_sse_stops_on_done(monkeypatch) -> None:
    async def fake_iter(session_id, *, api_key, after_seq) -> AsyncIterator[str]:
        yield 'event: frame\ndata: {"seq": 1, "type": "frame"}\n\n'
        yield 'event: done\ndata: {"seq": 2, "type": "done"}\n\n'

    monkeypatch.setattr(
        "inference.core.interfaces.http.handlers.http_worker.worker_snapshot",
        lambda session_id, *, api_key: {
            "session_id": session_id,
            "status": "running",
            "last_seq": 0,
        },
    )
    monkeypatch.setattr(
        "inference.core.interfaces.http.handlers.http_worker.iter_public_events",
        fake_iter,
    )
    app = FastAPI()
    register_http_worker_routes(app)
    client = TestClient(app)
    with client.stream(
        "GET",
        "/sam3/video/sessions/sess-1/events",
        params={"api_key": "rf_key"},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "event: done" in body
