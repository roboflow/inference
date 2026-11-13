"""Exercise real ingestion and decoding through both offline-video HTTP APIs."""

import base64
import tempfile

import pytest
from starlette.testclient import TestClient

from inference.core.utils import video_utils
from inference_models.models.base.action_recognition import VideoSampling
from tests.inference.unit_tests.core.interfaces.http import (
    test_sam3_video_routes as sam3_routes,
)
from tests.inference.unit_tests.core.interfaces.http.test_legacy_action_recognition_route import (
    _build_interface,
)
from tests.inference.unit_tests.core.models.test_action_recognition_adapter import (
    _adapter,
    _FakeModel,
)

loads = sam3_routes.loads


@pytest.mark.parametrize("route", ["sam3", "action", "generic"])
@pytest.mark.parametrize("transport", ["base64", "url"])
@pytest.mark.parametrize("manifest", [False, True])
def test_offline_routes_only_decode_the_submitted_asset(
    tmp_path, monkeypatch, loads, route, transport, manifest
):
    import inference.core.interfaces.http.http_api as http_api

    clip = sam3_routes.write_clip(tmp_path / "server-only.mp4")
    payload = clip.read_bytes()
    if manifest:
        payload = (
            "#EXTM3U\n#EXT-X-TARGETDURATION:1\n"
            f"#EXTINF:1.0,\n{clip}\n#EXT-X-ENDLIST\n"
        ).encode()
    scratch = tmp_path / "uploads"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(http_api, "SAM3_VIDEO_ENABLED", True)
    monkeypatch.setattr(http_api, "SAM3_EXEC_MODE", "local")
    monkeypatch.setattr(http_api, "CORE_MODEL_SAM3_ENABLED", True)
    monkeypatch.setattr(http_api, "ACTION_RECOGNITION_ENABLED", True)
    monkeypatch.setattr(video_utils, "OFFLINE_MODE", False)
    monkeypatch.setattr(video_utils, "_ensure_url_input_allowed", lambda: None)
    monkeypatch.setattr(video_utils, "_validate_url_destination", lambda value: value)
    monkeypatch.setattr(
        video_utils,
        "_fetch_image_bytes_from_url",
        lambda *, sink, **kwargs: sink(payload),
    )
    interface, manager = _build_interface(monkeypatch, lambda_mode=False)
    model = _FakeModel(
        [],
        class_names=["wave"],
        sampling=VideoSampling(
            window_seconds=1,
            sample_fps=2,
            min_frames=2,
        ),
    )
    adapter = _adapter(model)
    manager.infer_from_request_sync.side_effect = (
        lambda model_id, request, **kwargs: adapter.infer_from_request(
            request, **kwargs
        )
    )
    value = (
        base64.b64encode(payload).decode()
        if transport == "base64"
        else "https://example.com/clip.mp4"
    )
    with TestClient(interface.app) as client:
        if route == "generic":
            if transport == "url":
                response = client.post(
                    "/clips/1", params={"image": value, "api_key": "k"}
                )
            else:
                response = client.post(
                    "/clips/1?api_key=k",
                    content=value,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
        else:
            endpoint = (
                "/sam3/video/embed" if route == "sam3" else "/infer/action_recognition"
            )
            body = {"api_key": "k", "video": {"type": transport, "value": value}}
            body.update(
                {"video_id": "clip-1"} if route == "sam3" else {"model_id": "clips/1"}
            )
            response = client.post(endpoint, json=body)

    assert response.status_code == (400 if manifest else 200), response.text
    calls = loads.model.calls if route == "sam3" else model.calls
    assert bool(calls) is (not manifest)
    if manifest:
        assert str(clip) not in response.text
    assert list(scratch.iterdir()) == []
