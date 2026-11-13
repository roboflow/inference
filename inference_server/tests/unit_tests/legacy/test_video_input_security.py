"""Real decoding at the SAM3, explicit action and generic action HTTP routes."""

import base64
import tempfile

import pytest

from inference_server.legacy import video
from tests.unit_tests.legacy.test_action_recognition import write_clip
from tests.unit_tests.legacy.test_router_action_recognition import (
    _gateway as action_gateway,
)
from tests.unit_tests.legacy.test_router_sam3_video import _gateway as sam3_gateway


@pytest.mark.parametrize("route", ["sam3", "action", "generic"])
@pytest.mark.parametrize("transport", ["base64", "url"])
@pytest.mark.parametrize("manifest", [False, True])
def test_offline_routes_only_decode_the_submitted_asset(
    tmp_path, monkeypatch, legacy_client, fake_stat, route, transport, manifest
):
    clip = write_clip(tmp_path / "server-only.mp4")
    payload = clip.read_bytes()
    if manifest:
        payload = (
            "#EXTM3U\n#EXT-X-TARGETDURATION:1\n"
            f"#EXTINF:1.0,\n{clip}\n#EXT-X-ENDLIST\n"
        ).encode()
    scratch = tmp_path / "uploads"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(video, "ALLOW_URL_INPUT", True)
    monkeypatch.setattr(video, "LEGACY_OFFLINE_MODE", False)
    monkeypatch.setattr(video, "_check_url", lambda value: (value, None))

    async def fetch(url, *, sink, **kwargs):
        sink(payload)

    monkeypatch.setattr(video, "fetch_to_sink", fetch)
    gateway = sam3_gateway(fake_stat) if route == "sam3" else action_gateway(fake_stat)
    client = legacy_client(gateway)
    value = (
        base64.b64encode(payload).decode()
        if transport == "base64"
        else "https://example.com/clip.mp4"
    )
    if route == "generic":
        if transport == "url":
            response = client.post("/clips/1", params={"image": value, "api_key": "k"})
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
    calls = [
        call
        for call in gateway.calls
        if call[0] == "infer" and call[2] != "video_cache_limits"
    ]
    assert bool(gateway.embedded_frames if route == "sam3" else calls) is (not manifest)
    if manifest:
        assert str(clip) not in response.text
    assert list(scratch.iterdir()) == []
