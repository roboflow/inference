import asyncio
import base64
import hashlib
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import pytest
from starlette.testclient import TestClient

from inference.core.entities.requests.sam3_video import Sam3VideoPropagateRequest
from inference.core.interfaces.http import sam3_video
from inference_models.errors import ModelInputError

MODEL_ID = "sam3trackervideo"
API_KEY = "k"
RLE = {"size": [2, 2], "counts": "04"}
PROMPTS = [
    {"frame_idx": 0, "obj_id": 1, "x": 10.0, "y": 12.0, "label": 1},
    {"frame_idx": 3, "obj_id": 2, "x": 30.0, "y": 8.0, "label": 1},
    {"frame_idx": 3, "obj_id": 2, "x": 31.0, "y": 9.0, "label": 0},
]
SCOPED_VIDEO_ID = hashlib.sha256(API_KEY.encode()).hexdigest()[:16] + ":" + "clip-1"


class _DummyInstrumentator:
    def __init__(self, app, model_manager, endpoint="/metrics"):
        self.app = app
        self.model_manager = model_manager
        self.endpoint = endpoint

    def set_stream_manager_client(self, stream_manager_client) -> None:
        self.stream_manager_client = stream_manager_client


class SAM3TrackerVideo:
    """Stands in for the inference_models tracker; records every call."""

    def __init__(self):
        self.calls = []
        self.max_frames = 1507

    def video_cache_limits(self):
        return {
            "max_gb": 16.0,
            "bytes_per_frame": 10_616_832,
            "max_frames": self.max_frames,
        }

    def embed_video(self, frames, *, video_id, encode_frames=False):
        self.calls.append(
            (
                "embed_video",
                {
                    "frames": frames,
                    "video_id": video_id,
                    "encode_frames": encode_frames,
                },
            )
        )
        return {
            "video_id": video_id,
            "n_frames": len(frames),
            "height": frames[0].shape[0],
            "width": frames[0].shape[1],
        }

    def segment_video_frame(self, video_id, *, prompts, frame_idx):
        self.calls.append(
            (
                "segment_video_frame",
                {"video_id": video_id, "prompts": prompts, "frame_idx": frame_idx},
            )
        )
        return {"frame_idx": frame_idx, "masks": {"1": RLE}}

    def propagate_video(self, video_id, *, prompts, frame_indices=None):
        self.calls.append(
            (
                "propagate_video",
                {
                    "video_id": video_id,
                    "prompts": prompts,
                    "frame_indices": frame_indices,
                },
            )
        )
        return [
            {"frame_idx": i, "masks": {"1": RLE, "2": RLE}}
            for i in frame_indices or range(3)
        ]


class SAM3Torch:
    pass


def write_clip(path: Path, frame_count: int = 25) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 48))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:] = (200, 20 + index * 8, 60)
        writer.write(frame)
    writer.release()

    return path


def clip_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


@pytest.fixture
def clip(tmp_path) -> Path:
    return write_clip(tmp_path / "clip.mp4")


class _Loads:
    """Model ids the tracker loads, and the model each load returns."""

    def __init__(self):
        self.ids = []
        self.model = SAM3TrackerVideo()


@pytest.fixture
def loads(monkeypatch) -> _Loads:
    recorded = _Loads()

    def _load(self, model_id, **kwargs):
        recorded.ids.append(model_id)
        return recorded.model

    monkeypatch.setattr(sam3_video.Sam3VideoTracker, "_load", _load)

    return recorded


@pytest.fixture
def client(monkeypatch, loads):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "CORE_MODEL_SAM3_ENABLED", True)
    monkeypatch.setattr(http_api, "SAM3_VIDEO_ENABLED", True)
    monkeypatch.setattr(http_api, "SAM3_EXEC_MODE", "local")
    model_manager = MagicMock()
    model_manager.pingback = None
    model_manager.num_errors = 0
    interface = http_api.HttpInterface(model_manager=model_manager)

    with TestClient(interface.app) as test_client:
        yield test_client


def test_routes_are_absent_unless_enabled(monkeypatch):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", _DummyInstrumentator)
    monkeypatch.setattr(http_api, "SAM3_VIDEO_ENABLED", False)
    model_manager = MagicMock()
    model_manager.pingback = None
    interface = http_api.HttpInterface(model_manager=model_manager)

    paths = {route.path for route in interface.app.routes}

    assert not any(path.startswith("/sam3/video/") for path in paths)


def test_embed_decodes_every_frame_and_scopes_the_id_to_the_key(client, loads, clip):
    response = client.post(
        "/sam3/video/embed",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "video": {"type": "base64", "value": clip_base64(clip)},
            "encode_frames": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["video_id"] == "clip-1"
    assert body["n_frames"] == 25
    assert (body["height"], body["width"]) == (48, 64)
    [(action, params)] = loads.model.calls
    assert action == "embed_video"
    assert params["video_id"] == SCOPED_VIDEO_ID
    assert params["encode_frames"] is True
    assert len(params["frames"]) == 25
    assert params["frames"][0].shape == (48, 64, 3)
    assert loads.ids == [MODEL_ID]


def test_embed_refuses_a_clip_longer_than_the_cache_holds(client, loads, clip):
    loads.model.max_frames = 10

    response = client.post(
        "/sam3/video/embed",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "video": {"type": "base64", "value": clip_base64(clip)},
        },
    )

    assert response.status_code == 413, response.text
    message = response.json()["message"]
    assert "The video has 25 frames; the most a clip may have is 10" in message
    assert "SAM3_VIDEO_CACHE_MAX_GB" in message
    assert loads.model.calls == []


def test_a_clip_whose_header_understates_its_frames_is_refused(clip, monkeypatch):
    monkeypatch.setattr(sam3_video, "probe_video", lambda path: (25.0, 5))

    with pytest.raises(sam3_video.PayloadTooLargeError, match="more than 10 frames"):
        sam3_video.read_clip_frames(str(clip), max_frames=10)


def test_segment_passes_every_click_and_the_frame(client, loads):
    response = client.post(
        "/sam3/video/segment",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "prompts": PROMPTS,
            "frame_idx": 3,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["frame_idx"] == 3
    assert body["masks"] == {"1": RLE}
    [(action, params)] = loads.model.calls
    assert action == "segment_video_frame"
    assert params == {"video_id": SCOPED_VIDEO_ID, "prompts": PROMPTS, "frame_idx": 3}


def test_a_clip_that_is_not_embedded_answers_400(client, loads):
    def _segment(video_id, *, prompts, frame_idx):
        raise ModelInputError(message="No video embedded under id clip-1.")

    loads.model.segment_video_frame = _segment

    response = client.post(
        "/sam3/video/segment",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "prompts": PROMPTS,
            "frame_idx": 0,
        },
    )

    assert response.status_code == 400, response.text
    assert "No video embedded" in response.json()["message"]


def test_propagate_returns_the_asked_frames(client, loads):
    response = client.post(
        "/sam3/video/propagate",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "prompts": PROMPTS,
            "frame_indices": [0, 2, 4],
        },
    )

    assert response.status_code == 200, response.text
    frames = response.json()["frames"]
    assert [f["frame_idx"] for f in frames] == [0, 2, 4]
    assert frames[0]["masks"] == {"1": RLE, "2": RLE}
    [(action, params)] = loads.model.calls
    assert action == "propagate_video"
    assert params["frame_indices"] == [0, 2, 4]


def test_the_model_is_loaded_once_and_reused(client, loads):
    for _ in range(2):
        response = client.post(
            "/sam3/video/propagate",
            json={"api_key": API_KEY, "video_id": "clip-1", "prompts": PROMPTS},
        )
        assert response.status_code == 200, response.text

    assert loads.ids == [MODEL_ID]
    assert len(loads.model.calls) == 2


def test_a_click_label_other_than_0_or_1_is_refused(client, loads):
    prompts = [dict(PROMPTS[0], label=2)]

    response = client.post(
        "/sam3/video/propagate",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": prompts},
    )

    assert response.status_code == 422, response.text
    assert loads.ids == []


def test_routes_refuse_a_model_that_is_not_the_video_tracker(client, loads):
    loads.model = SAM3Torch()

    response = client.post(
        "/sam3/video/propagate",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": PROMPTS},
    )

    assert response.status_code == 400, response.text
    assert "SAM3TrackerVideo" in response.json()["message"]


def test_routes_answer_501_in_remote_sam3_mode(client, loads, monkeypatch):
    import inference.core.interfaces.http.http_api as http_api

    monkeypatch.setattr(http_api, "SAM3_EXEC_MODE", "remote")

    response = client.post(
        "/sam3/video/propagate",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": PROMPTS},
    )

    assert response.status_code == 501, response.text
    assert loads.ids == []


def test_a_key_without_access_to_the_model_is_refused(client, loads, monkeypatch):
    checks = []

    def _check(**kwargs):
        checks.append(kwargs)
        return False

    monkeypatch.setattr(sam3_video, "MODELS_CACHE_AUTH_ENABLED", True)
    monkeypatch.setattr(sam3_video, "OFFLINE_MODE", False)
    monkeypatch.setattr(sam3_video, "_check_if_api_key_has_access_to_model", _check)

    response = client.post(
        "/sam3/video/propagate",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": PROMPTS},
    )

    assert response.status_code == 401, response.text
    assert [(c["api_key"], c["model_id"]) for c in checks] == [(API_KEY, MODEL_ID)]
    assert loads.ids == []


def _stream_lines(response):
    lines = [json.loads(line) for line in response.text.splitlines() if line]

    return lines


def test_propagate_stream_sends_each_object_as_it_is_tracked(client, loads):
    response = client.post(
        "/sam3/video/propagate/stream",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "prompts": PROMPTS,
            "frame_indices": [0, 2],
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = _stream_lines(response)
    assert [line["type"] for line in lines] == ["object", "object", "done"]
    assert [line["obj_id"] for line in lines[:2]] == [1, 2]
    assert lines[0]["frames"] == [
        {"frame_idx": 0, "mask": RLE},
        {"frame_idx": 2, "mask": RLE},
    ]
    calls = loads.model.calls
    assert [action for action, _ in calls] == ["propagate_video", "propagate_video"]
    assert [params["prompts"] for _, params in calls] == [PROMPTS[:1], PROMPTS[1:]]
    assert {params["video_id"] for _, params in calls} == {SCOPED_VIDEO_ID}


def test_propagate_stream_ends_with_the_error_a_later_object_hit(client, loads):
    def _propagate(video_id, *, prompts, frame_indices=None):
        if prompts[0]["obj_id"] == 2:
            raise ModelInputError(message="No video embedded under id clip-1.")
        return [{"frame_idx": 0, "masks": {"1": RLE}}]

    loads.model.propagate_video = _propagate

    response = client.post(
        "/sam3/video/propagate/stream",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": PROMPTS},
    )

    assert response.status_code == 200, response.text
    lines = _stream_lines(response)
    assert [line["type"] for line in lines] == ["object", "error"]
    assert lines[1]["status"] == 400
    assert "No video embedded" in lines[1]["message"]


def test_propagate_stream_refuses_an_object_without_an_include_point(client, loads):
    prompts = PROMPTS + [{"frame_idx": 1, "obj_id": 3, "x": 1.0, "y": 1.0, "label": 0}]

    response = client.post(
        "/sam3/video/propagate/stream",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": prompts},
    )

    assert response.status_code == 400, response.text
    assert "[3]" in response.json()["message"]
    assert loads.model.calls == []


def test_the_tracker_loads_only_with_the_servers_local_package_policy(monkeypatch):
    import inference_models

    loaded = []

    def _from_pretrained(**kwargs):
        loaded.append(kwargs)
        return SAM3TrackerVideo()

    monkeypatch.setattr(inference_models.AutoModel, "from_pretrained", _from_pretrained)
    tracker = sam3_video.Sam3VideoTracker(
        MagicMock(content_addressed_artifact_cache=None)
    )

    tracker._load(
        "/models/elsewhere", api_key=API_KEY, countinference=None, service_secret=None
    )

    [kwargs] = loaded
    assert kwargs["model_id_or_path"] == "/models/elsewhere"
    assert kwargs["allow_direct_local_storage_loading"] is False
    assert kwargs["allow_untrusted_packages"] is False


class _BlockingModel:
    """Tracks an object only once ``release`` is set."""

    def __init__(self):
        self.release = threading.Event()
        self.calls = []

    def propagate_video(self, video_id, *, prompts, frame_indices=None):
        self.calls.append(prompts[0]["obj_id"])
        assert self.release.wait(timeout=10)
        return [{"frame_idx": 0, "masks": {str(prompts[0]["obj_id"]): RLE}}]


@pytest.mark.asyncio
async def test_propagate_stream_sends_heartbeats_while_an_object_is_tracked(
    monkeypatch,
):
    monkeypatch.setattr(sam3_video, "PROPAGATE_HEARTBEAT_SECONDS", 0.01)
    model = _BlockingModel()
    request = Sam3VideoPropagateRequest(
        api_key=API_KEY, video_id="clip-1", prompts=PROMPTS[:1]
    )
    body = sam3_video.stream_propagation(model, inference_request=request)

    first = json.loads(await body.__anext__())
    second = json.loads(await body.__anext__())
    model.release.set()
    rest = [json.loads(line) async for line in body]

    assert (first["type"], second["type"]) == ("heartbeat", "heartbeat")
    assert [line["type"] for line in rest][-2:] == ["object", "done"]


@pytest.mark.asyncio
async def test_closing_the_propagate_stream_stops_the_tracking(monkeypatch):
    monkeypatch.setattr(sam3_video, "PROPAGATE_HEARTBEAT_SECONDS", 0.01)
    model = _BlockingModel()
    request = Sam3VideoPropagateRequest(
        api_key=API_KEY, video_id="clip-1", prompts=PROMPTS
    )
    body = sam3_video.stream_propagation(model, inference_request=request)

    await body.__anext__()
    await body.aclose()
    model.release.set()
    await asyncio.sleep(0.1)

    assert model.calls == [1]
