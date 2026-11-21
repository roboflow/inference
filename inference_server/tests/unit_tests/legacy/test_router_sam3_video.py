import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from inference_model_manager.hash_namespacing import namespace_client_hash_id
from inference_models.errors import ModelInputError
from inference_models.models.sam3_tracker_video.sam3_tracker_video_hf import (
    _prompts_by_object,
)
from inference_server.legacy import sam3_video
from inference_server.legacy.entities import Sam3VideoPropagateRequest
from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.legacy.test_action_recognition import clip_base64, write_clip

MODEL_ID = "sam3trackervideo"
API_KEY = "k"
RLE = {"size": [2, 2], "counts": "04"}
PROMPTS = [
    {"frame_idx": 0, "obj_id": 1, "x": 10.0, "y": 12.0, "label": 1},
    {"frame_idx": 3, "obj_id": 2, "x": 30.0, "y": 8.0, "label": 1},
    {"frame_idx": 3, "obj_id": 2, "x": 31.0, "y": 9.0, "label": 0},
]


@pytest.fixture(autouse=True)
def temp_dir(tmp_path, monkeypatch) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    return scratch


@pytest.fixture
def clip(tmp_path) -> Path:
    return write_clip(tmp_path / "clip.mp4")


def _gateway(fake_stat, model_class_name="SAM3TrackerVideo", max_frames=1507):
    fake_stat[MODEL_ID] = ("instance-segmentation", "embed_video")

    def _limits(image, params):
        return {
            "max_gb": 16.0,
            "bytes_per_frame": 10_616_832,
            "max_frames": max_frames,
        }

    def _embed(image, params):
        frame = params["frames"][0]
        return {
            "video_id": params["video_id"],
            "n_frames": len(params["frames"]),
            "height": frame.shape[0],
            "width": frame.shape[1],
        }

    def _segment(image, params):
        _prompts_by_object(params["prompts"], n_frames=25)  # the tracker's rules
        return {"frame_idx": params["frame_idx"], "masks": {"1": RLE}}

    def _propagate(image, params):
        _prompts_by_object(params["prompts"], n_frames=25)  # the tracker's rules
        return [
            {"frame_idx": i, "masks": {"1": RLE, "2": RLE}}
            for i in params["frame_indices"] or range(3)
        ]

    gateway = FakeGateway(
        predictions={
            (MODEL_ID, "video_cache_limits"): _limits,
            (MODEL_ID, "embed_video"): _embed,
            (MODEL_ID, "segment_video_frame"): _segment,
            (MODEL_ID, "propagate_video"): _propagate,
        },
        model_info={
            MODEL_ID: {
                "model_class_name": model_class_name,
                "actions": {
                    "video_cache_limits": {},
                    "embed_video": {},
                    "segment_video_frame": {},
                    "propagate_video": {},
                },
            }
        },
    )

    return gateway


def _infer_calls(gateway):
    """Model calls other than the embed route's question for the cache limits."""
    return [
        call
        for call in gateway.calls
        if call[0] == "infer" and call[2] != "video_cache_limits"
    ]


def test_embed_decodes_every_frame_and_scopes_the_id_to_the_key(
    legacy_client, fake_stat, clip
):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        "/sam3/video/embed",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "video": {"type": "base64", "value": clip_base64(clip)},
            "encode_frames": True,
            "keep_existing": True,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["video_id"] == "clip-1"
    assert body["n_frames"] == 25
    assert (body["height"], body["width"]) == (48, 64)
    [call] = _infer_calls(gateway)
    assert call[1:3] == (MODEL_ID, "embed_video")
    params = call[3]
    assert params["video_id"] == namespace_client_hash_id("clip-1", API_KEY)
    assert params["encode_frames"] is True
    assert params["keep_existing"] is True
    assert len(params["frames"]) == 25
    assert params["frames"][0].shape == (48, 64, 3)


def test_embed_refuses_a_clip_longer_than_the_cache_holds(
    legacy_client, fake_stat, clip
):
    gateway = _gateway(fake_stat, max_frames=10)

    response = legacy_client(gateway).post(
        "/sam3/video/embed",
        json={
            "api_key": API_KEY,
            "video_id": "clip-1",
            "video": {"type": "base64", "value": clip_base64(clip)},
        },
    )

    assert response.status_code == 413, response.text
    assert "The video has 25 frames; the most a clip may have is 10" in response.text
    assert "SAM3_VIDEO_CACHE_MAX_GB" in response.text
    assert _infer_calls(gateway) == []


def test_segment_passes_every_click_and_the_frame(legacy_client, fake_stat):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
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
    [call] = _infer_calls(gateway)
    assert call[2] == "segment_video_frame"
    assert call[3] == {
        "video_id": namespace_client_hash_id("clip-1", API_KEY),
        "prompts": PROMPTS,
        "frame_idx": 3,
    }


def test_propagate_returns_the_asked_frames(legacy_client, fake_stat):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
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
    [call] = _infer_calls(gateway)
    assert call[2] == "propagate_video"
    assert call[3]["frame_indices"] == [0, 2, 4]


def test_a_click_label_other_than_0_or_1_is_refused(legacy_client, fake_stat):
    gateway = _gateway(fake_stat)
    prompts = [dict(PROMPTS[0], label=2)]

    response = legacy_client(gateway).post(
        "/sam3/video/propagate",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": prompts},
    )

    assert response.status_code == 422, response.text
    assert _infer_calls(gateway) == []


def test_routes_refuse_a_model_that_is_not_the_video_tracker(legacy_client, fake_stat):
    gateway = _gateway(fake_stat, model_class_name="SAM3Torch")

    response = legacy_client(gateway).post(
        "/sam3/video/propagate",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": PROMPTS},
    )

    assert response.status_code == 400, response.text
    assert "SAM3TrackerVideo" in response.json()["message"]
    assert _infer_calls(gateway) == []


def _stream_lines(response):
    lines = [json.loads(line) for line in response.text.splitlines() if line]

    return lines


def test_propagate_stream_sends_each_object_as_it_is_tracked(legacy_client, fake_stat):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
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
    calls = _infer_calls(gateway)
    assert [call[2] for call in calls] == ["propagate_video", "propagate_video"]
    assert [call[3]["prompts"] for call in calls] == [PROMPTS[:1], PROMPTS[1:]]
    assert {call[3]["video_id"] for call in calls} == {
        namespace_client_hash_id("clip-1", API_KEY)
    }


def test_propagate_stream_ends_with_the_error_a_later_object_hit(
    legacy_client, fake_stat
):
    gateway = _gateway(fake_stat)

    def _propagate(image, params):
        if params["prompts"][0]["obj_id"] == 2:
            raise ModelInputError(message="No video embedded under id clip-1.")
        return [{"frame_idx": 0, "masks": {"1": RLE}}]

    gateway.predictions[(MODEL_ID, "propagate_video")] = _propagate

    response = legacy_client(gateway).post(
        "/sam3/video/propagate/stream",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": PROMPTS},
    )

    assert response.status_code == 200, response.text
    lines = _stream_lines(response)
    assert [line["type"] for line in lines] == ["object", "error"]
    assert lines[1]["status"] == 400
    assert "No video embedded" in lines[1]["message"]


def test_propagate_stream_refuses_an_object_without_an_include_point(
    legacy_client, fake_stat
):
    gateway = _gateway(fake_stat)
    prompts = PROMPTS + [{"frame_idx": 1, "obj_id": 3, "x": 1.0, "y": 1.0, "label": 0}]

    response = legacy_client(gateway).post(
        "/sam3/video/propagate/stream",
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": prompts},
    )

    assert response.status_code == 400, response.text
    assert "[3]" in response.json()["message"]
    assert _infer_calls(gateway) == []


CORRECTED = [
    {"frame_idx": 4, "obj_id": 1, "x": 10.0, "y": 12.0, "label": 1},
    {"frame_idx": 9, "obj_id": 1, "x": 40.0, "y": 20.0, "label": 0},
    {"frame_idx": 1, "obj_id": 1, "x": 40.0, "y": 20.0, "label": 0},
]


@pytest.mark.parametrize(
    "route, extra",
    [
        ("/sam3/video/segment", {"frame_idx": 9}),
        ("/sam3/video/propagate", {}),
        ("/sam3/video/propagate/stream", {}),
    ],
)
def test_exclude_points_alone_are_accepted_off_the_objects_anchor(
    legacy_client, fake_stat, route, extra
):
    gateway = _gateway(fake_stat)

    response = legacy_client(gateway).post(
        route,
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": CORRECTED, **extra},
    )

    assert response.status_code == 200, response.text
    assert "error" not in response.text
    [call] = _infer_calls(gateway)
    assert call[3]["prompts"] == CORRECTED


NO_INCLUDE_POINT = "Objects [3] have no include point; every object needs one."


@pytest.mark.parametrize(
    "route, extra, message",
    [
        (
            "/sam3/video/segment",
            {"frame_idx": 1},
            f"Error with model input. Cause: {NO_INCLUDE_POINT}",
        ),
        (
            "/sam3/video/propagate",
            {},
            f"Error with model input. Cause: {NO_INCLUDE_POINT}",
        ),
        ("/sam3/video/propagate/stream", {}, NO_INCLUDE_POINT),
    ],
)
def test_an_object_with_exclude_points_only_is_refused(
    legacy_client, fake_stat, route, extra, message
):
    gateway = _gateway(fake_stat)
    prompts = PROMPTS + [
        {"frame_idx": 1, "obj_id": 3, "x": 1.0, "y": 1.0, "label": 0},
        {"frame_idx": 6, "obj_id": 3, "x": 1.0, "y": 1.0, "label": 0},
    ]

    response = legacy_client(gateway).post(
        route,
        json={"api_key": API_KEY, "video_id": "clip-1", "prompts": prompts, **extra},
    )

    assert response.status_code == 400, response.text
    assert response.json()["message"] == message


@pytest.mark.asyncio
async def test_propagate_stream_sends_heartbeats_while_an_object_is_tracked(
    monkeypatch,
):
    monkeypatch.setattr(sam3_video, "PROPAGATE_HEARTBEAT_SECONDS", 0.01)
    release = asyncio.Event()

    class SlowBridge:
        async def infer_params_only(self, route, api_key, action, params):
            await release.wait()
            return [{"frame_idx": 0, "masks": {"1": RLE}}]

    request = Sam3VideoPropagateRequest(
        api_key=API_KEY, video_id="clip-1", prompts=PROMPTS[:1]
    )
    body = sam3_video.stream_propagation(
        None, API_KEY, SlowBridge(), inference_request=request
    )

    first = json.loads(await body.__anext__())
    second = json.loads(await body.__anext__())
    release.set()
    rest = [json.loads(line) async for line in body]

    assert (first["type"], second["type"]) == ("heartbeat", "heartbeat")
    assert [line["type"] for line in rest][-2:] == ["object", "done"]


@pytest.mark.asyncio
async def test_closing_the_propagate_stream_stops_the_tracking(monkeypatch):
    monkeypatch.setattr(sam3_video, "PROPAGATE_HEARTBEAT_SECONDS", 0.01)
    calls = []

    class SlowBridge:
        async def infer_params_only(self, route, api_key, action, params):
            calls.append(params["prompts"][0]["obj_id"])
            await asyncio.sleep(3600)

    request = Sam3VideoPropagateRequest(
        api_key=API_KEY, video_id="clip-1", prompts=PROMPTS
    )
    body = sam3_video.stream_propagation(
        None, API_KEY, SlowBridge(), inference_request=request
    )

    await body.__anext__()
    await body.aclose()
    await asyncio.sleep(0)

    assert calls == [1]
