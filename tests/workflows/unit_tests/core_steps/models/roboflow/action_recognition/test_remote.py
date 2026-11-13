"""Remote window classification for the Action Recognition Model block."""

import base64
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pytest
from roboflow_workflows.core_steps.models.roboflow.action_recognition import remote
from roboflow_workflows.core_steps.models.roboflow.action_recognition.remote import (
    encode_window,
    load_remote_action_recognition_model,
)

import inference_models
from inference.core.utils.video_utils import probe_video, read_frame_windows
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models import ActionRecognitionContract, VideoSampling
from inference_models.models.base.action_recognition import plan_windows
from inference_sdk.config import outbound_service_secret, usage_source_tags

PUNCH_SAMPLING = VideoSampling(
    window_seconds=32 / 12,
    sample_fps=12.0,
    min_frames=4,
    max_frame_side=360,
    max_frames=32,
)
BLOCK_SAMPLING = VideoSampling(
    window_seconds=1.0, sample_fps=4.0, min_frames=2, max_frames=4
)
ENDPOINT_URL = "http://server:9001/infer/action_recognition"


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _FakeServer:
    """Answers like /infer/action_recognition after reading the sent clip."""

    def __init__(self, timeline: List[dict]):
        self.timeline = timeline
        self.requests: List[dict] = []

    def post(self, url, json, timeout, params=None, headers=None):
        self.requests.append(
            {"url": url, "json": json, "params": params, "headers": headers}
        )
        clip = base64.b64decode(json["video"]["value"])
        handle, path = tempfile.mkstemp(suffix=".video")
        os.close(handle)
        try:
            Path(path).write_bytes(clip)
            source_fps, frame_count = probe_video(path=path)
        finally:
            os.remove(path)
        payload = {
            "timeline": self.timeline,
            "source_fps": source_fps,
            "frame_count": frame_count,
            "windows_classified": 1,
        }

        return _FakeResponse(payload=payload)


def _frames(count: int, height: int = 24, width: int = 32) -> List[np.ndarray]:
    generator = np.random.default_rng(seed=7)
    frames = [
        generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for _ in range(count)
    ]

    return frames


def _fake_platform(monkeypatch, post, **contract_fields) -> list:
    """Fake only the contract lookup and the HTTP call."""
    contract = ActionRecognitionContract(**contract_fields)
    lookups = []

    def get_action_recognition_contract(model_id, api_key):
        lookups.append((model_id, api_key))
        return contract

    monkeypatch.setattr(
        inference_models,
        "get_action_recognition_contract",
        get_action_recognition_contract,
    )
    monkeypatch.setattr(remote, "WORKFLOWS_REMOTE_API_TARGET", "self-hosted")
    monkeypatch.setattr(remote, "LOCAL_INFERENCE_API_URL", "http://server:9001/")
    monkeypatch.setattr(remote.requests, "post", post)

    return lookups


def _read_back_through_the_endpoint_clip_path(clip: bytes):
    handle, path = tempfile.mkstemp(suffix=".video")
    os.close(handle)
    try:
        Path(path).write_bytes(clip)
        source_fps, probed_count = probe_video(path=path)
        windows = plan_windows(
            frame_count=probed_count, source_fps=source_fps, sampling=PUNCH_SAMPLING
        )
        decoded = list(
            read_frame_windows(
                path=path, windows=[window.frame_indices for window in windows]
            )
        )
    finally:
        os.remove(path)

    return source_fps, probed_count, windows, decoded


@pytest.mark.parametrize("frame_count", [4, 9, 31, 32])
def test_lossless_window_reads_back_exactly_through_the_endpoint_clip_path(
    frame_count,
):
    frames = _frames(frame_count)

    source_fps, probed_count, windows, decoded = (
        _read_back_through_the_endpoint_clip_path(
            encode_window(frames, fps=12.0, lossless=True)
        )
    )

    assert (source_fps, probed_count) == (12.0, frame_count)
    assert [list(window.frame_indices) for window in windows] == [
        list(range(frame_count))
    ]
    assert all(np.array_equal(sent, read) for sent, read in zip(frames, decoded[0]))


@pytest.mark.parametrize("frame_count", [4, 31, 32])
def test_default_window_is_a_smaller_lossy_clip_that_still_maps_one_to_one(
    frame_count,
):
    # A moving gradient compresses like video, unlike random noise.
    ramp = np.tile(np.arange(64, dtype=np.uint8) * 4, (48, 1))
    frames = [
        np.dstack([np.roll(ramp, shift, axis=1)] * 3) for shift in range(frame_count)
    ]

    clip = encode_window(frames, fps=12.0)
    source_fps, probed_count, windows, decoded = (
        _read_back_through_the_endpoint_clip_path(clip)
    )

    assert clip[4:8] == b"ftyp"
    assert len(clip) < len(encode_window(frames, fps=12.0, lossless=True))
    assert (source_fps, probed_count) == (12.0, frame_count)
    assert [list(window.frame_indices) for window in windows] == [
        list(range(frame_count))
    ]
    assert [read.shape for read in decoded[0]] == [sent.shape for sent in frames]


def test_continuous_spans_come_back_half_open_with_their_confidence(monkeypatch):
    server = _FakeServer(
        timeline=[
            {
                "start_frame_idx": 2,
                "end_frame_idx": 5,
                "class": "jab",
                "class_id": 1,
                "confidence": 0.8,
            }
        ]
    )
    _fake_platform(
        monkeypatch,
        server.post,
        class_names=["cross", "jab"],
        video_sampling=VideoSampling(window_seconds=2.0, sample_fps=8.0, max_frames=16),
        supports_confidence=True,
        supports_observed_duration=True,
        end_exclusive_spans=True,
    )
    model = load_remote_action_recognition_model("ws/jabs/1", api_key="key")

    segments = model.infer(
        frames=_frames(16), fps=8.0, confidence=0.4, duration_seconds=1.5
    )

    assert [
        (s.start_frame_idx, s.end_frame_idx, s.confidence, s.end_exclusive)
        for s in segments
    ] == [(2, 6, 0.8, True)]
    sent = server.requests[0]["json"]
    assert sent["confidence"] == 0.4
    assert "class_filter" not in sent
    assert "duration_seconds" not in sent


@pytest.mark.parametrize(
    "frame_count, match",
    [(1, "at least two frames"), (3, "would repeat frames"), (65, "would split it")],
)
def test_window_the_endpoint_would_pad_or_split_is_refused(
    monkeypatch, frame_count, match
):
    server = _FakeServer(timeline=[])
    _fake_platform(
        monkeypatch, server.post, class_names=None, video_sampling=VideoSampling()
    )
    model = load_remote_action_recognition_model("ws/m/1", api_key=None)

    with pytest.raises(ValueError, match=match):
        model.infer(frames=_frames(frame_count), fps=4.0)

    assert server.requests == []


def test_a_server_that_read_another_clip_is_an_error(monkeypatch):
    def post(url, json, timeout, **kwargs):
        return _FakeResponse(
            {
                "timeline": [],
                "source_fps": 12.0,
                "frame_count": 31,
                "windows_classified": 1,
            }
        )

    _fake_platform(monkeypatch, post, class_names=None, video_sampling=PUNCH_SAMPLING)
    model = load_remote_action_recognition_model("ws/m/1", api_key=None)

    with pytest.raises(RuntimeError, match="cannot map back"):
        model.infer(frames=_frames(32), fps=12.0)


@pytest.mark.parametrize(
    "transport, key_in_body, headers",
    [
        ("legacy", True, None),
        ("both", True, {"Authorization": "Bearer key"}),
        ("header", False, {"Authorization": "Bearer key"}),
    ],
)
def test_request_carries_workflow_attribution_billing_opt_out_and_key_transport(
    monkeypatch, transport, key_in_body, headers
):
    # An internal workflow run that opted out of billing must not have every
    # window billed to the customer's key, as with the other remote models.
    server = _FakeServer(timeline=[])
    _fake_platform(
        monkeypatch, server.post, class_names=None, video_sampling=PUNCH_SAMPLING
    )
    monkeypatch.setattr(remote, "WORKFLOWS_REMOTE_API_KEY_TRANSPORT", transport)
    model = load_remote_action_recognition_model("ws/m/1", api_key="key")
    tags = {"source": "workflow", "source_info": "ws/flow"}
    tags_token = usage_source_tags.set(tags)
    secret_token = outbound_service_secret.set("service-secret")
    try:
        model.infer(_frames(32), fps=12.0)
    finally:
        outbound_service_secret.reset(secret_token)
        usage_source_tags.reset(tags_token)

    request = server.requests[0]
    assert {key: request["json"][key] for key in tags} == tags
    assert ("api_key" in request["json"]) is key_in_body
    assert request["headers"] == headers
    assert request["params"] == {
        **tags,
        "source_info_origin": "context",
        "service_secret": "service-secret",
        "countinference": False,
    }


def test_server_errors_hide_the_api_key(monkeypatch):
    def post(url, json, timeout, **kwargs):
        return _FakeResponse(
            {"message": "no model for api_key=secret123"}, status_code=404
        )

    _fake_platform(monkeypatch, post, class_names=None, video_sampling=PUNCH_SAMPLING)
    model = load_remote_action_recognition_model("ws/m/1", api_key="secret123")

    with pytest.raises(remote.requests.HTTPError) as error:
        model.infer(frames=_frames(8), fps=12.0)

    assert "404" in str(error.value)
    assert "secret123" not in str(error.value)


def _video_frame(frame_number: int) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"clip:{frame_number}"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="clip:root"),
        video_metadata=VideoMetadata(
            video_identifier="clip",
            frame_number=frame_number,
            fps=8.0,
            frame_timestamp=datetime(2024, 1, 1),
        ),
        numpy_image=np.full((24, 32, 3), frame_number % 255, dtype=np.uint8),
    )


def _remote_block(monkeypatch, *, timeline: List[dict]):
    server = _FakeServer(timeline=timeline)
    lookups = _fake_platform(
        monkeypatch, server.post, class_names=["punch"], video_sampling=BLOCK_SAMPLING
    )
    block = ActionRecognitionModelBlockV1(
        api_key="key",
        step_execution_mode=StepExecutionMode.REMOTE,
        model_manager=None,
    )

    return block, server, lookups


def test_block_keeps_its_state_and_sends_only_window_calls_in_remote_mode(
    monkeypatch,
):
    block, server, lookups = _remote_block(
        monkeypatch,
        timeline=[{"start_frame_idx": 1, "end_frame_idx": 2, "class": "punch"}],
    )

    results = [
        block.run(
            images=[_video_frame(number)],
            model_id="ws/punch/1",
            class_filter=["punch"],
        )[0]
        for number in range(17)
    ]

    assert lookups == [("ws/punch/1", "key")]
    assert len(server.requests) == 2
    request = server.requests[0]
    assert request["url"] == ENDPOINT_URL
    assert request["json"]["model_id"] == "ws/punch/1"
    assert request["json"]["api_key"] == "key"
    assert request["json"]["class_filter"] == ["punch"]
    assert "confidence" not in request["json"]
    assert all(result["error_status"] == "" for result in results)
    # Window 1 samples source frames 2, 4, 6, 8 and window 2 samples 10..16;
    # sampled indices 1..2 map to source frames 4..6 and 12..14.
    assert [
        (entry.start_frame_idx, entry.end_frame_idx, entry.class_name, entry.class_id)
        for entry in results[-1]["timeline"]
    ] == [(4, 6, "punch", 0), (12, 14, "punch", 0)]
