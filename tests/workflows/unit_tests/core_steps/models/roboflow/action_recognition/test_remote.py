"""Remote window classification for the Action Recognition Model block."""

import base64
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.models.roboflow.action_recognition import remote
from roboflow_workflows.core_steps.models.roboflow.action_recognition.remote import (
    encode_window,
    load_remote_action_recognition_model,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v1_tensor import (
    ActionRecognitionModelBlockV1 as TensorActionRecognitionModelBlockV1,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v2 import (
    ActionRecognitionModelBlockV2,
)

import inference_models
from inference.core.utils.video_utils import probe_video, read_frame_windows
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
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
    """Answers like /infer/action_recognition after reading the sent clip.

    Windows are planned as the endpoint plans them, from the rate the clip's
    container declares and the model's sampling.
    """

    def __init__(self, timeline: List[dict]):
        self.timeline = timeline
        self.requests: List[dict] = []
        self.sampling: Optional[VideoSampling] = None
        self.decoded_shapes: List[tuple] = []

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
            windows = plan_windows(
                frame_count=frame_count, source_fps=source_fps, sampling=self.sampling
            )
            decoded = list(
                read_frame_windows(
                    path=path, windows=[window.frame_indices for window in windows]
                )
            )
        finally:
            os.remove(path)
        self.decoded_shapes.append(decoded[0][0].shape)
        payload = {
            "timeline": self.timeline,
            "source_fps": source_fps,
            "frame_count": frame_count,
            "windows_classified": len(windows),
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
    server = getattr(post, "__self__", None)
    if isinstance(server, _FakeServer):
        server.sampling = contract.video_sampling
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
def test_window_clip_maps_one_to_one_through_the_endpoint_clip_path(frame_count):
    frames = _frames(frame_count)

    clip = encode_window(frames, fps=12.0)
    source_fps, probed_count, windows, decoded = (
        _read_back_through_the_endpoint_clip_path(clip)
    )

    assert clip[4:8] == b"ftyp"
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


def test_connection_errors_hide_the_service_secret(monkeypatch):
    # requests quotes the request URL, query string included, in its errors.
    def post(url, json, timeout, params=None, **kwargs):
        raise remote.requests.ConnectionError(
            f"Max retries exceeded with url: /infer/action_recognition?"
            f"service_secret={params['service_secret']}&countinference=False"
        )

    _fake_platform(monkeypatch, post, class_names=None, video_sampling=PUNCH_SAMPLING)
    model = load_remote_action_recognition_model("ws/m/1", api_key="key")
    secret_token = outbound_service_secret.set("shared-service-secret")
    try:
        with pytest.raises(remote.requests.RequestException) as error:
            model.infer(frames=_frames(8), fps=12.0)
    finally:
        outbound_service_secret.reset(secret_token)

    assert "Max retries exceeded" in str(error.value)
    assert "shared-service-secret" not in str(error.value)
    # The block logs the error with its traceback; the original must not ride along.
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    "api_key", ["secret-key-123\n", "secret\nkey-123", "secret\r\nkey-123"]
)
def test_rejected_header_error_hides_the_api_key(monkeypatch, api_key):
    # requests quotes a header value it rejects, with repr escaping line breaks.
    def post(url, json, timeout, headers=None, **kwargs):
        remote.requests.Request("POST", url, headers=headers).prepare()

    _fake_platform(monkeypatch, post, class_names=None, video_sampling=PUNCH_SAMPLING)
    monkeypatch.setattr(remote, "WORKFLOWS_REMOTE_API_KEY_TRANSPORT", "header")
    model = load_remote_action_recognition_model("ws/m/1", api_key=api_key)

    with pytest.raises(remote.requests.RequestException) as error:
        model.infer(frames=_frames(8), fps=12.0)

    assert "Authorization header" in str(error.value)
    assert "secret" not in str(error.value)
    assert "key-123" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("fps", [30000 / 1001, 10 / 3, 7.4925])
def test_rate_the_clip_container_rounds_still_maps_one_to_one(monkeypatch, fps):
    # The mp4 container keeps about three decimals of the frame rate.
    server = _FakeServer(timeline=[])
    sampling = VideoSampling(
        window_seconds=16.0, sample_fps=fps, min_frames=4, max_frames=32
    )
    _fake_platform(monkeypatch, server.post, class_names=None, video_sampling=sampling)
    model = load_remote_action_recognition_model("ws/m/1", api_key=None)

    assert model.infer(frames=_frames(8), fps=fps) == []


def _video_frame(frame_number: int, fps: float = 8.0) -> WorkflowImageData:
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"clip:{frame_number}"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="clip:root"),
        video_metadata=VideoMetadata(
            video_identifier="clip",
            frame_number=frame_number,
            fps=fps,
            frame_timestamp=datetime(2024, 1, 1),
        ),
        numpy_image=np.full((24, 32, 3), frame_number % 255, dtype=np.uint8),
    )


def _remote_block(
    monkeypatch,
    *,
    timeline: List[dict],
    video_sampling: VideoSampling = BLOCK_SAMPLING,
    block_type=ActionRecognitionModelBlockV1,
):
    server = _FakeServer(timeline=timeline)
    lookups = _fake_platform(
        monkeypatch, server.post, class_names=["punch"], video_sampling=video_sampling
    )
    block = block_type(
        api_key="key",
        step_execution_mode=StepExecutionMode.REMOTE,
        model_manager=None,
    )

    return block, server, lookups


@pytest.mark.parametrize(
    "block_type", [ActionRecognitionModelBlockV1, ActionRecognitionModelBlockV2]
)
def test_block_keeps_its_state_and_sends_only_window_calls_in_remote_mode(
    monkeypatch,
    block_type,
):
    block, server, lookups = _remote_block(
        monkeypatch,
        block_type=block_type,
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
    if block_type is ActionRecognitionModelBlockV2:
        predictions = [result["predictions"] for result in results]
        assert all(result.error_status == "" for result in predictions)
        assert predictions[-1].covered_frame_ranges == ((2, 8), (10, 16))
        timeline = predictions[-1].timeline
    else:
        assert all(result["error_status"] == "" for result in results)
        timeline = results[-1]["timeline"]
    # Window 1 samples source frames 2, 4, 6, 8 and window 2 samples 10..16;
    # sampled indices 1..2 map to source frames 4..6 and 12..14.
    assert [
        (entry.start_frame_idx, entry.end_frame_idx, entry.class_name, entry.class_id)
        for entry in timeline
    ] == [(4, 6, "punch", 0), (12, 14, "punch", 0)]


@pytest.mark.parametrize("source_fps", [30000 / 1001, 24000 / 1001])
def test_ntsc_windows_without_a_frame_budget_still_classify_remotely(
    monkeypatch, source_fps
):
    # A rounded 16 s window at an NTSC rate holds 65 samples about once in 16
    # windows, and the endpoint would split such a clip.
    sampling = VideoSampling(window_seconds=16.0, sample_fps=4.0, min_frames=4)
    block, server, _ = _remote_block(monkeypatch, timeline=[], video_sampling=sampling)
    window_frames = round(16.0 * source_fps)

    results = [
        block.run(
            images=[_video_frame(number, fps=source_fps)],
            model_id="ws/cosmos/1",
        )[0]
        for number in range(20 * window_frames)
    ]

    assert len(server.requests) == 19
    assert all(result["error_status"] == "" for result in results)


@pytest.mark.parametrize(
    "fps", [30000 / 1001, 24000 / 1001, 15000 / 1001, 7.4925, 10 / 3, 7 / 3]
)
@pytest.mark.parametrize("window_seconds", [1.0, 2.0, 4.0])
def test_full_window_at_a_fractional_rate_stays_one_server_window(
    monkeypatch, fps, window_seconds
):
    # The container rounds a fractional rate, often down, which lengthens the
    # clip the server plans; a full window must still be one server window.
    server = _FakeServer(timeline=[])
    sampling = VideoSampling(
        window_seconds=window_seconds,
        sample_fps=fps,
        min_frames=2,
        max_frames=math.ceil(window_seconds * fps),
    )
    _fake_platform(monkeypatch, server.post, class_names=None, video_sampling=sampling)
    model = load_remote_action_recognition_model("ws/m/1", api_key=None)
    frame_count = math.floor(window_seconds * fps + 1e-9)
    if frame_count / fps > window_seconds:
        frame_count -= 1

    assert model.infer(frames=_frames(frame_count), fps=fps) == []
    assert len(server.requests) == 1


def test_window_the_container_rate_would_split_is_refused_before_sending(
    monkeypatch,
):
    server = _FakeServer(timeline=[])
    sampling = VideoSampling(
        window_seconds=2.0, sample_fps=3.0, min_frames=2, max_frames=6
    )
    _fake_platform(monkeypatch, server.post, class_names=None, video_sampling=sampling)
    # A container declaring 2 FPS stretches six frames to 3 s, two windows.
    monkeypatch.setattr(remote, "_read_declared_fps", lambda path: 2.0)
    model = load_remote_action_recognition_model("ws/m/1", api_key=None)

    with pytest.raises(ValueError, match="2 windows"):
        model.infer(frames=_frames(6), fps=3.0)

    assert server.requests == []


def test_chw_uint8_tensor_frames_encode_like_hwc_arrays():
    hwc_frames = _frames(8)
    chw_frames = [torch.from_numpy(frame).permute(2, 0, 1) for frame in hwc_frames]

    assert encode_window(chw_frames, fps=12.0) == encode_window(hwc_frames, fps=12.0)


def test_tensor_block_sends_chw_uint8_frames_remotely(monkeypatch):
    server = _FakeServer(timeline=[])
    _fake_platform(
        monkeypatch, server.post, class_names=["punch"], video_sampling=BLOCK_SAMPLING
    )
    block = TensorActionRecognitionModelBlockV1(
        api_key="key",
        step_execution_mode=StepExecutionMode.REMOTE,
        model_manager=None,
    )

    results = []
    for number in range(9):
        frame = _video_frame(number)
        tensor_frame = WorkflowImageData(
            parent_metadata=frame.parent_metadata,
            workflow_root_ancestor_metadata=frame.workflow_root_ancestor_metadata,
            video_metadata=frame.video_metadata,
            tensor_image=torch.full((3, 24, 32), number, dtype=torch.uint8),
        )
        results.append(block.run(images=[tensor_frame], model_id="ws/punch/1")[0])

    assert len(server.requests) == 1
    assert server.decoded_shapes == [(24, 32, 3)]
    assert all(result["error_status"] == "" for result in results)


def test_background_window_calls_keep_the_step_attribution_and_billing_opt_out(
    monkeypatch,
):
    # The worker thread does not inherit context variables, so without the
    # step's context an internal run would bill every window to the customer.
    block, server, _ = _remote_block(
        monkeypatch, timeline=[], block_type=ActionRecognitionModelBlockV2
    )
    tags = {"source": "workflow", "source_info": "ws/flow"}
    tags_token = usage_source_tags.set(tags)
    secret_token = outbound_service_secret.set("service-secret")
    try:
        for number in range(9):
            block.run(
                images=[_video_frame(number)],
                model_id="ws/punch/1",
                background_inference=True,
            )
        pending = block._video_bookkeeping["clip"].pending
        pending.future.result(timeout=10)
    finally:
        outbound_service_secret.reset(secret_token)
        usage_source_tags.reset(tags_token)
        block.close()

    assert len(server.requests) == 1
    request = server.requests[0]
    assert {key: request["json"][key] for key in tags} == tags
    assert request["params"]["service_secret"] == "service-secret"
    assert request["params"]["countinference"] is False
