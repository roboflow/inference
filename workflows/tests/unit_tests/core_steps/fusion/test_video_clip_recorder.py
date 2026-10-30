import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple
from unittest import mock

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.video_clips.capture import (
    PARTIAL_RESOLUTION_CHANGE,
    PARTIAL_SHUTDOWN,
    PARTIAL_STALLED,
    ClipAnchor,
    ClipUnavailableError,
)
from roboflow_workflows.core_steps.fusion.video_clip_recorder import v1
from roboflow_workflows.core_steps.fusion.video_clip_recorder.v1 import (
    AVAILABILITY_OK,
    BACKEND_NVENC,
    BACKEND_X264,
    MAX_TRACKED_VIDEOS,
    UNAVAILABLE_NO_ENCODER,
    UNAVAILABLE_NO_TIMESTAMP,
    UNAVAILABLE_SINKS_DISABLED,
    UNAVAILABLE_UNSUPPORTED_HOST,
    BlockManifest,
    VideoClipRecorderBlockV1,
    serialize_video_clip_anchor,
)
from roboflow_workflows.environment import ENABLE_TENSOR_DATA_REPRESENTATION
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

MANIFEST_TYPE = "roboflow_core/video_clip_recorder@v1"


class FakeRing:
    """Satisfies the `video_clips.ring.Ring` protocol; records every call."""

    def __init__(
        self,
        video_identifier: str,
        *,
        pre_roll_seconds: float,
        playback_speed: float = 1.0,
        clip_fps: Optional[float] = None,
        max_width: Optional[int] = None,
        declared_fps: Optional[float] = None,
        **kwargs: Any,
    ):
        self.video_identifier = video_identifier
        self.kwargs = {
            "pre_roll_seconds": pre_roll_seconds,
            "playback_speed": playback_speed,
            "clip_fps": clip_fps,
            "max_width": max_width,
            "declared_fps": declared_fps,
            **kwargs,
        }
        self.submitted: List[Tuple[datetime, int, WorkflowImageData]] = []
        self.stale: List[datetime] = []
        self.captures: List[Tuple[datetime, datetime]] = []
        self.retired_with: Optional[str] = None
        self.prepare_close_calls: List[Optional[float]] = []
        self.close_calls = 0
        self._closing = False

    def submit(self, frame_timestamp, frame_number, image) -> None:
        self.submitted.append((frame_timestamp, frame_number, image))

    def record_stale(self, frame_timestamp) -> None:
        self.stale.append(frame_timestamp)

    def capture(self, start, end_inclusive):
        if self._closing:
            raise ClipUnavailableError("unavailable:closing")
        self.captures.append((start, end_inclusive))
        return ("pending_capture", start, end_inclusive)

    def retire(self, reason: str) -> None:
        self.retired_with = reason

    def prepare_close(self, deadline_seconds) -> None:
        self._closing = True
        self.prepare_close_calls.append(deadline_seconds)

    def close(self) -> None:
        # Widen the window in which a concurrent close() could double-close.
        time.sleep(0.05)
        self.close_calls += 1

    def stats(self) -> dict:
        return {"frames_submitted": len(self.submitted)}


class RingFactory:
    def __init__(self, ring_class=FakeRing):
        self.ring_class = ring_class
        self.rings: List[FakeRing] = []

    def __call__(self, video_identifier, **kwargs):
        ring = self.ring_class(video_identifier, **kwargs)
        self.rings.append(ring)
        return ring


@pytest.fixture
def rings(monkeypatch) -> RingFactory:
    factory = RingFactory()
    monkeypatch.setattr(v1, "X264Ring", factory)
    monkeypatch.setattr(v1, "_available_nvenc_ring_class", lambda: None)
    return factory


def _image(
    frame_number: int,
    video_id: str = "camera-1",
    frame_timestamp: Optional[datetime] = None,
    shape: Tuple[int, int] = (48, 64),
    comes_from_video_file: Optional[bool] = False,
    as_tensor: Optional[bool] = None,
    video_metadata: Any = "default",
) -> WorkflowImageData:
    if frame_timestamp is None:
        frame_timestamp = datetime.now()
    if video_metadata == "default":
        video_metadata = VideoMetadata(
            video_identifier=video_id,
            frame_number=frame_number,
            frame_timestamp=frame_timestamp,
            fps=30,
            comes_from_video_file=comes_from_video_file,
        )
    if as_tensor is None:
        as_tensor = ENABLE_TENSOR_DATA_REPRESENTATION
    pixels = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if as_tensor:
        return WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id="frame"),
            tensor_image=torch.from_numpy(pixels).permute(2, 0, 1).contiguous(),
            video_metadata=video_metadata,
        )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame"),
        numpy_image=pixels,
        video_metadata=video_metadata,
    )


def _run(block: VideoClipRecorderBlockV1, *images: WorkflowImageData, **settings):
    results = block.run(image=list(images), **settings)
    return [result["anchor"] for result in results]


def _manifest(**fields) -> BlockManifest:
    return BlockManifest.model_validate(
        {"type": MANIFEST_TYPE, "name": "recorder", "image": "$inputs.image", **fields}
    )


# ---- manifest ----------------------------------------------------------------------


def test_manifest_defaults() -> None:
    # when
    manifest = _manifest()

    # then
    assert manifest.label == "Evidence"
    assert manifest.pre_roll_seconds == 5.0
    assert manifest.post_roll_seconds == 5.0
    assert manifest.playback_speed == 1.0
    assert manifest.clip_fps is None
    assert manifest.max_width is None
    assert manifest.max_processing_lag_seconds == 5.0
    assert manifest.get_parameters_accepting_batches() == ["image"]
    assert [o.name for o in manifest.describe_outputs()] == ["anchor"]
    assert manifest.describe_outputs()[0].kind[0].name == "video_clip_anchor"


@pytest.mark.parametrize(
    "fields",
    [
        {"pre_roll_seconds": -0.1},
        {"pre_roll_seconds": 60.5},
        {"post_roll_seconds": -1},
        {"post_roll_seconds": 61},
        {"playback_speed": 0.05},
        {"playback_speed": 33},
        {"clip_fps": 0.5},
        {"clip_fps": 121},
        {"max_width": 62},
        {"max_width": 641},
        {"max_processing_lag_seconds": 0},
        {"max_processing_lag_seconds": float("nan")},
        {"pre_roll_seconds": "$inputs.pre_roll"},
        {"clip_fps": "$inputs.clip_fps"},
    ],
)
def test_manifest_rejects_out_of_range_or_dynamic_settings(fields: dict) -> None:
    with pytest.raises(ValidationError):
        _manifest(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"pre_roll_seconds": 0, "post_roll_seconds": 0},
        {"pre_roll_seconds": 60, "post_roll_seconds": 60},
        {"playback_speed": 0.1},
        {"playback_speed": 32},
        {"clip_fps": 1},
        {"clip_fps": 120},
        {"max_width": 64},
        {"max_width": 1280},
    ],
)
def test_manifest_accepts_boundaries(fields: dict) -> None:
    # when
    manifest = _manifest(**fields)

    # then
    for name, value in fields.items():
        assert getattr(manifest, name) == value


def test_roll_caps_follow_environment(monkeypatch) -> None:
    # given
    monkeypatch.setenv("WORKFLOWS_VIDEO_CLIP_MAX_PRE_ROLL_SECONDS", "120")
    monkeypatch.setenv("WORKFLOWS_VIDEO_CLIP_MAX_POST_ROLL_SECONDS", "10")

    # when
    manifest = _manifest(pre_roll_seconds=120, post_roll_seconds=10)

    # then
    assert manifest.pre_roll_seconds == 120
    with pytest.raises(ValidationError):
        _manifest(pre_roll_seconds=121)
    with pytest.raises(ValidationError):
        _manifest(post_roll_seconds=11)


def test_roll_caps_are_clamped_to_hard_maximum(monkeypatch) -> None:
    # given
    monkeypatch.setenv("WORKFLOWS_VIDEO_CLIP_MAX_PRE_ROLL_SECONDS", "1000")

    # then
    assert v1.max_pre_roll_seconds() == 300.0
    assert _manifest(pre_roll_seconds=300).pre_roll_seconds == 300
    with pytest.raises(ValidationError):
        _manifest(pre_roll_seconds=301)


@pytest.mark.parametrize("raw_value", ["not-a-number", "-5", "nan", ""])
def test_invalid_roll_cap_environment_falls_back_to_default(
    monkeypatch, raw_value: str
) -> None:
    # given
    monkeypatch.setenv("WORKFLOWS_VIDEO_CLIP_MAX_POST_ROLL_SECONDS", raw_value)

    # then
    assert v1.max_post_roll_seconds() == 60.0


@pytest.mark.parametrize("label", ["", "   ", "x" * 257])
def test_manifest_rejects_invalid_labels(label: str) -> None:
    with pytest.raises(ValidationError):
        _manifest(label=label)


def test_manifest_strips_label_and_accepts_256_characters() -> None:
    # then
    assert _manifest(label="  Raw  ").label == "Raw"
    assert _manifest(label="y" * 256).label == "y" * 256


def test_manifest_declares_stateful_video_restrictions() -> None:
    # when
    codes = {r.code for r in BlockManifest.get_restrictions()}

    # then
    assert codes == {
        "stateful_video_state_resets_on_stateless_http",
        "temporal_block_no_benefit_on_still_image",
    }


# ---- recording -----------------------------------------------------------------------


def test_run_submits_each_frame_inside_run_and_emits_bound_anchors(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    timestamps = [datetime.now() + timedelta(seconds=i / 30) for i in range(3)]

    # when
    anchors = _run(
        block,
        *[_image(n + 1, frame_timestamp=ts) for n, ts in enumerate(timestamps)],
        label="Raw",
        pre_roll_seconds=2.0,
        post_roll_seconds=3.0,
        clip_fps=10.0,
        max_width=640,
    )

    # then
    assert len(rings.rings) == 1
    ring = rings.rings[0]
    assert ring.video_identifier == "camera-1"
    assert ring.kwargs["pre_roll_seconds"] == 2.0
    assert ring.kwargs["clip_fps"] == 10.0
    assert ring.kwargs["max_width"] == 640
    assert ring.kwargs["declared_fps"] == 30
    assert [(ts, n) for ts, n, _ in ring.submitted] == [
        (ts, n + 1) for n, ts in enumerate(timestamps)
    ]
    assert all(isinstance(anchor, ClipAnchor) for anchor in anchors)
    assert [a.availability for a in anchors] == [AVAILABILITY_OK] * 3
    assert anchors[1].label == "Raw"
    assert anchors[1].pre_roll_seconds == 2.0
    assert anchors[1].post_roll_seconds == 3.0
    assert anchors[1].recorder_step == "video_clip_recorder"
    assert anchors[1].frame_number == 2


def test_anchor_capture_requests_inclusive_window_from_its_ring(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    timestamp = datetime.now()
    (anchor,) = _run(block, _image(1, frame_timestamp=timestamp))

    # when
    pending = anchor.capture(2.0, 0.0)

    # then
    window = (timestamp - timedelta(seconds=2.0), timestamp)
    assert rings.rings[0].captures == [window]
    assert pending == ("pending_capture", *window)


def test_anchor_uses_step_name_attached_by_compiler(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    block._workflow_step_name = "raw_recorder"

    # when
    (anchor,) = _run(block, _image(1))

    # then
    assert anchor.recorder_step == "raw_recorder"


def test_two_identifiers_get_isolated_rings(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()

    # when
    anchors = _run(
        block,
        _image(1, video_id="a"),
        _image(1, video_id="b"),
        _image(2, video_id="a"),
    )
    anchors_b = _run(block, _image(2, video_id="b"))

    # then
    assert [ring.video_identifier for ring in rings.rings] == ["a", "b"]
    assert [n for _, n, _ in rings.rings[0].submitted] == [1, 2]
    assert [n for _, n, _ in rings.rings[1].submitted] == [1, 2]
    anchors[2].capture(1.0, 1.0)
    anchors_b[0].capture(1.0, 1.0)
    assert len(rings.rings[0].captures) == 1
    assert len(rings.rings[1].captures) == 1


def test_two_block_instances_never_share_rings(rings) -> None:
    # given
    raw_recorder = VideoClipRecorderBlockV1()
    annotated_recorder = VideoClipRecorderBlockV1()

    # when
    _run(raw_recorder, _image(1, video_id="0"))
    _run(annotated_recorder, _image(1, video_id="0"))
    _run(raw_recorder, _image(2, video_id="0"))

    # then
    assert len(rings.rings) == 2
    assert [n for _, n, _ in rings.rings[0].submitted] == [1, 2]
    assert [n for _, n, _ in rings.rings[1].submitted] == [1]


def test_frame_number_going_backwards_resets_ring(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    _run(block, _image(10), _image(11))

    # when
    (anchor,) = _run(block, _image(3))

    # then
    assert len(rings.rings) == 2
    old_ring, new_ring = rings.rings
    assert old_ring.retired_with == PARTIAL_STALLED
    assert [n for _, n, _ in new_ring.submitted] == [3]
    anchor.capture(1.0, 1.0)
    assert old_ring.captures == []
    assert len(new_ring.captures) == 1
    assert block.telemetry()["camera-1"]["resets"] == 1


def test_resolution_change_resets_ring(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    _run(block, _image(1, shape=(48, 64)))

    # when
    _run(block, _image(2, shape=(96, 128)), _image(3, shape=(96, 128)))

    # then
    old_ring, new_ring = rings.rings
    assert old_ring.retired_with == PARTIAL_RESOLUTION_CHANGE
    assert [n for _, n, _ in old_ring.submitted] == [1]
    assert [n for _, n, _ in new_ring.submitted] == [2, 3]


def test_duplicate_frame_number_is_deduplicated(rings) -> None:
    # given - WebRTC runs the workflow twice on frame 1
    block = VideoClipRecorderBlockV1()
    timestamp = datetime.now()

    # when
    first = _run(block, _image(1, frame_timestamp=timestamp))
    second = _run(block, _image(1, frame_timestamp=timestamp))
    third = _run(block, _image(2))

    # then
    assert len(rings.rings) == 1
    assert [n for _, n, _ in rings.rings[0].submitted] == [1, 2]
    assert first[0].availability == second[0].availability == AVAILABILITY_OK
    assert third[0].availability == AVAILABILITY_OK
    second[0].capture(0.0, 0.0)
    assert rings.rings[0].captures == [(timestamp, timestamp)]
    assert block.telemetry()["camera-1"]["duplicates_skipped"] == 1


def test_stale_frame_is_counted_but_not_buffered(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    stale_timestamp = datetime.now() - timedelta(seconds=10)

    # when
    (stale_anchor,) = _run(block, _image(1, frame_timestamp=stale_timestamp))
    (fresh_anchor,) = _run(block, _image(2))

    # then
    ring = rings.rings[0]
    assert ring.stale == [stale_timestamp]
    assert [n for _, n, _ in ring.submitted] == [2]
    assert stale_anchor.availability == AVAILABILITY_OK
    assert fresh_anchor.availability == AVAILABILITY_OK
    assert block.telemetry()["camera-1"]["stale_dropped"] == 1


def test_stale_rule_compares_aware_timestamps_in_utc(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    fresh = datetime.now(timezone.utc)
    stale = fresh - timedelta(seconds=7)

    # when
    _run(block, _image(1, frame_timestamp=stale), max_processing_lag_seconds=5.0)
    _run(block, _image(2, frame_timestamp=fresh), max_processing_lag_seconds=5.0)

    # then
    assert rings.rings[0].stale == [stale]
    assert [n for _, n, _ in rings.rings[0].submitted] == [2]


def test_stale_rule_honours_configured_lag(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    timestamp = datetime.now() - timedelta(seconds=10)

    # when
    _run(block, _image(1, frame_timestamp=timestamp), max_processing_lag_seconds=30)

    # then
    assert rings.rings[0].stale == []
    assert len(rings.rings[0].submitted) == 1


def test_stale_rule_does_not_apply_to_video_files(rings) -> None:
    # given - file timestamps follow the declared fps, not the wall clock
    block = VideoClipRecorderBlockV1()
    timestamp = datetime.now() - timedelta(hours=1)

    # when
    _run(block, _image(1, frame_timestamp=timestamp, comes_from_video_file=True))

    # then
    assert rings.rings[0].stale == []
    assert len(rings.rings[0].submitted) == 1


def test_missing_timestamp_is_unavailable(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    metadata = VideoMetadata.model_construct(
        video_identifier="camera-1",
        frame_number=1,
        frame_timestamp=None,
        fps=30,
        comes_from_video_file=False,
    )

    # when
    (anchor,) = _run(block, _image(1, video_metadata=metadata))

    # then
    assert anchor.availability == UNAVAILABLE_NO_TIMESTAMP
    assert rings.rings == []
    with pytest.raises(ClipUnavailableError):
        anchor.capture(1.0, 1.0)


def test_missing_encoder_is_unavailable_and_never_fails_the_workflow(
    monkeypatch,
) -> None:
    # given
    constructed = []

    def no_encoder(video_identifier, **kwargs):
        constructed.append(video_identifier)
        raise RuntimeError("libx264 not available")

    monkeypatch.setattr(v1, "X264Ring", no_encoder)
    monkeypatch.setattr(v1, "_available_nvenc_ring_class", lambda: None)
    block = VideoClipRecorderBlockV1()

    # when
    anchors = _run(block, _image(1), _image(2))

    # then
    assert [a.availability for a in anchors] == [UNAVAILABLE_NO_ENCODER] * 2
    assert constructed == ["camera-1"], "construction is not retried every frame"
    with pytest.raises(ClipUnavailableError):
        anchors[0].capture(1.0, 1.0)


def test_failing_submit_retires_ring_and_reports_no_encoder(monkeypatch) -> None:
    # given
    class BrokenRing(FakeRing):
        def submit(self, frame_timestamp, frame_number, image) -> None:
            raise RuntimeError("encoder crashed")

    factory = RingFactory(ring_class=BrokenRing)
    monkeypatch.setattr(v1, "X264Ring", factory)
    monkeypatch.setattr(v1, "_available_nvenc_ring_class", lambda: None)
    block = VideoClipRecorderBlockV1()

    # when
    anchors = _run(block, _image(1), _image(2))

    # then
    assert [a.availability for a in anchors] == [UNAVAILABLE_NO_ENCODER] * 2
    assert factory.rings[0].retired_with == PARTIAL_SHUTDOWN
    assert len(factory.rings) == 1


def test_disabled_sinks_idle_the_recorder(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1(disable_sinks=True)

    # when
    (anchor,) = _run(block, _image(1))

    # then
    assert anchor.availability == UNAVAILABLE_SINKS_DISABLED
    assert rings.rings == []
    with pytest.raises(ClipUnavailableError):
        anchor.capture(1.0, 1.0)


def test_still_image_without_video_metadata_is_unsupported_host(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()

    # when
    (anchor,) = _run(block, _image(0, video_metadata=None))

    # then
    assert anchor.availability == UNAVAILABLE_UNSUPPORTED_HOST
    assert rings.rings == []


def test_least_recently_seen_stream_is_retired_beyond_limit(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()

    # when
    for index in range(MAX_TRACKED_VIDEOS + 1):
        _run(block, _image(1, video_id=f"camera-{index}"))

    # then
    assert rings.rings[0].retired_with == PARTIAL_SHUTDOWN
    assert all(ring.retired_with is None for ring in rings.rings[1:])
    assert set(block.telemetry()) == {
        f"camera-{index}" for index in range(1, MAX_TRACKED_VIDEOS + 1)
    }


def test_telemetry_reports_ring_cost_percentiles(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()

    # when
    _run(block, *[_image(n) for n in range(1, 21)])
    telemetry = block.telemetry()["camera-1"]

    # then
    assert telemetry["backend"] == BACKEND_X264
    assert telemetry["frames_submitted"] == 20
    assert telemetry["ring_cost_ms_p50"] is not None
    assert telemetry["ring_cost_ms_p95"] >= telemetry["ring_cost_ms_p50"] >= 0


def test_backend_selection_prefers_nvenc_for_cuda_tensors(monkeypatch) -> None:
    # given
    nvenc_class = object()
    image = mock.MagicMock()
    image.is_tensor_materialised.return_value = True
    image.tensor_image.is_cuda = True
    monkeypatch.setattr(v1, "_available_nvenc_ring_class", lambda: nvenc_class)

    # then
    assert v1._select_backend(image=image) == (BACKEND_NVENC, nvenc_class)


def test_backend_selection_falls_back_to_x264(monkeypatch) -> None:
    # given
    cpu_tensor_image = mock.MagicMock()
    cpu_tensor_image.is_tensor_materialised.return_value = True
    cpu_tensor_image.tensor_image.is_cuda = False
    numpy_image = mock.MagicMock()
    numpy_image.is_tensor_materialised.return_value = False
    cuda_image = mock.MagicMock()
    cuda_image.is_tensor_materialised.return_value = True
    cuda_image.tensor_image.is_cuda = True

    # when
    monkeypatch.setattr(v1, "_available_nvenc_ring_class", lambda: object())
    from_cpu_tensor = v1._select_backend(image=cpu_tensor_image)
    from_numpy = v1._select_backend(image=numpy_image)
    monkeypatch.setattr(v1, "_available_nvenc_ring_class", lambda: None)
    without_nvenc = v1._select_backend(image=cuda_image)

    # then
    assert from_cpu_tensor[0] == BACKEND_X264
    assert from_numpy[0] == BACKEND_X264
    assert without_nvenc[0] == BACKEND_X264
    numpy_image.tensor_image.assert_not_called()


def test_nvenc_import_failure_is_guarded(monkeypatch) -> None:
    # given
    import builtins

    real_import = builtins.__import__

    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.endswith("video_clips") and fromlist and "nvenc_ring" in fromlist:
            raise ImportError("PyNvVideoCodec missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    # then
    assert v1._available_nvenc_ring_class() is None


# ---- lifecycle -----------------------------------------------------------------------


def test_prepare_close_fans_out_once_and_stops_recording(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    _run(block, _image(1, video_id="a"), _image(1, video_id="b"))
    (anchor,) = _run(block, _image(2, video_id="a"))

    # when
    block.prepare_close(4.0)
    block.prepare_close(1.0)
    (late_anchor,) = _run(block, _image(3, video_id="a"))

    # then
    assert [ring.prepare_close_calls for ring in rings.rings] == [[4.0], [4.0]]
    assert [n for _, n, _ in rings.rings[0].submitted] == [1, 2]
    assert late_anchor.availability == UNAVAILABLE_NO_ENCODER
    with pytest.raises(ClipUnavailableError):
        anchor.capture(1.0, 1.0)


def test_close_without_prepare_close_closes_every_ring_including_retired(
    rings,
) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    _run(block, _image(5), _image(1))

    # when
    block.close()
    block.close()

    # then
    assert len(rings.rings) == 2
    assert [ring.prepare_close_calls for ring in rings.rings] == [[None], [None]]
    assert [ring.close_calls for ring in rings.rings] == [1, 1]
    assert block.telemetry() == {}


def test_concurrent_prepare_close_and_close_are_idempotent(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    _run(block, *[_image(1, video_id=f"camera-{i}") for i in range(4)])
    barrier = threading.Barrier(8)
    errors = []

    def closer(index: int) -> None:
        try:
            barrier.wait()
            if index % 2:
                block.prepare_close(2.0)
            block.close()
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=closer, args=(i,)) for i in range(8)]

    # when
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # then
    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert [len(ring.prepare_close_calls) for ring in rings.rings] == [1] * 4
    assert [ring.close_calls for ring in rings.rings] == [1] * 4


def test_close_survives_ring_errors(monkeypatch) -> None:
    # given
    class FailingRing(FakeRing):
        def prepare_close(self, deadline_seconds) -> None:
            raise RuntimeError("prepare failed")

        def close(self) -> None:
            raise RuntimeError("close failed")

    monkeypatch.setattr(v1, "X264Ring", RingFactory(ring_class=FailingRing))
    monkeypatch.setattr(v1, "_available_nvenc_ring_class", lambda: None)
    block = VideoClipRecorderBlockV1()
    _run(block, _image(1))

    # when
    block.prepare_close(1.0)
    block.close()

    # then - no exception escapes; a later close is a no-op
    block.close()


# ---- serializer ----------------------------------------------------------------------


def test_serializer_emits_metadata_only(rings) -> None:
    # given
    block = VideoClipRecorderBlockV1()
    block._workflow_step_name = "raw_recorder"
    timestamp = datetime(2026, 10, 30, 11, 13, 44, 313999)
    (anchor,) = _run(
        block,
        _image(24, video_id="rtsp://camera-1", frame_timestamp=timestamp),
        max_processing_lag_seconds=1e9,
    )

    # when
    serialised = serialize_video_clip_anchor(anchor)

    # then
    assert serialised == {
        "kind": "video_clip_anchor",
        "recorder": "raw_recorder",
        "video_identifier": "rtsp://camera-1",
        "frame_timestamp": "2026-10-30T11:13:44.313999",
        "frame_number": 24,
        "availability": "ok",
    }
    assert serialize_video_clip_anchor(None) is None


def test_serializer_is_registered_for_the_kind() -> None:
    # given
    from roboflow_workflows.core_steps.loader import (
        KINDS_DESERIALIZERS,
        KINDS_SERIALIZERS,
        load_blocks,
        load_kinds,
    )

    # then
    assert KINDS_SERIALIZERS["video_clip_anchor"] is serialize_video_clip_anchor
    assert "video_clip_anchor" not in KINDS_DESERIALIZERS
    assert VideoClipRecorderBlockV1 in load_blocks()
    assert "video_clip_anchor" in {kind.name for kind in load_kinds()}
