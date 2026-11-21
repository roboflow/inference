"""Recorder -> real `X264Ring` -> MP4, with synthetic frames (no model, no network)."""

import time
from datetime import datetime, timedelta
from io import BytesIO

import av
import numpy as np
import pytest
from roboflow_workflows.core_steps.common.video_clips.capture import (
    ClipUnavailableError,
)
from roboflow_workflows.core_steps.fusion.video_clip_recorder.v1 import (
    AVAILABILITY_OK,
    BACKEND_X264,
    VideoClipRecorderBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

FPS = 30.0
FRAMES = 60


def _pace() -> None:
    # Real-time pacing keeps the bounded ingest queue from overflowing while the
    # encoder opens, which would drop post-roll frames as `queue_drops`.
    time.sleep(1.0 / FPS)


def _frame(index: int, start: datetime, video_id: str = "camera-1"):
    pixels = np.full((120, 160, 3), index * 4 % 255, dtype=np.uint8)
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=index + 1,
        frame_timestamp=start + timedelta(seconds=index / FPS),
        fps=FPS,
        comes_from_video_file=False,
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame"),
        numpy_image=pixels,
        video_metadata=metadata,
    )


def _mp4_frame_count(artifact) -> int:
    payload = artifact.bytes_or_path
    source = BytesIO(payload) if isinstance(payload, bytes) else str(payload)
    with av.open(source) as container:
        frames = sum(1 for _ in container.decode(video=0))
    return frames


@pytest.mark.timeout(60)
def test_capture_resolves_into_mp4_after_post_roll() -> None:
    # given
    block = VideoClipRecorderBlockV1()
    start = datetime.now()
    anchors = []
    pending = None

    try:
        # when
        for index in range(FRAMES):
            (result,) = block.run(
                image=[_frame(index, start)],
                pre_roll_seconds=0.5,
                post_roll_seconds=0.5,
            )
            anchors.append(result["anchor"])
            _pace()
            if index == 30:
                anchor = result["anchor"]
                pending = anchor.capture(
                    anchor.pre_roll_seconds, anchor.post_roll_seconds
                )
        artifact = pending.future.result(timeout=30)

        # then
        assert all(a.availability == AVAILABILITY_OK for a in anchors)
        assert artifact.size_bytes > 0
        assert artifact.content_type == "video/mp4"
        assert artifact.report.backend == BACKEND_X264
        assert artifact.report.partial is None
        assert _mp4_frame_count(artifact) >= 25
        assert block.telemetry()["camera-1"]["ring_cost_ms_p95"] is not None
        artifact.release()
    finally:
        if pending is not None:
            pending.release()
        block.close()


@pytest.mark.timeout(60)
def test_close_finalises_pending_capture_as_partial_shutdown() -> None:
    # given
    block = VideoClipRecorderBlockV1()
    start = datetime.now()
    pending = None

    try:
        for index in range(20):
            (result,) = block.run(
                image=[_frame(index, start)],
                pre_roll_seconds=0.3,
                post_roll_seconds=5.0,
            )
            _pace()
        anchor = result["anchor"]
        pending = anchor.capture(anchor.pre_roll_seconds, anchor.post_roll_seconds)

        # when - the post-roll never arrives
        block.prepare_close(5.0)
        block.close()
        block.close()

        # then
        artifact = pending.future.result(timeout=30)
        assert artifact.report.partial == "shutdown"
        assert artifact.size_bytes > 0
        artifact.release()
    finally:
        if pending is not None:
            pending.release()
        block.close()


@pytest.mark.timeout(60)
def test_close_without_prepare_close_finalises_and_releases() -> None:
    # given: the HTTP / batch path calls `close()` with no `prepare_close()`
    block = VideoClipRecorderBlockV1()
    start = datetime.now()
    pending = None

    try:
        for index in range(20):
            (result,) = block.run(
                image=[_frame(index, start)],
                pre_roll_seconds=0.3,
                post_roll_seconds=5.0,
            )
            _pace()
        anchor = result["anchor"]
        pending = anchor.capture(anchor.pre_roll_seconds, anchor.post_roll_seconds)
        ring = block._streams["camera-1"].ring

        # when
        block.close()

        # then: the capture is finalised from the ring and the ring holds nothing
        artifact = pending.future.result(timeout=30)
        assert artifact.report.partial == "shutdown"
        assert _mp4_frame_count(artifact) > 0
        stats = ring.stats()
        assert stats.bytes == 0
        assert stats.held_bytes == 0
        assert stats.pending_captures == 0
        assert block._streams == {}
        with pytest.raises(ClipUnavailableError):
            anchor.capture(anchor.pre_roll_seconds, anchor.post_roll_seconds)
        artifact.release()
    finally:
        if pending is not None:
            pending.release()
        block.close()


def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        time.sleep(0.01)


@pytest.mark.timeout(60)
def test_retired_rings_release_themselves_before_close() -> None:
    # given: five frame-number resets, the second with a capture still pending
    block = VideoClipRecorderBlockV1()
    retired = []
    pending = None

    try:
        for round_index in range(6):
            start = datetime.now()
            for index in range(10):
                (result,) = block.run(
                    image=[_frame(index, start)],
                    pre_roll_seconds=0.3,
                    post_roll_seconds=5.0,
                )
                _pace()
            if round_index == 1:
                anchor = result["anchor"]
                pending = anchor.capture(
                    anchor.pre_roll_seconds, anchor.post_roll_seconds
                )
            if round_index < 5:
                retired.append(block._streams["camera-1"].ring)

        # when: the pending capture of a retired ring is finalised
        artifact = pending.future.result(timeout=30)
        artifact.release()
        # a frame-number restart is reported as such, not as a stall
        assert artifact.report.partial == "reset"

        # then: every retired ring left the block, the budget and its threads
        _wait_for(lambda: block._retired_rings == [])
        assert len(retired) == 5
        for ring in retired:
            _wait_for(lambda: not ring._encoder_thread.is_alive())
            if ring._finalizer_thread is not None:
                _wait_for(lambda: not ring._finalizer_thread.is_alive())
            assert ring.budget_bytes() == 0
            assert ring.stats().held_bytes == 0
            assert ring not in ring._budget._sources
        live_ring = block._streams["camera-1"].ring
        assert live_ring not in retired
        assert live_ring in live_ring._budget._sources
    finally:
        if pending is not None:
            pending.release()
        block.close()
