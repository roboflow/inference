"""Recorder -> real `X264Ring` -> MP4, with synthetic frames (no model, no network)."""

import time
from datetime import datetime, timedelta
from io import BytesIO

import av
import numpy as np
import pytest
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
