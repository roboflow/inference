from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from roboflow_workflows.core_steps.common.video_clips.artifact import (
    ClipArtifact,
    allocate_clip_path,
)
from roboflow_workflows.core_steps.common.video_clips.report import (
    ClipReport,
    LossCounts,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _report() -> ClipReport:
    return ClipReport(requested_window=(START, START + timedelta(seconds=5)))


def _muxed_file(directory: Path, size: int) -> Path:
    path = allocate_clip_path(directory)
    path.write_bytes(b"\x00" * size)

    return path


def test_small_clip_is_loaded_into_memory_and_file_deleted(tmp_path: Path) -> None:
    # given
    path = _muxed_file(tmp_path, 100)

    # when
    artifact = ClipArtifact.from_muxed_file(
        path, duration_seconds=1.0, report=_report(), inmemory_max_bytes=1000
    )

    # then
    assert artifact.in_memory is True
    assert artifact.size_bytes == 100
    assert artifact.content_type == "video/mp4"
    assert not path.exists()
    with artifact.open_stream() as stream:
        assert stream.read() == b"\x00" * 100


def test_large_clip_stays_on_disk_until_release(tmp_path: Path) -> None:
    # given
    path = _muxed_file(tmp_path, 2000)
    released = []

    # when
    artifact = ClipArtifact.from_muxed_file(
        path,
        duration_seconds=1.0,
        report=_report(),
        inmemory_max_bytes=1000,
        on_release=lambda: released.append(True),
    )

    # then
    assert artifact.in_memory is False
    assert artifact.bytes_or_path == path
    assert path.exists()
    assert artifact.read_bytes() == b"\x00" * 2000

    artifact.release()
    artifact.release()
    assert not path.exists()
    assert released == [True]
    assert artifact.released is True
    with pytest.raises(RuntimeError):
        artifact.read_bytes()


def test_report_serialises_with_iso_timestamps() -> None:
    # given
    report = ClipReport(
        requested_window=(START, START + timedelta(seconds=5)),
        actual_start=START - timedelta(seconds=0.5),
        actual_end=START + timedelta(seconds=5.03),
        lead_seconds=0.5,
        frames=166,
        loss=LossCounts(decimated=3, stale_dropped=1),
        partial="stalled",
    )

    # when
    serialized = report.to_dict()

    # then
    assert serialized["requested_window"] == {
        "start": "2026-01-01T00:00:00+00:00",
        "end": "2026-01-01T00:00:05+00:00",
    }
    assert serialized["actual_start"] == "2025-12-31T23:59:59.500000+00:00"
    assert serialized["loss"] == {
        "decimated": 3,
        "ring_drops": 0,
        "queue_drops": 0,
        "stale_dropped": 1,
        "inference_gaps": 0,
        "pts_clamped": 0,
    }
    assert serialized["partial"] == "stalled"
    assert set(serialized) == {
        "requested_window",
        "actual_start",
        "actual_end",
        "lead_seconds",
        "frames",
        "frame_density_fps",
        "source_fps_declared",
        "playback_speed",
        "clip_fps",
        "width",
        "height",
        "backend",
        "partial",
        "loss",
        "wall_clock_jump_flagged",
        "encode_ms_p50",
        "encode_ms_p95",
        "ring_cost_ms_p50",
        "ring_cost_ms_p95",
        "finalize_ms",
    }
