import sys
from pathlib import Path

import pytest

NETWORKING_DIR = (
    Path(__file__).resolve().parents[3]
    / "development"
    / "video_poc"
    / "benchmarks"
    / "networking"
)
sys.path.insert(0, str(NETWORKING_DIR))

from relay_agent import (  # noqa: E402
    ProgressAccumulator,
    _placement,
    aggregate_progress,
    build_ffmpeg_command,
    redact_text,
    redacted_command,
    render_prometheus,
    resolve_url_from_environment,
)


def test_roles_use_copy_or_decode_as_declared_and_progress_is_enabled():
    publish = build_ffmpeg_command(
        "publish-copy", "ffmpeg", "/fixture.mp4", "rtsp://relay/out"
    )
    read_copy = build_ffmpeg_command("read-copy", "ffmpeg", "rtsp://relay/in")
    read_decode = build_ffmpeg_command("read-decode", "ffmpeg", "rtsp://relay/in")

    assert publish[publish.index("-c") : publish.index("-c") + 2] == ["-c", "copy"]
    assert read_copy[read_copy.index("-c") : read_copy.index("-c") + 2] == [
        "-c",
        "copy",
    ]
    assert "-c" not in read_decode
    for command in (publish, read_copy, read_decode):
        assert command[command.index("-progress") + 1] == "pipe:1"


def test_progress_parser_and_aggregate_report_frames_bytes_and_delivery():
    first = ProgressAccumulator()
    for line in (
        "frame=30\n",
        "fps=15.0\n",
        "bitrate=5000.0kbits/s\n",
        "total_size=625000\n",
        "out_time_us=2000000\n",
        "drop_frames=1\n",
        "dup_frames=2\n",
        "speed=1.0x\n",
        "progress=continue\n",
    ):
        first.feed_line(line, now=100.0)

    summary = first.summary()
    assert summary["frames"] == 30
    assert summary["bytes"] == 625000
    assert summary["mediaDurationSeconds"] == 2.0
    assert summary["reportedBitrateBps"] == 5_000_000
    assert summary["dropFrames"] == 1
    assert summary["duplicateFrames"] == 2
    assert summary["firstMediaAt"] == 100.0

    aggregate = aggregate_progress([{"progress": summary}], measured_seconds=2)
    assert aggregate["deliveredFps"] == 15
    assert aggregate["mediaToWallRatio"] == 1
    assert aggregate["lastReportedBitrateBps"] == 5_000_000


def test_report_and_command_redaction_remove_credentials_path_and_query():
    value = "failed rtsp://user:password@relay.example/private/key?token=abc"
    redacted = redact_text(value)
    assert "user" not in redacted
    assert "password" not in redacted
    assert "private" not in redacted
    assert "abc" not in redacted
    assert "relay.example" in redacted
    assert "[path-redacted]" in redacted
    command = redacted_command(["ffmpeg", value])
    assert "password" not in " ".join(command)


def test_media_url_environment_contract_requires_per_stream_template(monkeypatch):
    monkeypatch.setenv("BENCH_READ_URL", "rtsp://relay.example/shared")
    with pytest.raises(ValueError, match="must contain {stream}"):
        resolve_url_from_environment(
            "BENCH_READ_URL", "stream-1", require_stream_placeholder=True
        )
    monkeypatch.setenv("BENCH_READ_URL", "rtsp://relay.example/{stream}")
    assert (
        resolve_url_from_environment(
            "BENCH_READ_URL", "stream-1", require_stream_placeholder=True
        )
        == "rtsp://relay.example/stream-1"
    )


def test_prometheus_contract_uses_only_bounded_role_location_labels():
    progress = ProgressAccumulator()
    progress.feed_line("frame=12")
    progress.feed_line("progress=continue", now=100)
    payload = render_prometheus(
        {
            "role": "read-copy",
            "location": "processor",
            "running": 1,
            "reconnects": 0,
            "progress": progress,
        }
    )

    assert 'role="read-copy"' in payload
    assert 'location="processor"' in payload
    assert "stream=" not in payload
    assert "run_id=" not in payload
    assert "video_relay_benchmark_agent_frames" in payload


def test_placement_distinguishes_observed_node_from_requested_instance(monkeypatch):
    monkeypatch.setenv("NODE_NAME", "observed-node-1")
    monkeypatch.setenv("REQUESTED_NODE_INSTANCE_TYPE", "c1a.16x")
    monkeypatch.setenv("EXPECTED_CELL_ID", "crusoe-use1")
    monkeypatch.setenv("EXPECTED_CLUSTER_CONTEXT", "ck8s-stg")
    monkeypatch.setenv("MEDIA_PATH", "south-to-east")
    monkeypatch.setenv("MEDIA_PATH_KIND", "explicit-cross-cell")
    placement = _placement()
    assert placement["nodeName"] == "observed-node-1"
    assert placement["requestedNodeInstanceType"] == "c1a.16x"
    assert placement["expectedCell"] == "crusoe-use1"
    assert placement["expectedClusterContext"] == "ck8s-stg"
    assert placement["mediaPath"] == "south-to-east"
    assert placement["mediaPathKind"] == "explicit-cross-cell"
    assert "nodeInstanceType" not in placement
