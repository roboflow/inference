from pathlib import Path
import sys

import pytest


PROCESSOR_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "processor"
)
sys.path.insert(0, str(PROCESSOR_DIR))

from file_replay import build_file_replay_command  # noqa: E402


def test_uploaded_file_replay_is_local_paced_and_low_delay_encoded():
    command = build_file_replay_command(
        ffmpeg_bin="/usr/bin/ffmpeg",
        source_path="/tmp/uploaded.mp4",
        publish_url="rtsp://relay/sim-job",
        bitrate_kbps=2400,
    )

    assert command[:4] == [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    assert command[command.index("-i") + 1] == "/tmp/uploaded.mp4"
    assert command[command.index("-stream_loop") + 1] == "-1"
    assert "-re" in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-tune") + 1] == "zerolatency"
    assert command[command.index("-bf") + 1] == "0"
    assert command[command.index("-b:v") + 1] == "2400k"
    assert command[command.index("-maxrate") + 1] == "2400k"
    assert command[command.index("-bufsize") + 1] == "1200k"
    assert "copy" not in command
    assert command[-1] == "rtsp://relay/sim-job"


def test_uploaded_file_replay_rejects_non_positive_bitrate():
    with pytest.raises(ValueError, match="bitrate must be positive"):
        build_file_replay_command("ffmpeg", "/tmp/source.mp4", "rtsp://relay/job", 0)


def test_processor_downloads_uploaded_stream_before_starting_replay():
    source = (PROCESSOR_DIR / "processor.py").read_text()

    assert "replay_path = self._download_source(source_url)" in source
    assert "source_path=replay_path" in source
    assert "source_path=source_url" not in source
