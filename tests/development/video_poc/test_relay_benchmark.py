import json
import sys
from pathlib import Path

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from run_relay_benchmark import (  # noqa: E402
    load_config,
    parse_prometheus,
    publisher_command,
    reader_command,
    redacted_command,
)


def test_parse_prometheus_aggregates_series_without_retaining_labels():
    payload = """
# HELP paths active paths
paths{name="src-a",state="ready"} 1
paths{name="src-b",state="ready"} 1
paths_readers{name="src-a",state="ready"} 4
paths_readers{name="src-b",state="ready"} 2
unrelated_metric{workspace="secret"} 99
"""

    assert parse_prometheus(payload, prefixes=("paths",)) == {
        "paths": 2.0,
        "paths_readers": 6.0,
    }


def test_commands_replay_without_encoding_and_redact_all_urls():
    publish = publisher_command(
        "ffmpeg",
        Path("fixture.mp4"),
        "rtsp://user:secret@relay.example/src?token=secret",
    )
    read = reader_command(
        "ffmpeg",
        "rtsp://user:secret@relay.example/src?token=secret",
    )

    assert ["-c", "copy"] == publish[publish.index("-c") : publish.index("-c") + 2]
    assert "-re" in publish
    assert read[-1] == "-"
    rendered = " ".join(redacted_command(publish) + redacted_command(read))
    assert "user" not in rendered
    assert "secret" not in rendered
    assert "[credentials-redacted]" in rendered
    assert "[query-redacted]" in rendered


def test_load_config_resolves_fixture_and_normalizes_scenarios(tmp_path):
    fixture = tmp_path / "fixture.mp4"
    fixture.write_bytes(b"fixture")
    config_path = tmp_path / "matrix.json"
    config_path.write_text(
        json.dumps(
            {
                "fixture": "fixture.mp4",
                "publishUrlTemplate": "rtsp://localhost:8554/{stream}",
                "scenarios": [
                    {
                        "name": "smoke",
                        "sources": 2,
                        "readersPerSource": 3,
                    }
                ],
            }
        )
    )

    config = load_config(config_path)

    assert config["fixture"] == fixture
    assert config["readUrlTemplate"] == "rtsp://localhost:8554/{stream}"
    assert config["scenarios"][0]["sources"] == 2
    assert config["scenarios"][0]["readersPerSource"] == 3
