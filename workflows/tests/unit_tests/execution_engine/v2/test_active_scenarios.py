"""Exercise the runnable source examples through the public host entry point."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "development" / "workflows-2.0").is_dir()
)
DEMO = REPOSITORY_ROOT / "development/workflows-2.0/04-source-lifecycle/run_demo.py"


@pytest.fixture(scope="module")
def scenario_evidence(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("active-scenarios")
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        MPLCONFIGDIR=str(output_dir / "matplotlib"),
        PYTHONPATH=os.pathsep.join(
            str(REPOSITORY_ROOT / part)
            for part in (".", "workflows", "inference_models", "stream_vision")
        ),
    )
    completed = subprocess.run(
        [sys.executable, "-B", str(DEMO), "--output-dir", str(output_dir)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    evidence = json.loads((output_dir / "evidence.json").read_text())

    return evidence


def test_reader_independence_and_early_eof(scenario_evidence):
    evidence = scenario_evidence["lifecycle"]
    records = evidence["records"]
    frames = [record for record in records if record["group"] == "frames"]
    temperatures = [record for record in records if record["group"] == "temperatures"]

    assert len(frames) == 5
    assert len(temperatures) == 3
    assert records[0]["group"] == "frames"
    assert evidence["lifecycle"].index("temperature:eof") < evidence["lifecycle"].index(
        "frames:3"
    )
    assert [record["fields"]["original"]["pts"]["ticks"] for record in frames] == [
        0,
        40,
        80,
        120,
        160,
    ]
    assert [
        record["fields"]["fahrenheit"]["pts"]["ticks"] for record in temperatures
    ] == [10, 110, 210]
    assert frames[0]["lineage"] != temperatures[0]["lineage"]
    assert frames[0]["pulse"] == temperatures[0]["pulse"] == 0


def test_nested_gate_metadata_and_shared_work(scenario_evidence):
    evidence = scenario_evidence["lifecycle"]
    records = evidence["records"]
    frames = [record for record in records if record["group"] == "frames"]
    temperatures = [record for record in records if record["group"] == "temperatures"]
    audits = [record for record in records if record["group"] == "temperature_audit"]

    assert evidence["audit"] == [32.0, 68.0, 212.0]
    assert [record["fields"]["count"]["value"] for record in temperatures] == [1, 2, 3]
    assert [record["fields"] for record in temperatures] == [
        record["fields"] for record in audits
    ]
    assert [record["fields"]["forwarded"]["status"] for record in frames] == [
        "complete",
        "filtered",
        "complete",
        "filtered",
        "complete",
    ]
    for index in (0, 2, 4):
        original = frames[index]["fields"]["original"]
        resized = frames[index]["fields"]["resized"]
        assert resized["source"] == "camera"
        assert resized["pts"] == original["pts"]
        assert resized["observed"] == original["observed"]
        assert resized["pts"]["clock"] != resized["observed"]["clock"]
        assert resized["axis_ids"] == []
        assert resized["value"]["root"]["frame_id"] == f"frame-{index}"
        assert resized["value"]["tensor_shape"] == [3, 6, 8]


def test_stop_drains_admitted_work_and_failure_closes_sources(scenario_evidence):
    stopped = scenario_evidence["stop"]
    assert stopped["counters"]["camera"]["admitted"] == 2
    assert stopped["counters"]["temperature"]["admitted"] == 0
    assert [record["pulse"] for record in stopped["records"]] == [0, 1]
    for counts in stopped["counters"].values():
        assert counts["admitted"] == counts["processed"]
        assert counts["cancelled"] == 0
    for case in ("stop", "failure"):
        lifecycle = scenario_evidence[case]["lifecycle"]
        assert lifecycle.count("camera:close") == 1
        assert lifecycle.count("temperature:close") == 1
    failed = scenario_evidence["failure"]
    assert failed["error"]["type"] == "ActiveRunError"
    assert "frames" in failed["error"]["message"]
    assert "intentional frames handler failure" in failed["error"]["message"]


def test_passive_timestamp_and_independent_join_rejection(scenario_evidence):
    passive = scenario_evidence["passive"]
    assert passive["timestamped_rows"] == [{"fahrenheit": 68.0, "count": 1}]
    assert passive["unwrapped_rows"] == [{"fahrenheit": 32.0, "count": 2}]
    assert passive["source"] == "thermometer"
    assert passive["pts_ticks"] == 110
    assert passive["axis_ids"] == []
    rejection = scenario_evidence["invalid-join"]
    assert rejection["rejected"]
    assert "camera" in rejection["message"] and "temperature" in rejection["message"]


def test_plain_csv_source_needs_no_probe_resource(scenario_evidence):
    records = scenario_evidence["csv"]["records"]
    assert [record["fields"]["fahrenheit"]["value"] for record in records] == [
        32.0,
        68.0,
        212.0,
    ]
    assert [record["pulse"] for record in records] == [0, 1, 2]


def test_plain_csv_handler_can_request_stop_without_a_startup_barrier(
    scenario_evidence,
):
    evidence = scenario_evidence["csv-stop"]
    records = evidence["records"]
    counters = evidence["counters"]

    assert evidence["stop_requested"]
    assert records
    assert [record["pulse"] for record in records] == list(range(len(records)))
    assert [record["fields"]["fahrenheit"]["value"] for record in records] == [
        32.0,
        68.0,
        212.0,
    ][: len(records)]
    assert (
        counters["admitted"]
        == counters["processed"]
        == counters["delivered"]
        == len(records)
    )
    assert counters["cancelled"] == 0
    assert counters["closed"]
