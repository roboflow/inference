"""Bounded pipeline examples (development/workflows-2.0/07-bounded-pipeline).

The synthetic scheduling examples run in-process: each forces its
interleaving with probe events and raises on any observation that differs
from its documentation. ``test_real_model_example`` runs the command line
with the pinned trained ResNet-18 weights; it is marked ``slow`` and skipped
when the weights are not present locally.
"""

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
DEMO_DIR = REPOSITORY_ROOT / "development/workflows-2.0/07-bounded-pipeline"
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import bounded_authoring  # noqa: E402
import bounded_scheduling  # noqa: E402
import run_demo  # noqa: E402
from bounded_report import write_report  # noqa: E402
from click.testing import CliRunner  # noqa: E402


def test_passive_timeline_overlaps_phases_only_when_allowed(tmp_path) -> None:
    evidence = bounded_scheduling.passive_timeline()

    overlapping = evidence["phase_overlap=True"]["overlapping_calls"]
    assert ("second:p0", "first:p1") in overlapping
    whole_call = evidence["phase_overlap=False"]
    assert ("second:p0", "first:p1") not in whole_call["overlapping_calls"]
    assert whole_call["counters"]["stages"]["$steps.model#call"]["calls"] == 2
    assert write_report({"timeline": evidence}, destination=tmp_path).is_file()


def test_active_timeline_keeps_per_source_order() -> None:
    evidence = bounded_scheduling.active_timeline()

    order = [item["tag"] for item in evidence["delivery_order"]]
    assert order.index("b1") < order.index("a0")
    assert [tag for tag in order if tag.startswith("a")] == ["a0", "a1", "a2", "a3"]


def test_overload_policies_are_lossless_or_drop_stale_frames() -> None:
    evidence = bounded_scheduling.overload()

    assert evidence["block"]["source_counters"]["dropped"] == 0
    assert evidence["latest"]["delivered"] == ["f0", "f9"]
    assert evidence["latest"]["source_counters"]["dropped"] == 8


def test_lifecycle_examples_balance_counters_and_join_threads() -> None:
    evidence = bounded_scheduling.lifecycle()

    for scenario in ("stop", "cancel", "failure"):
        assert evidence[scenario]["surviving_threads"] == []
    assert evidence["failure"]["failure"] is not None
    assert evidence["passive_failure"]["submission_outcomes"][1:] == [
        "StepExecutionError",
        "PipelineAbortedError",
    ]


def test_authoring_trap_is_wrong_and_both_fixes_are_correct() -> None:
    evidence = bounded_authoring.authoring()

    trap = evidence["trap: scratch on self (intentionally unsafe)"]
    assert trap["pipelined_totals"] == [10.0, 10.0]
    assert trap["serial_totals"] == trap["correct_totals"] == [0.0, 10.0]
    assert trap["pulse_1_prepared_while_pulse_0_in_finish"] is True
    as_output = evidence["fix 1: scratch as phase output"]
    assert as_output["pipelined_totals"] == [0.0, 10.0]
    assert as_output["pulse_1_prepared_while_pulse_0_in_finish"] is True
    whole_call = evidence["fix 2: phase_overlap = False"]
    assert whole_call["pipelined_totals"] == [0.0, 10.0]
    assert whole_call["stages"] == ["$steps.block#call"]
    assert evidence["ordinary counter (run mode)"]["counts"] == [1.0, 2.0, 3.0, 4.0]


def test_runner_lists_authoring_in_all_and_renders_it(tmp_path) -> None:
    assert "authoring" in run_demo.EXAMPLES
    assert not run_demo.EXAMPLES["authoring"].needs_model

    result = CliRunner().invoke(
        run_demo.main, ["--case", "authoring", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    evidence = json.loads((tmp_path / "evidence.json").read_text())
    assert "trap: scratch on self (intentionally unsafe)" in evidence["authoring"]
    page = (tmp_path / "index.html").read_text()
    assert "<h3>fix 2: phase_overlap = False" in page


def _local_weights() -> Path:
    directory = os.environ.get("RESNET18_WEIGHTS_DIR")
    base = (
        Path(directory)
        if directory
        else Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
        / "hub/checkpoints"
    )
    weights = base / "resnet18-f37072fd.pth"

    return weights


@pytest.mark.slow
def test_real_model_example(tmp_path) -> None:
    weights = _local_weights()
    if not weights.is_file():
        pytest.skip("pinned ResNet-18 weights are not present locally")

    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join(
            str(REPOSITORY_ROOT / part)
            for part in (".", "workflows", "inference_models", "stream_vision")
        ),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(DEMO_DIR / "run_demo.py"),
            "--case",
            "model",
            "--weights",
            str(weights),
            "--repeats",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    evidence = json.loads((tmp_path / "evidence.json").read_text())["model"]
    cpu = evidence["passive"]["cpu"]
    assert cpu["images"] == 6
    assert all(record["equal"] for record in cpu["comparisons"])
    assert [frame["top_class"] for frame in evidence["active_cpu"]["frames"][:3]] == [
        "basset",
        "Norfolk terrier",
        "car wheel",
    ]
    assert (tmp_path / "index.html").is_file()
