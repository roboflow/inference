import json
import re
import signal
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from run_api_experiment_matrix import (  # noqa: E402
    _load_resume_summary,
    build_command,
    load_matrix,
    run_matrix,
    scenario_run_id,
)
from build_processor_jobs import load_corpus  # noqa: E402
from run_api_workflow_corpus import parse_workload  # noqa: E402

MANIFEST = BENCHMARK_DIR / "workflows" / "manifest.json"
MATRIX_DIR = BENCHMARK_DIR / "matrices"
EXPECTED_STAGING_MATRICES = {
    "cpu-controlled-fps.staging.example.json": 16,
    "gpu-controlled-fps.staging.example.json": 18,
    "gpu-ingest-gate.staging.example.json": 4,
    "l40s-runtime-capacity.staging.json": 7,
    "long-soak.staging.example.json": 8,
    "multi-workspace-fairness.staging.example.json": 6,
    "output-overhead.staging.example.json": 8,
}
SECRET_FIELD_MARKERS = {
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
}


def write_matrix(tmp_path, **updates):
    document = {
        "schemaVersion": 1,
        "environment": "staging",
        "defaults": {
            "workspace": "benchmark-workspace",
            "sourceId": "source-a",
            "maxPlannedJobs": 16,
        },
        "scenarios": [
            {
                "name": "light-then-heavy",
                "workloads": ["single-detection=4", "instance-segmentation=1@30"],
                "maxFps": 15,
                "recoveryTimeoutSeconds": 180,
            }
        ],
    }
    document.update(updates)
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(document))
    return path


def test_matrix_builds_a_staging_only_redacted_runner_command(tmp_path):
    matrix = load_matrix(write_matrix(tmp_path))
    scenario = matrix["scenarios"][0]
    command = build_command(
        BENCHMARK_DIR / "run_api_workflow_corpus.py",
        matrix,
        scenario,
        "suite-light-then-heavy-r1",
        tmp_path,
        execute=True,
    )

    assert "--execute" in command
    assert command.count("--workload") == 2
    assert "instance-segmentation=1@30" in command
    assert command[command.index("--max-fps") + 1] == "15.0"
    assert command[command.index("--recovery-timeout-seconds") + 1] == "180.0"
    assert command[command.index("--source-id") + 1] == "source-a"
    assert "VIDEO_BENCHMARK_API_KEY" in command
    assert not any(item.startswith("Bearer ") for item in command)


def test_matrix_rejects_production_and_excessive_job_counts(tmp_path):
    path = write_matrix(tmp_path, environment="production")
    with pytest.raises(ValueError, match="staging"):
        load_matrix(path)

    path = write_matrix(tmp_path)
    document = json.loads(path.read_text())
    document["scenarios"][0]["workloads"] = ["single-detection=17"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="safety cap"):
        load_matrix(path)


def test_matrix_bounds_recovery_and_requires_one_startup_fault_job(tmp_path):
    path = write_matrix(tmp_path)
    document = json.loads(path.read_text())
    document["scenarios"][0]["recoveryTimeoutSeconds"] = 3601
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="cannot exceed 3600"):
        load_matrix(path)

    path = write_matrix(tmp_path)
    document = json.loads(path.read_text())
    document["scenarios"][0]["startupFaultReadySeconds"] = 60
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="requires one job"):
        load_matrix(path)


def test_scenario_run_id_is_bounded_for_long_suite_names():
    run_id = scenario_run_id("s" * 64, "light-c24", 2)
    assert len(run_id) <= 64
    assert run_id.endswith("-light-c24-r2")


def test_checked_in_staging_matrices_are_valid_and_credential_free():
    profiles = load_corpus(MANIFEST)
    paths = sorted(MATRIX_DIR.glob("*.json"))

    assert {path.name for path in paths} == set(EXPECTED_STAGING_MATRICES)
    for path in paths:
        document = json.loads(path.read_text())
        matrix = load_matrix(path)

        assert document["environment"] == "staging"
        assert len(matrix["scenarios"]) == EXPECTED_STAGING_MATRICES[path.name]
        assert all(
            scenario["requireSingleProcessor"] for scenario in matrix["scenarios"]
        )
        for scenario in document["scenarios"]:
            for workload in scenario["workloads"]:
                profile = (
                    workload["profile"]
                    if isinstance(workload, dict)
                    else parse_workload(workload)["profile"]
                )
                assert profile in profiles

        def assert_no_inline_secret_fields(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
                    assert key == "apiKeyEnv" or not any(
                        marker in normalized for marker in SECRET_FIELD_MARKERS
                    )
                    assert_no_inline_secret_fields(child)
            elif isinstance(value, list):
                for child in value:
                    assert_no_inline_secret_fields(child)

        assert_no_inline_secret_fields(document)


class FakeProcess:
    next_pid = 100

    def __init__(self, _command, finish_immediately=False):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = 0 if finish_immediately else None
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)

    def wait(self, timeout=None):
        del timeout
        self.returncode = 1 if self.signals else 0
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_matrix_precheckpoints_and_forwards_stop_to_child(tmp_path):
    matrix = load_matrix(write_matrix(tmp_path))
    processes = []

    def spawn(command):
        process = FakeProcess(command)
        processes.append(process)
        return process

    path, summary = run_matrix(
        matrix,
        BENCHMARK_DIR / "run_api_workflow_corpus.py",
        "interruptible-suite",
        tmp_path / "results",
        execute=True,
        popen_factory=spawn,
        stop_requested=lambda: signal.SIGTERM,
        sleep=lambda _seconds: None,
    )

    assert path.exists()
    assert summary["success"] is False
    assert summary["interrupted"] is True
    assert summary["runs"][0]["status"] == "interrupted"
    assert summary["runs"][0]["forwardedSignal"] == "SIGTERM"
    assert processes[0].signals == [signal.SIGTERM]
    persisted = json.loads(path.read_text())
    assert persisted["runs"][0]["pid"] == processes[0].pid


def test_resume_reconciles_completed_child_report_without_restarting(tmp_path):
    matrix = load_matrix(write_matrix(tmp_path))
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    run_id = scenario_run_id("resume-suite", "light-then-heavy", 1)
    summary = {
        "schemaVersion": 2,
        "suiteId": "resume-suite",
        "environment": "staging",
        "matrix": str(matrix["path"]),
        "matrixSha256": matrix["sha256"],
        "startedAt": "start",
        "execute": True,
        "selectedScenarios": [],
        "continueOnError": False,
        "success": False,
        "interrupted": True,
        "runs": [
            {
                "scenario": "light-then-heavy",
                "repetition": 1,
                "runId": run_id,
                "status": "running",
            }
        ],
    }
    (output_dir / f"api-corpus-{run_id}.json").write_text(
        json.dumps(
            {
                "runId": run_id,
                "success": True,
                "endedAt": "end",
                "checkpoint": {"phase": "complete"},
            }
        )
    )

    def forbidden_spawn(_command):
        raise AssertionError("completed child must not restart")

    _path, resumed = run_matrix(
        matrix,
        BENCHMARK_DIR / "run_api_workflow_corpus.py",
        "resume-suite",
        output_dir,
        execute=True,
        popen_factory=forbidden_spawn,
        resume_summary=summary,
        sleep=lambda _seconds: None,
    )

    assert resumed["success"] is True
    assert resumed["interrupted"] is False
    assert resumed["runs"][0]["status"] == "completed"
    assert resumed["runs"][0]["reconciledOnResume"] is True


def test_resume_refuses_incomplete_child_that_requires_exact_cleanup(tmp_path):
    matrix = load_matrix(write_matrix(tmp_path))
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    run_id = scenario_run_id("resume-suite", "light-then-heavy", 1)
    summary = {
        "schemaVersion": 2,
        "suiteId": "resume-suite",
        "environment": "staging",
        "matrix": str(matrix["path"]),
        "matrixSha256": matrix["sha256"],
        "startedAt": "start",
        "execute": True,
        "selectedScenarios": [],
        "continueOnError": False,
        "success": True,
        "runs": [{"runId": run_id, "status": "running"}],
    }

    with pytest.raises(ValueError, match="exact-run cleanup"):
        run_matrix(
            matrix,
            BENCHMARK_DIR / "run_api_workflow_corpus.py",
            "resume-suite",
            output_dir,
            execute=True,
            resume_summary=summary,
            sleep=lambda _seconds: None,
        )


def test_resume_stops_at_a_previously_failed_gate(tmp_path):
    path = write_matrix(tmp_path)
    document = json.loads(path.read_text())
    document["scenarios"].append(
        {
            "name": "later-soak",
            "workloads": ["single-detection=1"],
        }
    )
    path.write_text(json.dumps(document))
    matrix = load_matrix(path)
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    run_id = scenario_run_id("gate-suite", "light-then-heavy", 1)
    summary = {
        "schemaVersion": 2,
        "suiteId": "gate-suite",
        "environment": "staging",
        "matrix": str(matrix["path"]),
        "matrixSha256": matrix["sha256"],
        "startedAt": "start",
        "execute": True,
        "selectedScenarios": [],
        "continueOnError": False,
        "success": False,
        "runs": [
            {
                "scenario": "light-then-heavy",
                "repetition": 1,
                "runId": run_id,
                "status": "completed",
                "returnCode": 1,
            }
        ],
    }
    (output_dir / f"api-corpus-{run_id}.json").write_text(
        json.dumps(
            {
                "runId": run_id,
                "success": False,
                "endedAt": "end",
                "checkpoint": {"phase": "complete"},
            }
        )
    )

    def forbidden_spawn(_command):
        raise AssertionError("a failed gate must stop later scenarios")

    _path, resumed = run_matrix(
        matrix,
        BENCHMARK_DIR / "run_api_workflow_corpus.py",
        "gate-suite",
        output_dir,
        execute=True,
        popen_factory=forbidden_spawn,
        resume_summary=summary,
        sleep=lambda _seconds: None,
    )

    assert resumed["success"] is False
    assert len(resumed["runs"]) == 1


def test_resume_identity_includes_selection_and_continue_policy(tmp_path):
    matrix = load_matrix(write_matrix(tmp_path))
    summary_path = tmp_path / "suite.json"
    summary_path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "suiteId": "suite-a",
                "environment": "staging",
                "matrixSha256": matrix["sha256"],
                "execute": True,
                "selectedScenarios": ["light-then-heavy"],
                "continueOnError": False,
                "runs": [],
            }
        )
    )

    with pytest.raises(ValueError, match="selectedScenarios"):
        _load_resume_summary(
            summary_path, matrix, "suite-a", True, [], False
        )
    with pytest.raises(ValueError, match="continueOnError"):
        _load_resume_summary(
            summary_path,
            matrix,
            "suite-a",
            True,
            ["light-then-heavy"],
            True,
        )
