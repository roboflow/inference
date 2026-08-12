import json
import re
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from run_api_experiment_matrix import (  # noqa: E402
    build_command,
    load_matrix,
    scenario_run_id,
)
from build_processor_jobs import load_corpus  # noqa: E402
from run_api_workflow_corpus import parse_workload  # noqa: E402

MANIFEST = BENCHMARK_DIR / "workflows" / "manifest.json"
MATRIX_DIR = BENCHMARK_DIR / "matrices"
EXPECTED_STAGING_MATRICES = {
    "cpu-controlled-fps.staging.example.json": 16,
    "gpu-controlled-fps.staging.example.json": 18,
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
