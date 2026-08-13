import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from development.mmp_staging_benchmark.analyze_curve import analyze_curve
from development.mmp_staging_benchmark.analyze_report import analyze
from development.mmp_staging_benchmark.render_run_spec import (
    current_clean_revision,
    render_point,
)
from development.mmp_staging_benchmark.validate_staging_plan import (
    EXPECTED_MODES,
    load_and_validate,
    render_plan,
)

ROOT = Path(__file__).resolve().parents[3]
MATRIX = ROOT / "development" / "mmp_staging_benchmark" / "matrix.staging.json"


def test_checked_in_matrix_renders_matched_mps_pair():
    matrix = load_and_validate(MATRIX)
    plan = render_plan(matrix, "mmp-capacity-001")

    assert {phase["id"] for phase in plan["phases"]} == set(EXPECTED_MODES)
    no_mps = plan["deployments"]["no-mps"]
    mps = plan["deployments"]["mps"]
    assert (
        no_mps["items"][1]["spec"]["template"]["spec"]["containers"][0]["image"]
        == matrix["artifact"]["image"]
    )
    no_mps_env = {
        item["name"]: item
        for item in no_mps["items"][1]["spec"]["template"]["spec"]["containers"][0][
            "env"
        ]
    }
    mps_env = {
        item["name"]: item
        for item in mps["items"][1]["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    differing = {
        key
        for key in set(no_mps_env) | set(mps_env)
        if no_mps_env.get(key) != mps_env.get(key)
    }
    assert differing == {"NVIDIA_MPS", "MMP_BENCHMARK_MODE", "MMP_BENCHMARK_RUN_ID"}


def test_rejects_missing_phase(tmp_path):
    matrix = json.loads(MATRIX.read_text())
    matrix["phases"].pop()
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="each required phase"):
        load_and_validate(path)


def test_rejects_nonexclusive_gpu(tmp_path):
    matrix = json.loads(MATRIX.read_text())
    matrix["fixed_runtime"]["exclusive_gpu"] = False
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="exclusive L40S"):
        load_and_validate(path)


def test_rejects_fixture_digest_mismatch(tmp_path):
    matrix = json.loads(MATRIX.read_text())
    matrix["fixed_runtime"]["fixture_sha256"] = "0" * 64
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix))
    with pytest.raises(ValueError, match="fixture digest"):
        load_and_validate(path)


def test_render_point_splits_equal_work_between_clients():
    template = (
        ROOT
        / "development"
        / "mmp_staging_benchmark"
        / "spec.same-model-isolated.example.json"
    )
    point = render_point(template, 8, 120, 15)
    assert point["server_url"] == "http://127.0.0.1:18000"
    assert [client["concurrency"] for client in point["clients"]] == [4, 4]


def test_render_point_c1_uses_one_client():
    template = (
        ROOT / "development" / "mmp_staging_benchmark" / "spec.same-model.example.json"
    )
    point = render_point(template, 1, 120, 15)
    assert len(point["clients"]) == 1
    assert point["clients"][0]["concurrency"] == 1


def test_render_point_rejects_odd_two_client_point():
    template = (
        ROOT / "development" / "mmp_staging_benchmark" / "spec.same-model.example.json"
    )
    with pytest.raises(ValueError, match="even total concurrency"):
        render_point(template, 3, 120, 15)


def test_run_spec_renderer_rejects_dirty_harness_revision():
    dirty = subprocess.CompletedProcess([], 0, stdout=" M file.py\n", stderr="")
    with patch("subprocess.run", return_value=dirty):
        with pytest.raises(ValueError, match="must be clean"):
            current_clean_revision(ROOT)


FIXTURE_SHA256 = "83bfa4e706f274ce1da7309cec6374d542f9938b3538481035588681cdaff139"


def passing_report(mps="0"):
    return {
        "experiment": {
            "run_id": "run-001",
            "mode": "mmp-shared-mps" if mps == "1" else "mmp-shared-no-mps",
            "mps_enabled": mps,
            "fixture_sha256": FIXTURE_SHA256,
            "harness_revision": "1" * 40,
            "decoder": "imagecodecs",
            "slots": 128,
            "input_mb_per_slot": 20,
            "batch_max_size": 0,
            "batch_max_wait_ms": 5,
            "server_image_ref": "example@sha256:" + "2" * 64,
            "server_source_revision": "3" * 40,
            "server_pod": "server-pod",
            "server_node": "l40s-node",
        },
        "run": {
            "duration_s": 10,
            "warmup_s": 0,
            "sample_interval_s": 1,
            "pod_name": "server-pod",
            "node_name": "l40s-node",
            "image_sha256": FIXTURE_SHA256,
            "image_ref": "example@sha256:" + "2" * 64,
            "source_revision": "3" * 40,
        },
        "aggregate": {
            "requests": 200,
            "successes": 200,
            "errors": 0,
            "jain_fairness_delivered_fps": 1.0,
        },
        "clients": [
            {
                "client_id": "a",
                "routing_key": "model",
                "concurrency": 1,
                "latency_ms": {"p95": 20},
            },
            {
                "client_id": "b",
                "routing_key": "model",
                "concurrency": 1,
                "latency_ms": {"p95": 21},
            },
        ],
        "metrics_evidence": {
            "sample_count": 10,
            "mmp_pool_full_rejects_delta": 0,
            "model_deltas": {
                "model": {
                    "inference_count_delta": 200,
                    "batch_count_delta": 100,
                    "error_count_delta": 0,
                }
            },
        },
        "metrics_samples": [
            {
                "status": 200,
                "metrics": {"mmp_models": {"model": {"worker_pid": 42}}},
            }
        ],
    }


def passing_identity(report):
    experiment = report["experiment"]
    pod = {
        "metadata": {
            "name": "server-pod",
            "annotations": {
                "roboflow.com/source-revision": experiment["server_source_revision"],
                "roboflow.com/run-id": experiment["run_id"],
                "roboflow.com/mps": (
                    "enabled" if experiment["mps_enabled"] == "1" else "disabled"
                ),
                "roboflow.com/decoder": "imagecodecs",
            },
        },
        "spec": {
            "nodeName": "l40s-node",
            "containers": [
                {
                    "name": "server",
                    "image": experiment["server_image_ref"],
                    "env": [
                        {"name": "NVIDIA_MPS", "value": experiment["mps_enabled"]},
                        {"name": "INFERENCE_DECODER", "value": "imagecodecs"},
                        {"name": "INFERENCE_N_SLOTS", "value": "128"},
                        {"name": "INFERENCE_INPUT_MB", "value": "20"},
                        {"name": "INFERENCE_BATCH_MAX_SIZE", "value": "0"},
                        {"name": "INFERENCE_BATCH_MAX_WAIT_MS", "value": "5"},
                    ],
                }
            ],
        },
        "status": {
            "containerStatuses": [
                {
                    "name": "server",
                    "imageID": "registry/thing@sha256:" + "2" * 64,
                    "restartCount": 0,
                    "ready": True,
                }
            ]
        },
    }
    capability = {
        "checks": {"passed": True},
        "source_revision": experiment["server_source_revision"],
        "image_ref": experiment["server_image_ref"],
        "nvidia_runtime": {
            "gpu_query": {
                "exit_code": 0,
                "stdout": "0, NVIDIA L40S, GPU-123, 570.1, 46068, Default",
            }
        },
    }
    return {
        "pod": pod,
        "capability": capability,
        "expected_node": "l40s-node",
        "expected_gpu_uuid": "GPU-123",
    }


def test_strict_analyzer_accepts_clean_shared_report():
    report = passing_report()
    result = analyze(
        report,
        phase="mmp-shared-no-mps",
        server_log="server interval start run-001\n",
        **passing_identity(report),
    )
    assert result["success"] is True


def test_strict_analyzer_rejects_pid_change_and_cuda_error():
    report = passing_report(mps="1")
    report["metrics_samples"].append(
        {
            "status": 200,
            "metrics": {"mmp_models": {"model": {"worker_pid": 99}}},
        }
    )
    result = analyze(
        report,
        phase="mmp-shared-mps",
        server_log="server interval start\nCUDA error: illegal memory access",
        **passing_identity(report),
    )
    assert result["success"] is False
    assert "model worker PID missing or changed" in result["failures"]
    assert "CUDA failure found in server log" in result["failures"]


def test_strict_analyzer_rejects_workload_identity_mismatch():
    report = passing_report()
    report["experiment"]["mode"] = "mmp-isolated-no-mps"
    report["experiment"]["fixture_sha256"] = "4" * 64
    report["experiment"]["slots"] = 64
    result = analyze(
        report,
        phase="mmp-shared-no-mps",
        server_log="server interval start run-001\n",
        **passing_identity(report),
    )
    assert result["success"] is False
    assert "report phase identity does not match analyzer phase" in result["failures"]
    assert "fixture digest does not match report image" in result["failures"]
    assert "runtime identity mismatch: slots" in result["failures"]


def analyzed_point(point, repetition, success=True):
    return {
        "phase": "mmp-shared-no-mps",
        "success": success,
        "evidence": {
            "total_concurrency": point,
            "run_id": f"shared-c{point:02d}-r{repetition}",
        },
    }


def test_curve_analyzer_certifies_highest_pass_twice_next_fail_twice():
    results = [
        analyzed_point(1, 1),
        analyzed_point(1, 2),
        analyzed_point(2, 1),
        analyzed_point(2, 2),
        analyzed_point(4, 1, False),
        analyzed_point(4, 2, False),
    ]
    result = analyze_curve(
        results,
        phase="mmp-shared-no-mps",
        allowed_points=[1, 2, 4, 8],
    )
    assert result["success"] is True
    assert result["capacity_total_concurrency"] == 2
    assert result["next_failed_total_concurrency"] == 4


def test_curve_analyzer_rejects_single_repetition_and_right_censoring():
    single = analyze_curve(
        [analyzed_point(1, 1), analyzed_point(2, 1, False)],
        phase="mmp-shared-no-mps",
        allowed_points=[1, 2],
    )
    assert single["success"] is False
    assert any("expected 2 repetitions" in item for item in single["failures"])

    censored = analyze_curve(
        [analyzed_point(1, 1), analyzed_point(1, 2)],
        phase="mmp-shared-no-mps",
        allowed_points=[1],
    )
    assert censored["success"] is False
    assert any("right-censored" in item for item in censored["failures"])
