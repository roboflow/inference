import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from development.mmp_staging_benchmark.analyze_curve import (
    analyze_curve,
    compare_mps_pair,
)
from development.mmp_staging_benchmark.analyze_report import analyze
from development.mmp_staging_benchmark.capture_cache_evidence import (
    capture as capture_cache,
)
from development.mmp_staging_benchmark.capture_mps_evidence import capture
from development.mmp_staging_benchmark.render_run_spec import (
    current_clean_revision,
    render_point,
)
from development.mmp_staging_benchmark.run_concurrent_clients import ClientSpec
from development.mmp_staging_benchmark.validate_staging_plan import (
    EXPECTED_MODES,
    canonical_sha256,
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


def test_mps_evidence_capture_binds_staging_pod_and_gpu():
    responses = iter(
        [
            "ck8s-stg",
            json.dumps({"metadata": {"name": "server-pod", "uid": "pod-uid-123"}}),
            "GPU-123",
            "101\n202",
            "42",
            "",
        ]
    )
    with patch(
        "development.mmp_staging_benchmark.capture_mps_evidence._run",
        side_effect=lambda *_args, **_kwargs: next(responses),
    ), patch(
        "development.mmp_staging_benchmark.capture_mps_evidence.time.time",
        return_value=1060,
    ):
        evidence = capture("server-pod")
    assert evidence == {
        "schema_version": 1,
        "context": "ck8s-stg",
        "namespace": "video-proc-bench-mmp",
        "pod": "server-pod",
        "pod_uid": "pod-uid-123",
        "gpu_uuid": "GPU-123",
        "captured_unix_s": 1060,
        "command": "get_server_list",
        "exit_code": 0,
        "server_list": "101\n202",
        "clients_by_server": {"101": [42], "202": []},
    }


def test_cache_evidence_capture_records_routes_without_api_key():
    responses = iter(
        [
            "ck8s-stg",
            json.dumps({"metadata": {"name": "server-pod", "uid": "pod-uid-123"}}),
            "/models/cache/a\n/models/cache/b",
            "",
        ]
    )
    metrics = {
        "mmp_models": {
            "yolov8n-640": {
                "worker_pid": 42,
                "inference_count": 10,
                "batch_count": 5,
                "error_count": 0,
            }
        }
    }
    with patch(
        "development.mmp_staging_benchmark.capture_cache_evidence._run",
        side_effect=lambda *_args, **_kwargs: next(responses),
    ), patch(
        "development.mmp_staging_benchmark.capture_cache_evidence._metrics",
        return_value=metrics,
    ) as metrics_call, patch(
        "development.mmp_staging_benchmark.capture_cache_evidence.time.time",
        return_value=990,
    ):
        evidence = capture_cache("server-pod", "http://127.0.0.1:18000", "secret")
    metrics_call.assert_called_once_with("http://127.0.0.1:18000", "secret")
    assert "secret" not in json.dumps(evidence)
    assert evidence["pod_uid"] == "pod-uid-123"
    assert evidence["cache_file_count"] == 2
    assert set(evidence["routes"]) == {"yolov8n-640"}


FIXTURE_SHA256 = "83bfa4e706f274ce1da7309cec6374d542f9938b3538481035588681cdaff139"


def passing_report(mps="0"):
    matrix = load_and_validate(MATRIX)
    phase = "mmp-shared-mps" if mps == "1" else "mmp-shared-no-mps"
    phase_config = next(item for item in matrix["phases"] if item["id"] == phase)
    template = MATRIX.parent / phase_config["spec"]
    workload = render_point(template, 2, 120, 15)
    workload_clients = [asdict(ClientSpec(**item)) for item in workload["clients"]]
    samples = []
    for index in range(135):
        samples.append(
            {
                "offset_s": index,
                "status": 200,
                "metrics": {
                    "mmp_rejects_pool_full": 0,
                    "mmp_models": {
                        "yolov8n-640": {
                            "worker_pid": 42,
                            "inference_count": 2 * index,
                            "batch_count": index,
                            "error_count": 0,
                        }
                    },
                },
            }
        )
    return {
        "experiment": {
            "run_id": "run-001",
            "mode": phase,
            "mps_enabled": mps,
            "fixture_sha256": FIXTURE_SHA256,
            "harness_revision": "1" * 40,
            "decoder": "imagecodecs",
            "slots": 128,
            "input_mb_per_slot": 20,
            "batch_max_size": 0,
            "batch_max_wait_ms": 5,
            "server_image_ref": matrix["artifact"]["image"],
            "server_source_revision": matrix["artifact"]["source_revision"],
            "server_pod": "server-pod",
            "server_node": "l40s-node",
            "matrix_sha256": canonical_sha256(matrix),
            "template_sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
            "workload_sha256": canonical_sha256(workload),
            "cache_state": "warm",
        },
        "run": {
            "server_url": "http://127.0.0.1:18000",
            "duration_s": 120,
            "warmup_s": 15,
            "sample_interval_s": 1,
            "request_timeout_s": 60.0,
            "max_latency_samples_per_client": 250_000,
            "started_unix_s": 1000,
            "finished_unix_s": 1135,
            "pod_name": "server-pod",
            "node_name": "l40s-node",
            "image_sha256": FIXTURE_SHA256,
            "image_ref": matrix["artifact"]["image"],
            "source_revision": matrix["artifact"]["source_revision"],
            "clients": workload_clients,
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
                "routing_key": "yolov8n-640",
                "concurrency": 1,
                "latency_ms": {"p95": 20},
            },
            {
                "client_id": "b",
                "routing_key": "yolov8n-640",
                "concurrency": 1,
                "latency_ms": {"p95": 21},
            },
        ],
        "metrics_evidence": {
            "sample_count": 135,
            "mmp_pool_full_rejects_delta": 0,
            "model_deltas": {
                "yolov8n-640": {
                    "inference_count_delta": 268,
                    "batch_count_delta": 134,
                    "error_count_delta": 0,
                }
            },
        },
        "metrics_samples": samples,
    }


def passing_identity(report):
    experiment = report["experiment"]
    pod = {
        "metadata": {
            "name": "server-pod",
            "namespace": "video-proc-bench-mmp",
            "uid": "pod-uid-123",
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
                    "imageID": "registry/thing@"
                    + experiment["server_image_ref"].split("@", 1)[1],
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
    identity = {
        "pod": pod,
        "capability": capability,
        "expected_node": "l40s-node",
        "expected_gpu_uuid": "GPU-123",
        "matrix": load_and_validate(MATRIX),
        "matrix_dir": MATRIX.parent,
        "expected_harness_revision": "1" * 40,
        "cache_evidence": {
            "schema_version": 1,
            "context": "ck8s-stg",
            "namespace": "video-proc-bench-mmp",
            "pod": "server-pod",
            "pod_uid": "pod-uid-123",
            "captured_unix_s": 990,
            "cache_file_count": 1,
            "cache_paths_sha256": "5" * 64,
            "routes": {
                "yolov8n-640": {
                    "worker_pid": 42,
                    "inference_count": 10,
                    "batch_count": 5,
                    "error_count": 0,
                }
            },
        },
    }
    if experiment["mps_enabled"] == "1":
        identity["mps_evidence"] = {
            "context": "ck8s-stg",
            "namespace": "video-proc-bench-mmp",
            "captured_unix_s": 1060,
            "pod": "server-pod",
            "pod_uid": "pod-uid-123",
            "gpu_uuid": "GPU-123",
            "command": "get_server_list",
            "exit_code": 0,
            "server_list": "12345",
            "clients_by_server": {"12345": [42]},
        }
    return identity


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
            "offset_s": 135,
            "status": 200,
            "metrics": {
                "mmp_rejects_pool_full": 0,
                "mmp_models": {
                    "yolov8n-640": {
                        "worker_pid": 99,
                        "inference_count": 270,
                        "batch_count": 135,
                        "error_count": 0,
                    }
                },
            },
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


def test_strict_analyzer_recomputes_raw_metrics_and_matrix_workload():
    report = passing_report()
    report["metrics_samples"] = report["metrics_samples"][:1]
    report["run"]["clients"][0]["model_id"] = "wrong-model"
    result = analyze(
        report,
        phase="mmp-shared-no-mps",
        server_log="server interval start run-001\n",
        **passing_identity(report),
    )
    assert result["success"] is False
    assert "raw metrics coverage failed" in result["failures"]
    assert "raw metrics interval span failed" in result["failures"]
    assert "report clients differ from matrix workload" in result["failures"]


def test_strict_analyzer_rejects_counter_reset_and_sparse_route_metrics():
    report = passing_report()
    report["metrics_samples"][100]["metrics"]["mmp_models"]["yolov8n-640"][
        "inference_count"
    ] = 1
    for sample in report["metrics_samples"][:20]:
        sample["metrics"]["mmp_models"] = {}
    result = analyze(
        report,
        phase="mmp-shared-no-mps",
        server_log="server interval start run-001\n",
        **passing_identity(report),
    )
    assert result["success"] is False
    assert any(
        "raw model metric coverage/continuity failed" in failure
        for failure in result["failures"]
    )


def test_strict_analyzer_rejects_self_attested_cold_cache_and_warmup_mps():
    report = passing_report(mps="1")
    report["experiment"]["cache_state"] = "cold"
    identity = passing_identity(report)
    identity["mps_evidence"]["captured_unix_s"] = 1005
    result = analyze(
        report,
        phase="mmp-shared-mps",
        server_log="server interval start run-001\n",
        **identity,
    )
    assert result["success"] is False
    assert "cold run did not start with empty routes/cache" in result["failures"]
    assert "MPS evidence was not captured during the run" in result["failures"]


def test_strict_analyzer_rejects_mps_server_without_measured_worker_client():
    report = passing_report(mps="1")
    identity = passing_identity(report)
    identity["mps_evidence"]["clients_by_server"] = {"12345": [99999]}
    result = analyze(
        report,
        phase="mmp-shared-mps",
        server_log="server interval start run-001\n",
        **identity,
    )
    assert result["success"] is False
    assert "measured model workers are not all MPS clients" in result["failures"]


def test_strict_analyzer_accepts_operationally_cold_pre_run_state():
    report = passing_report()
    report["experiment"]["cache_state"] = "cold"
    identity = passing_identity(report)
    identity["cache_evidence"].update(
        {
            "cache_file_count": 0,
            "cache_paths_sha256": hashlib.sha256(b"").hexdigest(),
            "routes": {},
        }
    )
    result = analyze(
        report,
        phase="mmp-shared-no-mps",
        server_log="server interval start run-001\n",
        **identity,
    )
    assert result["success"] is True


def test_strict_analyzer_accepts_cold_route_created_after_first_sample():
    report = passing_report()
    report["experiment"]["cache_state"] = "cold"
    report["metrics_samples"][0]["metrics"]["mmp_models"] = {}
    report["metrics_evidence"]["model_deltas"]["yolov8n-640"] = {
        "inference_count_delta": 268,
        "batch_count_delta": 134,
        "error_count_delta": 0,
    }
    identity = passing_identity(report)
    identity["cache_evidence"].update(
        {
            "cache_file_count": 0,
            "cache_paths_sha256": hashlib.sha256(b"").hexdigest(),
            "routes": {},
        }
    )
    result = analyze(
        report,
        phase="mmp-shared-no-mps",
        server_log="server interval start run-001\n",
        **identity,
    )
    assert result["success"] is True


def test_strict_analyzer_rejects_warm_worker_pid_mismatch():
    report = passing_report()
    identity = passing_identity(report)
    identity["cache_evidence"]["routes"]["yolov8n-640"]["worker_pid"] = 99999
    result = analyze(
        report,
        phase="mmp-shared-no-mps",
        server_log="server interval start run-001\n",
        **identity,
    )
    assert result["success"] is False
    assert any("warm pre-run worker PID differs" in item for item in result["failures"])


def analyzed_point(point, repetition, success=True):
    return {
        "phase": "mmp-shared-no-mps",
        "success": success,
        "failures": [] if success else ["a: latency p95 failed"],
        "evidence": {
            "total_concurrency": point,
            "run_id": f"shared-c{point:02d}-r{repetition}",
            "server_image_ref": "image@sha256:" + "1" * 64,
            "server_source_revision": "2" * 40,
            "harness_revision": "3" * 40,
            "fixture_sha256": "4" * 64,
            "matrix_sha256": "5" * 64,
            "template_sha256": "6" * 64,
            "workload_sha256": f"{point:064x}",
            "cache_state": "warm",
            "server_node": "l40s-node",
            "gpu_uuid": "GPU-123",
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


def test_curve_analyzer_rejects_cross_cohort_results():
    results = [
        analyzed_point(1, 1),
        analyzed_point(1, 2),
        analyzed_point(2, 1, False),
        analyzed_point(2, 2, False),
    ]
    results[-1]["evidence"]["gpu_uuid"] = "GPU-other"
    result = analyze_curve(
        results,
        phase="mmp-shared-no-mps",
        allowed_points=[1, 2],
    )
    assert result["success"] is False
    assert "single-report results do not form one exact cohort" in result["failures"]


def test_curve_analyzer_requires_exact_matched_mps_pair():
    no_mps = analyze_curve(
        [
            analyzed_point(1, 1),
            analyzed_point(1, 2),
            analyzed_point(2, 1, False),
            analyzed_point(2, 2, False),
        ],
        phase="mmp-shared-no-mps",
        allowed_points=[1, 2],
    )
    mps_results = [
        {**analyzed_point(1, 1), "phase": "mmp-shared-mps"},
        {**analyzed_point(1, 2), "phase": "mmp-shared-mps"},
        {**analyzed_point(2, 1, False), "phase": "mmp-shared-mps"},
        {**analyzed_point(2, 2, False), "phase": "mmp-shared-mps"},
    ]
    mps = analyze_curve(
        mps_results,
        phase="mmp-shared-mps",
        allowed_points=[1, 2],
    )
    assert compare_mps_pair(no_mps, mps)["success"] is True
    mps["cohort"]["gpu_uuid"] = "GPU-other"
    assert compare_mps_pair(no_mps, mps)["success"] is False


def test_curve_analyzer_rejects_disjoint_or_identity_only_failures():
    results = [
        analyzed_point(1, 1),
        analyzed_point(1, 2),
        analyzed_point(2, 1, False),
        analyzed_point(2, 2, False),
    ]
    results[-2]["failures"].append("CUDA failure found in server log")
    result = analyze_curve(
        results,
        phase="mmp-shared-no-mps",
        allowed_points=[1, 2],
    )
    assert result["success"] is False
    assert any("non-capacity failure" in item for item in result["failures"])
