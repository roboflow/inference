import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = (
    Path(__file__).resolve().parents[3] / "development" / "video_poc" / "benchmarks"
)
sys.path.insert(0, str(BENCHMARK_DIR))

from analysis.fairness import FairnessConfig, analyze_fairness  # noqa: E402
from run_api_multi_workspace_corpus import attach_fairness_analysis  # noqa: E402


def _job(job_id, ordinal, label, frames, processor="processor-shared", latency=20):
    return {
        "id": job_id,
        "ordinal": ordinal,
        "workspaceLabel": label,
        "profile": "single-detection",
        "state": "running",
        "processorId": processor,
        "stats": {
            "frames": frames,
            "decodeToResultLatencyMs": latency,
            "decodeToResultLatency": {
                "count": frames,
                "sum": frames * latency,
                "max": latency,
                "histogram": {
                    "bounds": [50, 100, None],
                    "cumulativeCounts": [frames, frames, frames],
                },
            },
        },
    }


def _report(a_measurement_rate=10, b_measurement_rate=10, b_processor=None):
    b_processor = b_processor or "processor-shared"
    samples = [
        {
            "phase": "baseline",
            "elapsedSeconds": 0,
            "jobs": [_job("a", 1, "tenant-a", 0)],
        },
        {
            "phase": "baseline",
            "elapsedSeconds": 10,
            "jobs": [_job("a", 1, "tenant-a", 100)],
        },
    ]
    for elapsed, step in ((20, 0), (30, 1), (40, 2), (50, 3)):
        samples.append(
            {
                "phase": "measurement",
                "elapsedSeconds": elapsed,
                "jobs": [
                    _job("a", 1, "tenant-a", 100 + step * a_measurement_rate * 10),
                    _job(
                        "b",
                        2,
                        "tenant-b",
                        step * b_measurement_rate * 10,
                        processor=b_processor,
                    ),
                ],
            }
        )
    return {
        "schemaVersion": 1,
        "kind": "multi-workspace-api-corpus",
        "runId": "fairness-test",
        "scenarioName": "incumbent-arrival",
        "plannedConcurrency": 2,
        "success": True,
        "errors": [],
        "workloads": [
            {
                "ordinal": 1,
                "workspaceLabel": "tenant-a",
                "profile": "single-detection",
                "maxFps": 10,
                "startAfterSeconds": 0,
            },
            {
                "ordinal": 2,
                "workspaceLabel": "tenant-b",
                "profile": "single-detection",
                "maxFps": 10,
                "startAfterSeconds": 10,
            },
        ],
        "starts": [
            {"job": _job("a", 1, "tenant-a", 0)},
            {"job": _job("b", 2, "tenant-b", 0, processor=b_processor)},
        ],
        "samples": samples,
    }


def _config(require_shared=True):
    return FairnessConfig(
        warmup_seconds=0,
        min_steady_intervals=2,
        require_shared_processor=require_shared,
    )


def test_balanced_two_tenant_run_passes_and_reports_incumbent_retention():
    result = analyze_fairness(_report(), _config())

    assert result["success"] is True
    assert result["crossTenant"]["targetAttainmentJainIndex"] == 1.0
    assert result["crossTenant"]["targetAttainmentSpreadRatio"] == 0.0
    assert result["crossTenant"]["allTenantsShareOneProcessor"] is True
    tenants = {item["workspaceLabel"]: item for item in result["tenants"]}
    assert tenants["tenant-a"]["incumbentRetentionRatio"] == 1.0
    assert tenants["tenant-b"]["incumbentRetentionRatio"] is None


def test_noisy_neighbor_regression_fails_incumbent_retention_and_fairness():
    result = analyze_fairness(_report(a_measurement_rate=5), _config())

    assert result["success"] is False
    assert result["checks"]["incumbentRetention"] is False
    assert result["checks"]["tenantAttainmentSpread"] is False
    assert result["crossTenant"]["targetAttainmentJainIndex"] < 1


def test_missing_max_fps_propagation_is_detected_as_target_overshoot():
    result = analyze_fairness(
        _report(a_measurement_rate=60, b_measurement_rate=60), _config()
    )

    assert result["success"] is False
    assert result["checks"]["targetFpsPropagated"] is False
    assert {item["targetAttainmentRatio"] for item in result["tenants"]} == {6.0}


def test_tenant_aggregate_cannot_hide_one_overshooting_and_one_starved_job():
    report = _report()
    report["workloads"].append(
        {
            "ordinal": 3,
            "workspaceLabel": "tenant-a",
            "profile": "single-detection",
            "maxFps": 10,
            "startAfterSeconds": 0,
        }
    )
    report["plannedConcurrency"] = 3
    report["starts"].append({"job": _job("a-starved", 3, "tenant-a", 0)})
    for sample in report["samples"]:
        if sample["phase"] == "baseline":
            sample["jobs"].append(_job("a-starved", 3, "tenant-a", 0))
            continue
        step = int((sample["elapsedSeconds"] - 20) / 10)
        sample["jobs"][0]["stats"]["frames"] = 100 + step * 200
        sample["jobs"].append(_job("a-starved", 3, "tenant-a", 0))

    result = analyze_fairness(report, _config())
    tenants = {item["workspaceLabel"]: item for item in result["tenants"]}

    assert tenants["tenant-a"]["targetAttainmentRatio"] == 1.0
    assert tenants["tenant-b"]["targetAttainmentRatio"] == 1.0
    assert result["checks"]["tenantAttainmentSpread"] is True
    assert result["checks"]["targetFpsPropagated"] is False
    assert result["checks"]["targetAttainment"] is False
    assert result["success"] is False


def test_distribution_is_allowed_only_when_explicitly_requested():
    report = _report(b_processor="processor-other")

    strict = analyze_fairness(report, _config(require_shared=True))
    distributed = analyze_fairness(report, _config(require_shared=False))

    assert strict["checks"]["sharedProcessor"] is False
    assert strict["success"] is False
    assert distributed["checks"]["sharedProcessor"] is True
    assert distributed["success"] is True


def test_workspace_local_job_id_collision_does_not_merge_logical_jobs():
    report = _report()
    for start in report["starts"]:
        start["job"]["id"] = "workspace-local-collision"
    for sample in report["samples"]:
        for job in sample["jobs"]:
            job["id"] = "workspace-local-collision"

    result = analyze_fairness(report, _config())

    assert result["checks"]["allJobsSampled"] is True
    assert result["checks"]["stableJobIdentity"] is True
    assert len(result["jobs"]) == 2
    assert result["success"] is True


def test_rejects_single_workspace_report_shape():
    report = _report()
    report["kind"] = "api-workflow-corpus"

    try:
        analyze_fairness(report, _config())
    except ValueError as error:
        assert "multi-workspace" in str(error)
    else:
        raise AssertionError("expected report kind validation")


def test_rejects_unknown_multi_workspace_report_schema():
    report = _report()
    report["schemaVersion"] = 999

    with pytest.raises(ValueError, match="schemaVersion"):
        analyze_fairness(report, _config())


def test_runner_attachment_promotes_fairness_failure_to_overall_failure():
    passing = attach_fairness_analysis(_report(), {"requireSingleProcessor": True})
    failing = attach_fairness_analysis(
        _report(a_measurement_rate=5), {"requireSingleProcessor": True}
    )

    assert passing["operationalSuccess"] is True
    assert passing["checkpoint"]["phase"] == "complete"
    assert passing["fairnessAnalysis"]["success"] is True
    assert passing["success"] is True
    assert failing["operationalSuccess"] is True
    assert failing["fairnessAnalysis"]["success"] is False
    assert failing["success"] is False


def test_frame_counter_reset_fails_even_when_remaining_intervals_hit_target():
    report = _report()
    measurement = [
        sample for sample in report["samples"] if sample["phase"] == "measurement"
    ]
    measurement[2]["jobs"][0] = _job("a", 1, "tenant-a", 0)
    measurement[3]["jobs"][0] = _job("a", 1, "tenant-a", 100)

    result = analyze_fairness(report, _config())

    assert result["checks"]["noFrameCounterResets"] is False
    assert result["success"] is False


def test_baseline_counter_reset_also_fails_delayed_arrival_certification():
    report = _report()
    report["samples"].insert(
        2,
        {
            "phase": "baseline",
            "elapsedSeconds": 15,
            "jobs": [_job("a", 1, "tenant-a", 0)],
        },
    )

    result = analyze_fairness(report, _config())

    incumbent = next(job for job in result["jobs"] if job["ordinal"] == 1)
    assert incumbent["baselineFrameCounterResets"] == 1
    assert result["checks"]["noFrameCounterResets"] is False
    assert result["success"] is False


def test_delayed_arrival_requires_measured_incumbent_baseline():
    report = _report()
    report["samples"] = [
        sample for sample in report["samples"] if sample["phase"] != "baseline"
    ]

    result = analyze_fairness(report, _config())

    assert result["checks"]["incumbentBaselineCoverage"] is False
    assert result["checks"]["incumbentRetention"] is False
    assert result["success"] is False


def test_missing_planned_job_is_not_silently_ignored():
    report = _report()
    report["starts"] = report["starts"][:1]
    for sample in report["samples"]:
        sample["jobs"] = [job for job in sample["jobs"] if job["ordinal"] != 2]

    result = analyze_fairness(report, _config())

    assert result["checks"]["allJobsStarted"] is False
    assert result["checks"]["allJobsSampled"] is False
    assert result["checks"]["completeMeasurementSamples"] is False
    assert result["success"] is False


def test_missing_target_on_any_job_is_not_silently_ignored():
    report = _report()
    report["workloads"][1]["maxFps"] = None

    result = analyze_fairness(report, _config())

    assert result["checks"]["allTargetsDefined"] is False
    assert result["checks"]["targetAttainment"] is False
    assert result["success"] is False


def test_sampled_ema_latency_is_not_accepted_for_certification():
    report = _report()
    for sample in report["samples"]:
        for job in sample["jobs"]:
            job["stats"].pop("decodeToResultLatency", None)

    result = analyze_fairness(report, _config())

    assert result["checks"]["frameLatencyHistogramAvailable"] is False
    assert result["success"] is False
