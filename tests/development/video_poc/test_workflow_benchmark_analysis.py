import sys
from pathlib import Path

ANALYSIS_DIR = (
    Path(__file__).resolve().parents[3]
    / "development"
    / "video_poc"
    / "benchmarks"
    / "analysis"
)
sys.path.insert(0, str(ANALYSIS_DIR))

from report import (  # noqa: E402
    AnalysisConfig,
    analyze_report,
    analyze_reports,
    render_markdown,
)


def make_report(run_id, streams, processor_ids=None, success=True):
    processor_ids = processor_ids or ["processor-a"] * streams
    starts = []
    profiles = []
    for ordinal in range(1, streams + 1):
        job = {
            "id": f"job-{ordinal}",
            "state": "queued",
            "tier": "gpu",
            "mode": "stream",
            "stats": {"frames": 0},
        }
        starts.append(
            {
                "ordinal": ordinal,
                "profile": "simple-detector",
                "httpStatus": 201,
                "job": job,
            }
        )
        profiles.append(
            {
                "ordinal": ordinal,
                "profile": "simple-detector",
                "tier": "gpu",
                "mode": "stream",
                "imageOutput": None,
            }
        )

    samples = []

    def add_sample(phase, elapsed, state, frame_multiplier):
        jobs = []
        for index in range(streams):
            frames = frame_multiplier * (20 if index == 0 else 18)
            jobs.append(
                {
                    "id": f"job-{index + 1}",
                    "state": state,
                    "processorId": processor_ids[index],
                    "attempts": 0,
                    "stats": {
                        "frames": frames,
                        "decodeToResultLatencyMs": 10 + 2 * index,
                        "pipelineStartS": 2 + index,
                        "timeToFirstResultS": 3 + index,
                    },
                }
            )
        samples.append({"phase": phase, "elapsedSeconds": elapsed, "jobs": jobs})

    add_sample("startup", 2, "running", 0)
    add_sample("measurement", 4, "running", 1)
    add_sample("measurement", 6, "running", 2)
    add_sample("measurement", 8, "running", 3)
    add_sample("measurement", 10, "running", 4)
    return {
        "schemaVersion": 1,
        "runId": run_id,
        "success": success,
        "plannedConcurrency": streams,
        "source": {"id": "source-a", "name": "fixture"},
        "profiles": profiles,
        "starts": starts,
        "samples": samples,
        "errors": [] if success else [{"phase": "measurement", "error": "failed"}],
    }


def test_analyze_report_derives_frame_counter_fps_startup_and_fairness():
    analysis = analyze_report(
        make_report("c2", 2), AnalysisConfig(warmup_seconds=2)
    )

    assert analysis["aggregate"] == {
        "totalDeliveredFps": 19.0,
        "streamsWithSteadyFps": 2,
    }
    assert analysis["streams"][0]["steadyState"] == {
        "deliveredFps": 10.0,
        "intervalFpsP05": 10.0,
        "intervalFpsP50": 10.0,
        "intervalFpsP95": 10.0,
        "steadyIntervals": 2,
        "steadyObservedSeconds": 4,
        "steadyDeliveredFrames": 40,
        "frameCounterResets": 0,
        "sampledEmaLatencyMeanMs": 10,
        "sampledEmaLatencyP50Ms": 10,
        "sampledEmaLatencyP95Ms": 10.0,
        "sampledEmaLatencyMaxMs": 10,
        "latencySamples": 3,
        "frameLatencyHistogramCount": 0,
        "frameLatencyMeanMs": None,
        "frameLatencyP50ApproxMs": None,
        "frameLatencyP95ApproxMs": None,
        "frameLatencyP99ApproxMs": None,
        "latencyP95ForSloMs": 10.0,
        "latencySource": "sampled_ema",
        "counterDeltas": {},
    }
    assert analysis["streams"][1]["steadyState"]["deliveredFps"] == 9.0
    assert analysis["streams"][1]["startup"]["timeToFirstResultS"] == 4
    assert analysis["placement"]["allStreamsCoLocated"] is True
    assert analysis["placement"]["peakConcurrentJobsByProcessor"] == {
        "processor-a": 2
    }
    assert analysis["fairness"]["deliveredFpsJainIndex"] == 0.997238
    assert analysis["fairness"]["deliveredFpsSpreadRatio"] == 0.105


def test_capacity_curve_compares_streams_to_baseline_and_checks_placement():
    config = AnalysisConfig(warmup_seconds=2, max_fps_spread_ratio=0.20)
    analysis = analyze_reports(
        [
            make_report("c1", 1),
            make_report("c2", 2),
            make_report("c2-split", 2, ["processor-a", "processor-b"]),
        ],
        config,
    )

    capacity = analysis["capacitySummaries"][0]
    assert capacity["baselineConcurrency"] == 1
    assert capacity["baselineDeliveredFps"] == 10.0
    assert capacity["maxPassingConcurrency"] == 2
    by_id = {run["runId"]: run for run in analysis["runs"]}
    assert by_id["c2"]["streams"][1]["baselineComparison"][
        "fpsRetentionRatio"
    ] == 0.9
    assert by_id["c2"]["capacitySlo"]["passed"] is True
    assert by_id["c2-split"]["capacitySlo"]["checks"]["singleProcessor"] is False
    assert by_id["c2-split"]["capacitySlo"]["passed"] is False


def test_counter_reset_is_excluded_and_failed_report_cannot_pass():
    report = make_report("reset", 1, success=False)
    report["samples"][-1]["jobs"][0]["stats"]["frames"] = 5
    config = AnalysisConfig(warmup_seconds=0, min_steady_intervals=1)
    analysis = analyze_reports([report], config)

    stream = analysis["runs"][0]["streams"][0]
    assert stream["steadyState"]["frameCounterResets"] == 1
    assert stream["steadyState"]["steadyIntervals"] == 2
    assert analysis["runs"][0]["capacitySlo"]["checks"]["reportSucceeded"] is False
    assert analysis["runs"][0]["capacitySlo"]["passed"] is False


def test_markdown_warns_that_latency_is_sampled_ema():
    analysis = analyze_reports(
        [make_report("c1", 1)], AnalysisConfig(warmup_seconds=2)
    )
    rendered = render_markdown(analysis)

    assert "sampled rolling EMAs, not per-frame percentiles" in rendered
    assert "| c1 | 1 | 10.0" in rendered


def test_schema_v2_uses_frame_histogram_and_counter_deltas_for_slo():
    report = make_report("schema-v2", 1)
    report["schemaVersion"] = 2
    for sample in report["samples"]:
        stats = sample["jobs"][0]["stats"]
        frames = stats["frames"]
        stats["schemaVersion"] = 2
        stats["counters"] = {
            "captured": frames + 2,
            "decoded": frames + 1,
            "dropped": 1,
            "inferred": frames,
            "rendered": frames,
            "published": 0,
        }
        stats["decodeToResultLatency"] = {
            "count": frames,
            "sum": frames * 15,
            "max": 15,
            "histogram": {
                "bounds": [10, 20, None],
                "cumulativeCounts": [0, frames, frames],
            },
        }

    analysis = analyze_reports([report], AnalysisConfig(warmup_seconds=2))
    steady = analysis["runs"][0]["streams"][0]["steadyState"]

    assert steady["frameLatencyHistogramCount"] == 40
    assert steady["frameLatencyMeanMs"] == 15
    assert steady["frameLatencyP95ApproxMs"] == 20
    assert steady["latencyP95ForSloMs"] == 20
    assert steady["latencySource"] == "frame_histogram"
    assert steady["counterDeltas"] == {
        "captured": 40,
        "decoded": 40,
        "dropped": 0,
        "inferred": 40,
        "published": 0,
        "rendered": 40,
    }
    assert analysis["runs"][0]["capacitySlo"]["maxLatencyP95Ms"] == 20


def test_capacity_curves_keep_controlled_fps_separate_from_unbounded():
    unbounded = make_report("unbounded-c1", 1)
    controlled = make_report("controlled-c1", 1)
    controlled["profiles"][0]["maxFps"] = 15

    analysis = analyze_reports(
        [unbounded, controlled], AnalysisConfig(warmup_seconds=2)
    )

    assert len(analysis["capacitySummaries"]) == 2
    assert {item["maxFps"] for item in analysis["capacitySummaries"]} == {
        None,
        15,
    }
    rendered = render_markdown(analysis)
    assert "unbounded input" in rendered
    assert "max 15 FPS" in rendered


def test_recovery_tolerant_fault_runs_are_summarized_but_not_certified():
    report = make_report("recovered", 1)
    report["recoveryTimeoutSeconds"] = 180
    report["recoveries"] = [
        {
            "outcome": "recovered",
            "observedControlPlaneRecoverySeconds": 8.5,
        },
        {
            "outcome": "recovered",
            "observedControlPlaneRecoverySeconds": 3.0,
        },
    ]

    analysis = analyze_reports([report], AnalysisConfig(warmup_seconds=2))

    assert analysis["capacitySummaries"] == []
    assert analysis["runs"][0]["capacityExcludedReason"] == (
        "recovery-tolerant fault run"
    )
    assert analysis["runs"][0]["recovery"] == {
        "toleranceSeconds": 180,
        "eventCount": 2,
        "recoveredCount": 2,
        "failedCount": 0,
        "incompleteCount": 0,
        "totalObservedControlPlaneRecoverySeconds": 11.5,
        "maxObservedControlPlaneRecoverySeconds": 8.5,
    }


def test_incomplete_recovery_is_counted_and_excluded_from_capacity():
    report = make_report("incomplete", 1)
    report["recoveries"] = [{"startedElapsedSeconds": 12.0}]

    analysis = analyze_reports([report], AnalysisConfig(warmup_seconds=2))

    assert analysis["capacitySummaries"] == []
    assert analysis["runs"][0]["recovery"]["incompleteCount"] == 1
