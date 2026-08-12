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
