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

from recommendations import build_recommendations, render_markdown  # noqa: E402
from report import AnalysisConfig  # noqa: E402
from test_workflow_benchmark_analysis import make_report  # noqa: E402


def test_recommendation_is_bounded_and_does_not_invent_economics():
    c1 = make_report("c1", 1)
    c2 = make_report("c2", 2)
    c2["samples"][-1]["jobs"][1]["stats"]["decodeToResultLatencyMs"] = 60
    config = AnalysisConfig(warmup_seconds=2, max_fps_spread_ratio=0.20)

    result = build_recommendations(
        [c1, c2], strict_config=config, relaxed_max_latency_ms=75
    )

    recommendation = result["recommendations"][0]
    assert recommendation["recommendation"]["maxStreamsPerWorker"] == 1
    assert (
        recommendation["recommendation"]["relaxedLatencyMaxStreamsPerWorker"]
        == 2
    )
    assert recommendation["evidence"]["firstObservedFailureAboveStrict"] == 2
    assert recommendation["recommendation"]["pricingReady"] is False
    assert "controlled input FPS curve" in recommendation["recommendation"][
        "missingEvidence"
    ]
    assert result["economics"]["status"] == "not-computed"


def test_right_censored_curve_is_labeled_lower_bound_only():
    c1 = make_report("c1", 1)
    c1["profiles"][0]["maxFps"] = 10
    result = build_recommendations(
        [c1], strict_config=AnalysisConfig(warmup_seconds=2)
    )

    recommendation = result["recommendations"][0]
    assert recommendation["evidence"]["boundaryClassification"] == (
        "lower-bound-only"
    )
    assert "observed failing boundary" in "; ".join(
        recommendation["recommendation"]["missingEvidence"]
    )


def test_markdown_calls_recommendations_provisional_not_pricing():
    result = build_recommendations(
        [make_report("c1", 1)],
        strict_config=AnalysisConfig(warmup_seconds=2),
    )
    rendered = render_markdown(result)

    assert "Provisional" in rendered
    assert "not pricing" in rendered
    assert "worker cost" in rendered
