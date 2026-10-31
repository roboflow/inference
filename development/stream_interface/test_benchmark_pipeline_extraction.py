import copy
import json
import os
import subprocess
import sys
import tempfile

import pytest

from development.stream_interface import benchmark_pipeline_extraction as harness
from development.stream_interface.benchmark_pipeline_extraction import SinkRecord

SECOND = 1_000_000_000
MS = 1_000_000


def test_percentile_matches_linear_interpolation() -> None:
    values = list(range(1, 11))

    assert harness.percentile(values, 50) == pytest.approx(5.5)
    assert harness.percentile(values, 95) == pytest.approx(9.55)
    assert harness.percentile(values, 0) == 1
    assert harness.percentile(values, 100) == 10
    assert harness.percentile(list(reversed(values)), 95) == pytest.approx(9.55)
    assert harness.percentile([7.0], 95) == 7.0
    assert harness.percentile([], 50) is None


@pytest.mark.parametrize(
    "frame_ids, expected",
    [
        ([1, 2, 3], {"missing": 0, "reordered": 0, "duplicates": 0}),
        ([1, 2, 5], {"missing": 2, "reordered": 0, "duplicates": 0}),
        ([1, 3, 2, 4], {"missing": 0, "reordered": 1, "duplicates": 0}),
        ([1, 2, 2, 3], {"missing": 0, "reordered": 0, "duplicates": 1}),
        ([5, 4], {"missing": 0, "reordered": 1, "duplicates": 0}),
    ],
)
def test_analyze_frame_ids_detects_gaps_reordering_and_duplicates(
    frame_ids, expected
) -> None:
    result = harness.analyze_frame_ids(frame_ids)

    assert {key: result[key] for key in expected} == expected
    assert result["delivered"] == len(frame_ids)


def _known_records():
    start = 10 * SECOND
    records = [
        # Warmup delivery before the window: integrity only.
        SinkRecord(start - MS, 0, 1, 99 * MS, None, "a"),
    ]
    for index in range(10):
        records.append(
            SinkRecord(
                start + index * 50 * MS, 0, index + 2, (index + 1) * MS, 2 * MS, "b"
            )
        )
    # Delivered after the window closed: excluded from fps and latency.
    records.append(SinkRecord(start + SECOND, 0, 12, 500 * MS, None, "c"))
    for frame_id in [1, 2, 4, 3, 5, 5]:
        records.append(SinkRecord(start + 100 * MS, 1, frame_id, None, None, "d"))
    return start, records


def test_summarize_repeat_counts_window_and_latency_percentiles() -> None:
    start, records = _known_records()

    summary = harness.summarize_repeat(
        records, start, start + SECOND, [0, 1], latency_label="capture_to_sink"
    )

    metrics = summary["metrics"]
    first, second = summary["sources"]["0"], summary["sources"]["1"]
    assert first["window_frames"] == 10
    assert first["fps"] == pytest.approx(10.0)
    assert first["delivered"] == 12
    assert (first["missing"], first["reordered"], first["duplicates"]) == (0, 0, 0)
    assert second["window_frames"] == 6
    assert (second["missing"], second["reordered"], second["duplicates"]) == (0, 1, 1)
    assert metrics["window_frames"] == 16
    assert metrics["total_fps"] == pytest.approx(16.0)
    assert metrics["min_source_fps"] == pytest.approx(6.0)
    assert metrics["source_fairness"] == pytest.approx(0.6)
    assert metrics["capture_to_sink_count"] == 10
    assert metrics["capture_to_sink_p50_ms"] == pytest.approx(5.5)
    assert metrics["capture_to_sink_p95_ms"] == pytest.approx(9.55)
    assert metrics["consume_to_inference_completed_count"] == 10
    assert metrics["reordered_frames"] == 1
    assert metrics["duplicate_frames"] == 1
    assert metrics["starved_sources"] == 0


def test_summarize_repeat_reports_starved_source() -> None:
    start, records = _known_records()

    summary = harness.summarize_repeat(records, start, start + SECOND, [0, 1, 2], None)

    assert summary["metrics"]["starved_sources"] == 1
    assert summary["metrics"]["min_source_fps"] == 0.0
    assert not any(key.startswith("capture_to") for key in summary["metrics"])


def _result(fps=100.0, frame_ids=range(1, 101), digests=None, repeats=3):
    frame_ids = list(frame_ids)
    records = [
        SinkRecord(
            SECOND + i * MS,
            0,
            frame_id,
            10 * MS,
            None,
            (digests or {}).get(frame_id, f"d{frame_id}"),
        )
        for i, frame_id in enumerate(frame_ids)
    ]
    summary = harness.summarize_repeat(
        records, SECOND, 2 * SECOND, [0], "capture_to_sink"
    )
    runs = []
    for index in range(repeats):
        metrics = dict(summary["metrics"], total_fps=fps + index * 0.1)
        metrics["min_source_fps"] = metrics["total_fps"]
        metrics.setdefault("cpu_percent_mean", 50.0)
        metrics.setdefault("rss_mb_peak", 200.0)
        runs.append(
            {
                "repeat": index,
                "errors": [],
                "metrics": metrics,
                "raw": {"digests": harness._digests_by_source(records)},
            }
        )
    return {
        "fingerprint": {"workflow_sha256": "w", "copies": 1, "sources": ["s0"]},
        "repeats": runs,
    }


def content_missing_on_candidate(comparison: dict) -> bool:
    """True if the comparison correctly refused to treat the candidate as having
    proven content coverage (either an outright failure, or inconclusive because
    digests were effectively recorded on only one side)."""
    return any(
        "no content digests for expected source(s)" in f for f in comparison["failures"]
    ) or any(
        "content digests recorded on only one side" in msg
        for msg in comparison["inconclusive"]
    )


def test_compare_passes_for_equivalent_runs() -> None:
    comparison = harness.compare_results(_result(), _result(fps=98.0))

    assert comparison["passed"], comparison["failures"]
    assert comparison["content"]["compared"] == 300
    assert comparison["metrics"]["total_fps"]["relative_delta"] == pytest.approx(
        -0.02, abs=1e-3
    )


def test_compare_fails_on_throughput_regression() -> None:
    comparison = harness.compare_results(_result(), _result(fps=95.0))

    assert not comparison["passed"]
    assert any("total_fps regressed" in f for f in comparison["failures"])


def test_compare_fails_on_missing_frames() -> None:
    candidate = _result(frame_ids=[i for i in range(1, 101) if i != 50])

    comparison = harness.compare_results(_result(), candidate)

    assert any("missing frames" in f for f in comparison["failures"])


def test_compare_fails_on_reordered_frames() -> None:
    frame_ids = list(range(1, 101))
    frame_ids[10], frame_ids[11] = frame_ids[11], frame_ids[10]

    comparison = harness.compare_results(_result(), _result(frame_ids=frame_ids))

    assert any("reordered" in f for f in comparison["failures"])


def test_compare_fails_on_changed_content() -> None:
    comparison = harness.compare_results(_result(), _result(digests={7: "other"}))

    assert any("content digest mismatch on 3" in f for f in comparison["failures"])


def test_compare_fails_for_incomparable_configuration() -> None:
    candidate = _result()
    candidate["fingerprint"]["copies"] = 4

    comparison = harness.compare_results(_result(), candidate)

    assert any("'copies' differs" in f for f in comparison["failures"])


def test_compare_fails_when_physical_cpu_differs() -> None:
    """Physical hardware must match: performance comparisons are only valid on the
    same CPU."""
    baseline = _result()
    candidate = _result(fps=98.0)
    candidate["fingerprint"]["cpu_model"] = "Intel Core i9 (different)"

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("'cpu_model' differs" in f for f in comparison["failures"])


def test_compare_fails_when_physical_gpus_differ() -> None:
    """Physical GPU hardware must match: performance comparisons are only valid on
    the same GPU."""
    baseline = _result()
    candidate = _result(fps=98.0)
    candidate["fingerprint"]["gpus"] = "NVIDIA RTX A100 (different)"

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("'gpus' differs" in f for f in comparison["failures"])


def test_compare_allows_software_host_label_to_differ() -> None:
    """Software --host labels (legacy/nextgen) can differ; only physical hardware
    must match."""
    baseline = _result()
    candidate = _result(fps=98.0)
    baseline["labels"] = {"host": "legacy", "transport": "in-process"}
    candidate["labels"] = {"host": "nextgen-direct", "transport": "in-process"}

    comparison = harness.compare_results(baseline, candidate)

    assert comparison["passed"], comparison["failures"] + comparison["inconclusive"]


def test_compare_fails_when_a_run_is_invalid() -> None:
    candidate = copy.deepcopy(_result())
    candidate["repeats"][1]["errors"] = ["pipeline ended before warmup completed"]

    comparison = harness.compare_results(_result(), candidate)

    assert any("candidate repeat 1 invalid" in f for f in comparison["failures"])


def test_compare_marks_inconclusive_for_unequal_repeat_counts() -> None:
    comparison = harness.compare_results(_result(repeats=3), _result(repeats=2))

    assert not comparison["passed"]
    assert any("repeat count differs" in msg for msg in comparison["inconclusive"])


def test_compare_marks_inconclusive_when_digests_present_on_one_side_only() -> None:
    baseline = _result()
    candidate = _result()
    for repeat in candidate["repeats"]:
        repeat["raw"]["digests"] = {}

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("only one side" in msg for msg in comparison["inconclusive"])


def test_compare_fails_when_digests_unsupported_on_both_sides() -> None:
    """Output content evidence is mandatory even when neither side captured it."""
    baseline = _result()
    candidate = _result(fps=98.0)
    for result in (baseline, candidate):
        for repeat in result["repeats"]:
            repeat["raw"]["digests"] = {}

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("content digests not captured" in f for f in comparison["failures"])


def test_compare_marks_inconclusive_for_missing_source_coverage() -> None:
    baseline = _result()
    baseline["repeats"][0]["raw"]["digests"]["1"] = [[1, "extra"]]
    candidate = _result()

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("missing source coverage" in msg for msg in comparison["inconclusive"])


def test_compare_fails_when_candidate_digests_are_an_empty_list_per_source() -> None:
    """An empty digest list for a source is not coverage: it must not count as if
    the source's content had been captured and compared (it must not silently pass
    as though both sides had matching digests)."""
    baseline = _result()
    candidate = _result(fps=98.0)
    for repeat in candidate["repeats"]:
        repeat["raw"]["digests"] = {source: [] for source in repeat["raw"]["digests"]}

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert content_missing_on_candidate(comparison)


def test_compare_fails_when_digests_are_null() -> None:
    """A null digest proves nothing about frame content and must not count as a
    match (or as coverage) against the baseline's real digest."""
    baseline = _result()
    candidate = _result(fps=98.0)
    for repeat in candidate["repeats"]:
        repeat["raw"]["digests"] = {
            source: [[frame_id, None] for frame_id, _ in frames]
            for source, frames in repeat["raw"]["digests"].items()
        }

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert content_missing_on_candidate(comparison)


def test_compare_marks_inconclusive_when_no_common_digests() -> None:
    baseline = _result()
    candidate = _result()
    for repeat in candidate["repeats"]:
        repeat["raw"]["digests"] = {
            "0": [
                [frame_id + 1000, digest]
                for frame_id, digest in repeat["raw"]["digests"]["0"]
            ]
        }

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("no common content digests" in msg for msg in comparison["inconclusive"])


def test_compare_marks_inconclusive_for_metric_missing_on_one_side() -> None:
    baseline = _result()
    candidate = _result(fps=98.0)
    for repeat in candidate["repeats"]:
        del repeat["metrics"]["total_fps"]

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(
        "total_fps: metric present on only one side" in msg
        for msg in comparison["inconclusive"]
    )


def test_compare_marks_inconclusive_when_metric_missing_from_single_candidate_repeat() -> (
    None
):
    """Aggregation across repeats must not hide a single paired repeat gap."""
    baseline = _result()
    candidate = _result(fps=98.0)
    del candidate["repeats"][1]["metrics"]["total_fps"]

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(
        "total_fps: metric present on only one side" in msg and "repeats [1]" in msg
        for msg in comparison["inconclusive"]
    )


def test_compare_fails_when_declared_source_has_no_digests_in_every_repeat() -> None:
    """Expected source ids come from declared sources/copies, not observed digests."""
    baseline = _result()
    baseline["fingerprint"]["sources"] = ["s0", "s1"]
    baseline["fingerprint"]["copies"] = 1
    candidate = _result(fps=98.0)
    candidate["fingerprint"]["sources"] = ["s0", "s1"]
    candidate["fingerprint"]["copies"] = 1
    # Both sides only ever recorded digests for source 0: this must fail, not pass by
    # aggregation over an observed-sources set that never included source 1.

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(
        "no content digests for expected source(s) 1" in f
        for f in comparison["failures"]
    )


def test_compare_does_not_require_equal_result_counts_between_different_speed_windows() -> (
    None
):
    baseline = _result(fps=100.0, frame_ids=range(1, 101))
    candidate = _result(fps=150.0, frame_ids=range(1, 151))

    comparison = harness.compare_results(baseline, candidate)

    assert comparison["passed"], comparison["failures"] + comparison["inconclusive"]


def test_compare_sanitizes_raw_fingerprint_values_in_failures() -> None:
    """Old/user-provided artifacts can carry unredacted references; diagnostics and
    saved comparisons must not leak them just because they came from stored data
    rather than a fresh run."""
    baseline = _result()
    baseline["fingerprint"]["sources"] = [
        "rtsp://user:pass@cam/live?custom_secret=TOPSECRET"
    ]
    candidate = _result(fps=98.0)
    candidate["fingerprint"]["sources"] = ["rtsp://other/live"]

    comparison = harness.compare_results(baseline, candidate)

    dumped = json.dumps(comparison)
    assert "TOPSECRET" not in dumped
    assert "user:pass" not in dumped
    assert any("'sources' differs" in f for f in comparison["failures"])


def test_compare_marks_inconclusive_for_excessive_baseline_cv() -> None:
    baseline = _result()
    for repeat, fps in zip(baseline["repeats"], [90.0, 100.0, 110.0]):
        repeat["metrics"]["total_fps"] = fps
        repeat["metrics"]["min_source_fps"] = fps

    comparison = harness.compare_results(baseline, _result(fps=100.0))

    assert not comparison["passed"]
    assert any("baseline CV" in msg for msg in comparison["inconclusive"])


def test_compare_marks_inconclusive_for_excessive_candidate_cv() -> None:
    """A candidate that varies more than the declared band is just as unproven as a
    noisy baseline: it must not silently pass."""
    candidate = _result(fps=100.0)
    for repeat, fps in zip(candidate["repeats"], [90.0, 100.0, 110.0]):
        repeat["metrics"]["total_fps"] = fps
        repeat["metrics"]["min_source_fps"] = fps

    comparison = harness.compare_results(_result(), candidate)

    assert not comparison["passed"]
    assert any("candidate CV" in msg for msg in comparison["inconclusive"])


def test_compare_fails_when_mandatory_metric_missing_in_same_repeat_on_both_sides() -> (
    None
):
    """A mandatory metric (CPU here) absent for the same repeat on both sides must not
    be skipped by the aggregate n>0 check: that repeat has no real evidence."""
    baseline, candidate = _result(), _result(fps=98.0)
    del baseline["repeats"][1]["metrics"]["cpu_percent_mean"]
    del candidate["repeats"][1]["metrics"]["cpu_percent_mean"]

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(
        "cpu_percent_mean: mandatory metric" in f and "repeat(s) [1]" in f
        for f in comparison["failures"]
    )


def test_compare_does_not_require_manager_or_gpu_metrics_for_a_plain_in_process_run() -> (
    None
):
    """capture_to_consume and GPU memory are optional unless the run's labels/
    resources actually apply (manager transport / GPU evidence)."""
    comparison = harness.compare_results(_result(), _result(fps=98.0))

    assert comparison["passed"], comparison["failures"] + comparison["inconclusive"]
    assert "capture_to_consume_p50_ms" not in comparison["metrics"]
    assert "gpu_memory_mb_peak" not in comparison["metrics"]


def test_compare_requires_capture_to_consume_evidence_for_manager_transport() -> None:
    baseline, candidate = _result(), _result(fps=98.0)
    baseline["labels"] = candidate["labels"] = {"transport": "manager"}

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(
        "capture_to_consume_p50_ms: mandatory metric" in f
        for f in comparison["failures"]
    )


def test_compare_requires_gpu_evidence_when_gpu_resources_were_sampled() -> None:
    baseline, candidate = _result(), _result(fps=98.0)
    baseline["repeats"][0]["resources"] = {"gpu": {"tree_process_memory_mb_peak": 10}}

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(
        "gpu_memory_mb_peak: mandatory metric" in f for f in comparison["failures"]
    )


def test_compare_requires_model_call_evidence_when_model_calls_occurred() -> None:
    baseline, candidate = _result(), _result(fps=98.0)
    for repeat in baseline["repeats"] + candidate["repeats"]:
        repeat["instrumentation"] = {"model_calls": {"available": True}}

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(
        "model_call_p50_ms: mandatory metric" in f for f in comparison["failures"]
    )


@pytest.mark.parametrize("metric_name", ["model_call_p50_ms", "model_call_p95_ms"])
def test_compare_fails_on_model_call_latency_regression(metric_name) -> None:
    """A 4% model-call latency regression must fail under the 3% latency band (it
    would have passed unnoticed under the previous, looser 5% band)."""
    baseline, candidate = _result(), _result(fps=98.0)
    baseline["model"] = {"weights_verified": True}
    candidate["model"] = {"weights_verified": True}
    baseline_latency_ms = 10.0
    for repeat in baseline["repeats"]:
        repeat["instrumentation"] = {"model_calls": {"available": True}}
        repeat["metrics"]["model_call_p50_ms"] = baseline_latency_ms
        repeat["metrics"]["model_call_p95_ms"] = baseline_latency_ms
    for repeat in candidate["repeats"]:
        repeat["instrumentation"] = {"model_calls": {"available": True}}
        repeat["metrics"]["model_call_p50_ms"] = baseline_latency_ms
        repeat["metrics"]["model_call_p95_ms"] = baseline_latency_ms
        repeat["metrics"][metric_name] = baseline_latency_ms * 1.04

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(f"{metric_name} regressed" in f for f in comparison["failures"])


def test_compare_filters_non_finite_metric_values_as_missing_evidence() -> None:
    """NaN/inf readings are not real evidence; they must not inflate n or pass a
    metric that has no finite support."""
    baseline, candidate = _result(), _result(fps=98.0)
    for repeat in baseline["repeats"]:
        repeat["metrics"]["cpu_percent_mean"] = float("nan")
    for repeat in candidate["repeats"]:
        repeat["metrics"]["cpu_percent_mean"] = float("nan")

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any(
        "cpu_percent_mean: mandatory metric" in f for f in comparison["failures"]
    )


def test_output_digest_ignores_random_ids_and_float_noise() -> None:
    first = {"predictions": [{"x": 1.00001, "detection_id": "a", "class": "person"}]}
    second = {"predictions": [{"x": 1.00002, "detection_id": "b", "class": "person"}]}
    third = {"predictions": [{"x": 1.5, "detection_id": "a", "class": "person"}]}

    assert harness.output_digest(first) == harness.output_digest(second)
    assert harness.output_digest(first) != harness.output_digest(third)


@pytest.mark.parametrize("host", ["nextgen-direct", "nextgen-gateway"])
def test_future_hosts_fail_clearly(host, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        harness.parse_args(["--host", host, "--workflow", "w.json", "--source", "x"])

    assert error.value.code == 2
    assert "not implemented yet" in capsys.readouterr().err


def test_sanitize_reference_redacts_credentials() -> None:
    sanitized = harness.sanitize_reference(
        "rtsp://user:secret@cam.local/stream?token=abc"
    )

    assert "user" not in sanitized
    assert "secret" not in sanitized
    assert "token=abc" not in sanitized


def test_sanitize_reference_strips_userinfo() -> None:
    assert (
        harness.sanitize_reference("rtsp://admin:hunter2@10.0.0.5:554/live")
        == "rtsp://10.0.0.5:554/live"
    )


def test_sanitize_reference_strips_arbitrary_unknown_query_credentials() -> None:
    sanitized = harness.sanitize_reference(
        "https://example.com/video.mp4?auth=SECRETVALUE&signature=XYZ123"
    )

    assert "SECRETVALUE" not in sanitized
    assert "XYZ123" not in sanitized
    assert "auth" not in sanitized
    assert "signature" not in sanitized


def test_sanitize_command_redacts_schemeless_source_credentials() -> None:
    """`sanitize_reference` alone only redacts scheme-URL tokens: a bare
    `user:pass@host:port/path` source reference (no `rtsp://` prefix) needs the
    source-aware sanitizer applied to the argv value that follows --source."""
    sanitized = harness._sanitize_command(
        ["prog", "--source", "user:pass@host:554/live", "--model-id", "coco/36"]
    )

    assert sanitized == ["prog", "--source", "host:554/live", "--model-id", "coco/36"]


@pytest.mark.parametrize(
    "flag,form,credential",
    [
        ("--source", "separate", "user:pass@host:554/live"),
        ("--source", "equals", "user:pass@host:554/live"),
        ("--source", "separate", "rtsp://user:pass@host:554/live"),
        ("--source", "equals", "rtsp://user:pass@host:554/live"),
        ("--model-id", "separate", "user:pass@host:554/live"),
        ("--model-id", "equals", "user:pass@host:554/live"),
        ("--model-id", "separate", "rtsp://user:pass@host:554/live"),
        ("--model-id", "equals", "rtsp://user:pass@host:554/live"),
    ],
)
def test_sanitize_command_reference_flags_both_forms_and_credentials(
    flag: str, form: str, credential: str
) -> None:
    """Verify credential redaction works for both space-separated and equals forms."""
    if form == "separate":
        argv = ["prog", flag, credential, "other"]
        expected_credential = credential.replace("user:pass@", "")
        expected = ["prog", flag, expected_credential, "other"]
    else:
        argv = ["prog", f"{flag}={credential}", "other"]
        expected_credential = credential.replace("user:pass@", "")
        expected = ["prog", f"{flag}={expected_credential}", "other"]

    sanitized = harness._sanitize_command(argv)
    assert sanitized == expected


def test_build_result_fingerprint_records_dependencies_transport_and_host_provenance(
    tmp_path,
) -> None:
    workflow = tmp_path / "w.json"
    workflow.write_text("{}")
    source = tmp_path / "s.mp4"
    source.write_bytes(b"")
    args = harness.parse_args(
        [
            "--workflow", str(workflow), "--source", str(source),
            "--output", str(tmp_path / "out.json"), "--transport", "manager",
        ]
    )  # fmt: skip

    result = harness.build_result(args, [])

    fingerprint = result["fingerprint"]
    assert fingerprint["transport"] == "manager"
    assert fingerprint["python_version"]
    assert "numpy" in fingerprint["dependency_versions"]
    # Physical hardware must be in the comparability gate: performance comparisons
    # require the same physical CPU, GPU, and driver. Software --host labels
    # (legacy/nextgen) are allowed to differ.
    assert fingerprint["cpu_model"] == result["environment"]["cpu_model"]
    assert fingerprint["gpus"] == result["environment"]["gpus"]


def test_build_result_fingerprint_includes_cuda_child_main_thread_init_only_for_linux_manager(
    tmp_path, monkeypatch
) -> None:
    """Mac manager baselines predate this field: gating it on `manager_env_overrides`
    (Linux-only) rather than on transport alone keeps them comparable, since the
    child records a non-empty `skipped:not-linux` status on Mac too."""
    workflow = tmp_path / "w.json"
    workflow.write_text("{}")
    source = tmp_path / "s.mp4"
    source.write_bytes(b"")
    args = harness.parse_args(
        [
            "--workflow", str(workflow), "--source", str(source),
            "--output", str(tmp_path / "out.json"), "--transport", "manager",
        ]
    )  # fmt: skip
    repeats = [
        {
            "instrumentation": {"cuda_child_main_thread_init": "skipped:not-linux"},
            "metrics": {},
        }
    ]

    monkeypatch.setattr(harness, "manager_launch_env_overrides", lambda: {})
    mac_fingerprint = harness.build_result(args, copy.deepcopy(repeats))["fingerprint"]
    assert "cuda_child_main_thread_init" not in mac_fingerprint

    monkeypatch.setattr(
        harness, "manager_launch_env_overrides", lambda: {"PYTORCH_CUDA_INIT": "0"}
    )
    linux_fingerprint = harness.build_result(args, copy.deepcopy(repeats))[
        "fingerprint"
    ]
    assert linux_fingerprint["cuda_child_main_thread_init"] == ["skipped:not-linux"]


def test_build_result_sanitizes_command_source_reference(tmp_path) -> None:
    workflow = tmp_path / "w.json"
    workflow.write_text("{}")
    source = tmp_path / "s.mp4"
    source.write_bytes(b"")
    args = harness.parse_args(
        [
            "--workflow", str(workflow), "--source", str(source),
            "--model-id", "user:pass@host:554/live",
            "--output", str(tmp_path / "out.json"),
        ]
    )  # fmt: skip

    result = harness.build_result(args, [])

    dumped = json.dumps(result)
    assert "user:pass" not in dumped


def test_main_sanitizes_child_repeat_output_before_writing(
    tmp_path, monkeypatch
) -> None:
    """Sanitizing only at the parent (build_result) is too late: the child writes
    its own repeat.json, which is later read back and folded in verbatim."""
    workflow = tmp_path / "w.json"
    workflow.write_text("{}")
    source = tmp_path / "s.mp4"
    source.write_bytes(b"")
    output = tmp_path / "repeat.json"

    def fake_run_single_repeat(args, repeat):
        return {
            "repeat": repeat,
            "errors": [
                "failed to open rtsp://user:pass@cam.local/stream?token=abc: timeout"
            ],
            "metrics": {},
        }

    monkeypatch.setattr(harness, "run_single_repeat", fake_run_single_repeat)

    exit_code = harness.main(
        [
            "--workflow", str(workflow), "--source", str(source),
            "--output", str(output), "--child-repeat", "0",
        ]
    )  # fmt: skip

    assert exit_code == 0
    dumped = output.read_text()
    assert "user:pass" not in dumped
    assert "token=abc" not in dumped


def test_instrumentation_sanitize_nested_redacts_credentials() -> None:
    """Probe/CUDA diagnostic write boundaries need the same sanitizer as the harness."""
    payload = {"errors": ["opened rtsp://user:pass@cam.local/stream?token=abc"]}

    sanitized = harness.instrumentation._sanitize_nested(payload)

    dumped = json.dumps(sanitized)
    assert "user:pass" not in dumped
    assert "token=abc" not in dumped


def test_sanitize_nested_redacts_credentials_in_nested_error_diagnostics() -> None:
    payload = {
        "repeats": [
            {
                "errors": [
                    "failed to open rtsp://user:pass@cam.local/stream?token=abc: "
                    "timeout"
                ],
                "decoders": ["opened https://svc:key@cdn.example.com/x?signature=zzz"],
            }
        ]
    }

    sanitized = harness._sanitize_nested(payload)

    dumped = json.dumps(sanitized)
    assert "user:pass" not in dumped
    assert "token=abc" not in dumped
    assert "svc:key" not in dumped
    assert "signature=zzz" not in dumped


@pytest.mark.parametrize(
    "expected, delivered, pending, drained, discard_ids, checks",
    [
        # Warmup ids need not start at 1; complete delivery is clean.
        (range(37, 50), range(37, 50), (), True, (), {"lost": 0, "trailing_lost": 0}),
        # Leading loss: the first completed outputs never reached the sink.
        (range(1, 11), range(3, 11), (), True, (), {"leading_lost": 2, "lost": 2}),
        # Interior loss.
        (range(1, 11), [1, 2, 3, 5, 6, 7, 8, 9, 10], (), True, (), {"interior_lost": 1}),
        # Drained: completed outputs missing at the tail were lost, not "in flight".
        (range(1, 4), [1], (), True, (), {"trailing_lost": 2, "lost": 2}),
        (range(1, 4), [], (), True, (), {"trailing_lost": 3, "lost": 3}),
        # Not drained: the tail is unresolved, neither okay nor proven lost.
        (range(1, 11), range(1, 8), (), False, (), {"trailing_unresolved": 3, "lost": 0}),
        (range(1, 11), range(1, 8), (), None, (), {"trailing_unresolved": 3, "lost": 0}),
        # Trailing ids known to sit in the buffer: pending, not loss.
        (range(1, 11), range(1, 8), [8, 9, 10], True, (), {"trailing_pending": 3, "lost": 0}),
        # Pending evidence also covers ids before the first delivered one.
        (range(1, 6), [3, 4, 5], [1, 2], True, (), {"interior_pending": 2, "lost": 0}),
        # Interior id still buffered (consumer reads out of order): pending.
        (range(1, 6), [1, 2, 4, 5], [3], True, (), {"interior_pending": 1, "lost": 0}),
        # Explicitly observed terminate-time discard: reported, not loss...
        (range(5, 9), range(5, 8), (), True, [8], {"terminate_discarded": 1, "lost": 0}),
        # ...but only the ids actually observed discarded: anything else missing at
        # the tail, even the single newest one, is still lost without observation.
        (range(5, 9), range(5, 7), (), True, [8], {"terminate_discarded": 1, "trailing_lost": 1}),
        (range(5, 9), range(5, 7), (), True, (), {"terminate_discarded": 0, "trailing_lost": 2}),
        # Downstream saw ids upstream never reported: counted, not loss.
        ([2, 3], [1, 2, 3], (), True, (), {"delivered_unexpected": 1, "lost": 0}),
    ],
)  # fmt: skip
def test_analyze_boundary_loss(
    expected, delivered, pending, drained, discard_ids, checks
) -> None:
    result = harness.analyze_boundary_loss(
        expected, delivered, pending, drained, discard_ids
    )

    assert {key: result[key] for key in checks} == checks
    assert result["expected"] == len(set(expected))
    assert result["delivered"] == len(set(delivered))
    assert len(result["lost_ids_sample"]) == min(
        result["lost"], harness.BOUNDARY_ID_SAMPLE
    )


def _probe(**overrides) -> dict:
    probe = {
        "schema_version": 1,
        "consumed_ids": {"0": [3, 4, 5, 6], "1": [3, 4, 5]},
        "completed_ids": {"0": [3, 4, 5, 6], "1": [3, 4, 5]},
        "terminate_discarded_ids": {},
        # [source_id, frame_id, captured_wall_ns, capture_to_sink_ns, workflow_ns]
        "sink_entries": [
            [0, 3, 100 * MS, 4 * MS, 2 * MS],
            [0, 4, 110 * MS, 4 * MS, 2 * MS],
            [0, 5, 120 * MS, 4 * MS, 2 * MS],
            [0, 6, 130 * MS, 4 * MS, 2 * MS],
            [1, 3, 100 * MS, 5 * MS, 2 * MS],
            [1, 4, 110 * MS, 5 * MS, 2 * MS],
            [1, 5, 120 * MS, 5 * MS, 2 * MS],
        ],
        "model_calls": [],
        "model_call_methods": list(harness.instrumentation.MODEL_CALL_METHODS),
        "model_ids": [],
        "model_backends": {},
        "model_input_kinds": {},
        "model_output_kinds": {},
        "dropped_events": {},
        "decoders": ["CV2VideoFrameProducer", "CV2VideoFrameProducer"],
        "pending_at_termination": [],
        "drained": True,
        "errors": [],
    }
    probe.update(overrides)
    return probe


def test_build_boundary_report_separates_sink_overflow_from_loss() -> None:
    # Source 0: consumer missed frame 4 (memory sink overflow) and frame 6 was still
    # buffered at termination. Source 1: frame 3 completed but never entered the sink.
    probe = _probe(
        sink_entries=[e for e in _probe()["sink_entries"] if e[:2] != [1, 3]],
        pending_at_termination=[[0, 6]],
    )

    report = harness.build_boundary_report(probe, {0: [3, 5], 1: [4, 5]}, "manager")

    assert report["available"]
    assert report["stages"]["consumed_to_completed"]["lost"] == 0
    completed_to_sink = report["stages"]["completed_to_sink"]
    assert completed_to_sink["sources"]["1"]["leading_lost"] == 1
    assert completed_to_sink["lost"] == 1
    sink_to_consumer = report["stages"]["sink_to_consumer"]["sources"]
    assert sink_to_consumer["0"]["interior_lost"] == 1
    assert sink_to_consumer["0"]["trailing_pending"] == 1
    assert report["sink_overflow_discards"] == 1
    assert report["lost"] == 1  # overflow is not counted as boundary loss


def test_build_boundary_report_in_process_and_without_probe() -> None:
    # Drained: source 0's completed frame 6 never reached the sink -> lost.
    report = harness.build_boundary_report(
        _probe(), {0: [3, 4, 5], 1: [3, 4, 5]}, "in-process"
    )
    assert set(report["stages"]) == {"consumed_to_completed", "completed_to_sink"}
    assert report["stages"]["completed_to_sink"]["sources"]["0"]["trailing_lost"] == 1
    assert report["lost"] == 1 and report["unresolved"] == 0
    assert report["drained"] is True

    # Not drained (join never returned): the same tail is unresolved, not loss.
    report = harness.build_boundary_report(
        _probe(drained=False), {0: [3, 4, 5], 1: [3, 4, 5]}, "in-process"
    )
    assert report["lost"] == 0 and report["unresolved"] == 1

    assert harness.build_boundary_report(None, {}, "manager") == {
        "available": False,
        "reason": "no probe data",
        "lost": None,
    }


def test_build_boundary_report_drained_tail_loss_and_consumed_only_sources() -> None:
    # consumed = completed = [1, 2, 3], delivered = [1], nothing pending: lost = 2.
    probe = _probe(
        consumed_ids={"0": [1, 2, 3], "2": [7, 8]},
        completed_ids={"0": [1, 2, 3]},
        sink_entries=[[0, 1, None, None, None]],
        # Only frame 8 was actually observed being discarded by a retrieval call
        # that itself returned `None`; frame 7 has no such evidence, so it is loss.
        terminate_discarded_ids={"2": [8]},
    )
    report = harness.build_boundary_report(probe, {0: [1]}, "in-process")

    source_0 = report["stages"]["completed_to_sink"]["sources"]["0"]
    assert source_0["trailing_lost"] == 2 and source_0["lost_ids_sample"] == [2, 3]
    # Source 2 was only ever consumed: accounted, frame 8 is the observed terminate
    # discard, frame 7 (no evidence) is lost.
    source_2 = report["stages"]["consumed_to_completed"]["sources"]["2"]
    assert source_2["terminate_discarded"] == 1 and source_2["lost"] == 1
    assert report["lost"] == 3 and report["terminate_discarded"] == 1


def test_build_boundary_report_consumed_only_source_without_discard_evidence_is_lost() -> (
    None
):
    # Source 2 was only ever consumed (never completed) and no retrieval call
    # reported it as discarded: without the old blanket per-source allowance, both
    # frames are loss, not just the older one.
    probe = _probe(
        consumed_ids={"0": [1, 2, 3], "2": [7, 8]},
        completed_ids={"0": [1, 2, 3]},
        sink_entries=[[0, 1, None, None, None]],
    )
    report = harness.build_boundary_report(probe, {0: [1]}, "in-process")

    source_2 = report["stages"]["consumed_to_completed"]["sources"]["2"]
    assert source_2["terminate_discarded"] == 0 and source_2["lost"] == 2


def test_build_boundary_report_manager_drained_overflow_vs_loss() -> None:
    # Source 0: the consumer stopped at frame 4; frame 5 was then discarded by the
    # memory-sink deque during the drain and frame 6 was still buffered. Source 1:
    # completed frame 5 never entered the sink, which is pipeline loss.
    probe = _probe(
        sink_entries=[e for e in _probe()["sink_entries"] if e[:2] != [1, 5]],
        pending_at_termination=[[0, 6]],
    )
    report = harness.build_boundary_report(probe, {0: [3, 4], 1: [3, 4]}, "manager")

    assert report["stages"]["completed_to_sink"]["sources"]["1"]["trailing_lost"] == 1
    assert report["lost"] == 1
    sink_to_consumer = report["stages"]["sink_to_consumer"]["sources"]["0"]
    assert sink_to_consumer["trailing_lost"] == 1  # deque discard, not loss
    assert sink_to_consumer["trailing_pending"] == 1
    assert report["sink_overflow_discards"] == 1

    # A probe from a child that never finished terminate(): unresolved tails.
    report = harness.build_boundary_report(
        _probe(drained=None, pending_at_termination=[]),
        {0: [3, 4], 1: [3, 4]},
        "manager",
    )
    assert report["lost"] == 0 and report["sink_overflow_discards"] == 0
    assert report["unresolved"] == 3


def test_summarize_model_calls_restricts_to_window_and_names_backends() -> None:
    window = (1000 * MS, 2000 * MS)
    calls = [
        # [start_wall_ns, duration_ns, method_index, model_index, batch, thread_id]
        [900 * MS, 30 * MS, 1, 0, 1, 7],  # before the window
        [1100 * MS, 10 * MS, 1, 0, 1, 7],
        [1200 * MS, 20 * MS, 1, 0, 1, 7],
        [1300 * MS, 40 * MS, 2, 1, 4, 8],
        [2000 * MS, 10 * MS, 1, 0, 1, 7],  # at window end: excluded
    ]
    probe = _probe(
        model_calls=calls,
        model_ids=["coco/36", "<local package>"],
        model_backends={
            "coco/36": "RFDetrObjectDetectionOnnx",
            "<local package>": "TorchModel(YoloV8)",
        },
        model_input_kinds={"ndarray": 3, "torch:cuda": 1},
    )

    summary = harness.summarize_model_calls(probe, *window)

    assert summary["available"]
    assert summary["calls"] == 3 and summary["calls_total"] == 5
    assert summary["p50_ms"] == pytest.approx(20)
    assert summary["busy_fraction"] == pytest.approx(0.07)
    assert summary["threads"] == 2
    assert summary["by_model"]["coco/36 via infer_from_request_sync"]["count"] == 2
    assert (
        summary["by_model"]["coco/36 via infer_from_request_sync"]["backend"]
        == "RFDetrObjectDetectionOnnx"
    )
    tensor = summary["by_model"]["<local package> via run_tensor_native_inference"]
    assert tensor["batch_mean"] == 4 and tensor["backend"] == "TorchModel(YoloV8)"
    assert summary["input_kinds"] == {"ndarray": 3, "torch:cuda": 1}
    assert harness.summarize_model_calls(_probe(), *window) == {
        "available": False,
        "calls": 0,
    }
    assert harness.summarize_model_calls(None, *window)["available"] is False


def test_merge_probe_latencies_uses_child_capture_clock() -> None:
    recorder = harness.Recorder(source_count=1)
    recorder.consumed_wall_ns = {(0, 3): 125 * MS, (0, 4): 140 * MS}
    records = [
        # latency_ns here is the synthetic file-timestamp value the manager reports
        SinkRecord(10 * SECOND, 0, 3, 999 * MS, None, "a"),
        SinkRecord(10 * SECOND + MS, 0, 4, 999 * MS, None, "b"),
        SinkRecord(10 * SECOND + 2 * MS, 0, 9, 999 * MS, None, "c"),  # unknown
    ]

    merged, capture_to_sink = harness.merge_probe_latencies(records, _probe(), recorder)

    assert [r.latency_ns for r in merged] == [25 * MS, 30 * MS, 999 * MS]
    assert [r.workflow_ns for r in merged] == [2 * MS, 2 * MS, None]
    assert capture_to_sink == {(0, 3): 4 * MS, (0, 4): 4 * MS}
    assert harness.merge_probe_latencies(records, None, recorder) == (records, {})


def test_resource_sampler_peaks_use_the_measurement_window_only() -> None:
    """RSS/CPU peaks must share CPU/FPS's steady window: a warmup spike (model load,
    allocator warm-up) outside it must not inflate the reported peak."""
    sampler = harness.ResourceSampler()
    sampler._gpu_available = False
    sampler.samples = [
        {"t_ns": 0, "processes": 1, "cpu_percent": 400.0, "rss_mb": 900.0},
        {"t_ns": 100, "processes": 1, "cpu_percent": 10.0, "rss_mb": 100.0},
        {"t_ns": 200, "processes": 1, "cpu_percent": 12.0, "rss_mb": 110.0},
    ]

    summary = sampler.summarize(window_start_ns=100, window_end_ns=200)

    assert summary["rss_mb_peak"] == 110.0
    assert summary["cpu_percent_mean"] == pytest.approx(11.0)


def test_resource_sampler_gpu_uses_windowed_samples_and_owned_tree_memory() -> None:
    sampler = harness.ResourceSampler()
    sampler._gpu_available = True
    sampler.samples = []
    sampler.gpu_samples = [
        {
            "t_ns": 0,  # co-tenant spike before the window: excluded
            "gpus": [{"index": "0", "memory_used_mb": 9000.0, "utilization_percent": 90.0}],
            "tree_process_memory_mb": {},
        },
        {
            "t_ns": 100,
            "gpus": [{"index": "0", "memory_used_mb": 500.0, "utilization_percent": 20.0}],
            "tree_process_memory_mb": {"111": 300.0},
        },
        {
            "t_ns": 200,
            "gpus": [{"index": "0", "memory_used_mb": 600.0, "utilization_percent": 25.0}],
            "tree_process_memory_mb": {"111": 350.0},
        },
    ]  # fmt: skip

    summary = sampler.summarize(window_start_ns=100, window_end_ns=200)

    assert summary["gpu"]["memory_mb_peak"] == 600.0
    assert summary["gpu"]["tree_process_memory_mb_peak"] == 350.0


def test_resource_sampler_gpu_tree_memory_is_none_when_never_mapped() -> None:
    """Owned process-tree memory is unsupported (None), not a false zero, when
    nvidia-smi never mapped a pid in this window (containers, Jetson)."""
    sampler = harness.ResourceSampler()
    sampler._gpu_available = True
    sampler.samples = []
    sampler.gpu_samples = [
        {
            "t_ns": 100,
            "gpus": [
                {"index": "0", "memory_used_mb": 500.0, "utilization_percent": 20.0}
            ],
            "tree_process_memory_mb": {},
        }
    ]

    summary = sampler.summarize(window_start_ns=0, window_end_ns=200)

    assert summary["gpu"]["tree_process_memory_mb_peak"] is None


def _sample_gpu_with(monkeypatch, apps: str, tgids: dict) -> dict:
    sampler = harness.ResourceSampler()
    sampler._tracked = {10: None, 20: None}  # the harness process tree
    outputs = {
        "--query-gpu=index,memory.used,utilization.gpu": "0, 900, 30",
        "--query-compute-apps=pid,used_memory": apps,
    }
    monkeypatch.setattr(harness, "_run_text", lambda command: outputs[command[1]])
    monkeypatch.setattr(harness, "_thread_group_id", tgids.get)
    sampler._sample_gpu()
    return sampler.gpu_samples[-1]


def test_resource_sampler_gpu_maps_owned_thread_contexts_to_their_process(
    monkeypatch,
) -> None:
    """NVML may list a CUDA context under the id of the thread that created it (the
    manager's pipeline process uses a worker thread). Only thread ids whose /proc
    Tgid is in our tree count, summed into that process; unrelated ids never do."""
    sample = _sample_gpu_with(
        monkeypatch,
        apps="10, 100\n21, 300\n22, 50\n31, 700\n99, 800",
        # 21, 22: threads of tree process 20; 31: thread of another process;
        # 99: exited / unreadable.
        tgids={21: 20, 22: 20, 31: 30},
    )

    assert sample["tree_process_memory_mb"] == {"10": 100.0, "20": 350.0}
    assert sample["tree_thread_contexts"] == {
        "21": {"tgid": 20, "memory_mb": 300.0},
        "22": {"tgid": 20, "memory_mb": 50.0},
    }


def test_resource_sampler_gpu_pid_path_is_unchanged_without_thread_contexts(
    monkeypatch,
) -> None:
    sample = _sample_gpu_with(monkeypatch, apps="10, 100\n31, 700", tgids={31: 30})

    assert sample["tree_process_memory_mb"] == {"10": 100.0}
    assert "tree_thread_contexts" not in sample


def test_resource_sampler_gpu_owned_peak_is_none_when_thread_context_unreported(
    monkeypatch,
) -> None:
    """Driver 550 reports a worker-thread context with 0 MiB: owned memory is then
    unobservable (None + reason), never a false 0 or the device total."""
    sample = _sample_gpu_with(monkeypatch, apps="10, 100\n21, 0", tgids={21: 20})
    assert sample["tree_process_memory_mb"] == {"10": 100.0}
    sampler = harness.ResourceSampler()
    sampler._gpu_available = True
    sampler.gpu_samples = [{**sample, "t_ns": 100}]

    summary = sampler.summarize(window_start_ns=0, window_end_ns=200)

    assert summary["gpu"]["tree_process_memory_mb_peak"] is None
    assert summary["gpu"]["memory_mb_peak"] == 900.0
    assert summary["gpu"]["unreported_tree_thread_contexts"]["thread_to_process"] == {
        "21": 20
    }


def test_recorder_retires_captured_timestamp_on_frame_dropped() -> None:
    """A dropped frame never reaches the sink; its FRAME_CAPTURED timestamp must not
    be retained for the rest of the run."""
    from types import SimpleNamespace

    recorder = harness.Recorder(source_count=1)

    def event(event_type, **payload):
        recorder.on_status_update(
            SimpleNamespace(
                event_type=event_type,
                payload=payload,
                severity=SimpleNamespace(name="DEBUG"),
            )
        )

    event("FRAME_CAPTURED", source_id=0, frame_id=5)
    assert (0, 5) in recorder._captured
    event("FRAME_DROPPED", source_id=0, frame_id=5)
    assert (0, 5) not in recorder._captured


def test_recorder_joins_workflow_time_when_sink_runs_before_completion() -> None:
    """Production enqueues the result before sending INFERENCE_COMPLETED, so the
    dispatcher thread can deliver it first."""
    from types import SimpleNamespace

    recorder = harness.Recorder(source_count=1)

    def event(event_type, **payload):
        recorder.on_status_update(
            SimpleNamespace(
                event_type=event_type,
                payload=payload,
                severity=SimpleNamespace(name="DEBUG"),
            )
        )

    frame = SimpleNamespace(source_id=0, frame_id=3, image=None)
    event("FRAME_CAPTURED", source_id=0, frame_id=3)
    event("FRAME_CONSUMED", source_id=0, frame_id=3)
    recorder.on_prediction([None], [frame])  # sink first
    event("INFERENCE_COMPLETED", sources_id=[0], frames_ids=[3])

    [record] = recorder.snapshot()
    assert record.workflow_ns is not None and record.workflow_ns >= 0
    assert record.latency_ns is not None


def test_boundary_probe_joins_workflow_time_when_sink_runs_before_completion() -> None:
    from types import SimpleNamespace

    probe = harness.instrumentation.BoundaryProbe()
    sink = probe.wrap_sink(lambda predictions, frames: None)

    def event(event_type, **payload):
        probe.on_status_update(SimpleNamespace(event_type=event_type, payload=payload))

    event("FRAME_CAPTURED", source_id=0, frame_id=3)
    event("FRAME_CONSUMED", source_id=0, frame_id=3)
    sink(None, SimpleNamespace(source_id=0, frame_id=3))  # sink first
    event("INFERENCE_COMPLETED", sources_id=[0], frames_ids=[3])

    [entry] = probe.export()["sink_entries"]
    assert entry[4] is not None and entry[4] >= 0
    assert probe.export()["drained"] is None  # termination never observed


def test_boundary_probe_retires_captured_timestamp_on_frame_dropped() -> None:
    from types import SimpleNamespace

    probe = harness.instrumentation.BoundaryProbe()

    def event(event_type, **payload):
        probe.on_status_update(SimpleNamespace(event_type=event_type, payload=payload))

    event("FRAME_CAPTURED", source_id=0, frame_id=5)
    assert (0, 5) in probe._captured
    event("FRAME_DROPPED", source_id=0, frame_id=5)
    assert (0, 5) not in probe._captured


def test_describe_model_marks_registry_ids_as_unverified(tmp_path) -> None:
    assert harness.describe_model("coco/36")["weights_verified"] is False

    package = tmp_path / "package"
    package.mkdir()
    (package / "weights.onnx").write_bytes(b"abc")

    assert harness.describe_model(str(package))["weights_verified"] is True


def test_describe_model_hashes_nested_weights_directories(tmp_path) -> None:
    """Real local packages keep weights nested (e.g. `base/weights.onnx`): only
    hashing top-level files would miss them while still claiming verification."""
    package = tmp_path / "package"
    (package / "base").mkdir(parents=True)
    (package / "base" / "weights.onnx").write_bytes(b"abc")
    (package / "model_config.json").write_text("{}")

    description = harness.describe_model(str(package))

    assert description["weights_verified"] is True
    assert description["files_sha256"][os.path.join("base", "weights.onnx")]
    assert "model_config.json" in description["files_sha256"]


def test_describe_model_does_not_claim_verified_without_file_evidence(tmp_path) -> None:
    package = tmp_path / "empty_package"
    package.mkdir()

    description = harness.describe_model(str(package))

    assert description["weights_verified"] is False
    assert description["package_sha256"] is None


def test_compare_fails_for_unverifiable_registry_weights_when_model_calls_occurred() -> (
    None
):
    baseline = _result()
    baseline["model"] = {"weights_verified": False}
    baseline["repeats"][0]["instrumentation"] = {"model_calls": {"available": True}}
    candidate = _result(fps=98.0)
    candidate["model"] = {"weights_verified": True}

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("weights could not be verified" in f for f in comparison["failures"])


def test_compare_fails_closed_when_weights_verified_key_is_missing() -> None:
    """A missing `weights_verified` key (older/partial artifact) must not default to
    an implicit pass when model calls occurred."""
    baseline = _result()
    baseline["model"] = {}
    baseline["repeats"][0]["instrumentation"] = {"model_calls": {"available": True}}
    candidate = _result(fps=98.0)
    candidate["model"] = {}

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("weights could not be verified" in f for f in comparison["failures"])


def test_compare_allows_unverified_weights_when_no_model_calls_occurred() -> None:
    """A no-model CI smoke legitimately has no weights to verify."""
    baseline = _result()
    baseline["model"] = {"weights_verified": False}
    candidate = _result(fps=98.0)
    candidate["model"] = {"weights_verified": False}

    comparison = harness.compare_results(baseline, candidate)

    assert comparison["passed"], comparison["failures"]


def test_compare_fails_when_candidate_rss_growth_outpaces_baseline() -> None:
    baseline = _result()
    candidate = _result(fps=98.0)
    for repeat in baseline["repeats"]:
        repeat["metrics"]["rss_mb_slope_per_min"] = 2.0  # the harness's own retention
    for repeat in candidate["repeats"]:
        repeat["metrics"]["rss_mb_slope_per_min"] = 20.0  # candidate-side leak

    comparison = harness.compare_results(baseline, candidate)

    assert not comparison["passed"]
    assert any("rss_mb_slope_per_min grew" in f for f in comparison["failures"])


def test_compare_tolerates_shared_retention_growth_in_rss_slope() -> None:
    """Both sides accrue comparable growth from the harness's own retained
    per-frame records; only a materially larger gap indicates a leak."""
    baseline = _result()
    candidate = _result(fps=98.0)
    for repeat in baseline["repeats"]:
        repeat["metrics"]["rss_mb_slope_per_min"] = 3.0
    for repeat in candidate["repeats"]:
        repeat["metrics"]["rss_mb_slope_per_min"] = 4.0

    comparison = harness.compare_results(baseline, candidate)

    assert comparison["passed"], comparison["failures"]


def test_main_compare_prints_absolute_slope_metric_without_crashing(
    tmp_path, capsys
) -> None:
    """Regression: `rss_mb_slope_per_min` is reported with `absolute_delta_mb_per_min`
    / `band_mb_per_min`, not `relative_delta` / `band` / `paired_relative_ci95` like
    every other metric, so `print_comparison` must not unconditionally index those
    keys - it used to crash with a KeyError on any real comparison that included
    this metric. This runs `main(--compare ... --output ...)` end to end against two
    real-shaped result artifacts on disk; it does not run any pipeline."""
    baseline = _result()
    candidate = _result(fps=98.0)
    for repeat in baseline["repeats"]:
        repeat["metrics"]["rss_mb_slope_per_min"] = 3.0
    for repeat in candidate["repeats"]:
        repeat["metrics"]["rss_mb_slope_per_min"] = 4.0
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))

    exit_code = harness.main(
        ["--compare", str(baseline_path), "--output", str(candidate_path)]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "rss_mb_slope_per_min" in out
    assert "delta=+1.00MB/min" in out
    assert "band=5.0MB/min" in out


@pytest.mark.parametrize(
    "platform_name,cuda_available,expected",
    [
        ("darwin", True, "skipped:not-linux"),
        ("linux", False, "skipped:cuda-unavailable"),
        ("linux", True, "initialized"),
    ],
)
def test_maybe_init_cuda_on_child_main_thread_guards_platform_and_availability(
    monkeypatch, platform_name, cuda_available, expected
) -> None:
    """Only calls `torch.cuda.init()` on Linux with CUDA available; otherwise it
    must not even import torch (macOS dev boxes / CPU-only hosts have neither)."""
    from types import ModuleType

    instrumentation = harness.instrumentation
    monkeypatch.setattr(instrumentation.sys, "platform", platform_name)
    calls = []
    if platform_name.startswith("linux"):
        fake_cuda = ModuleType("torch.cuda")
        fake_cuda.is_available = lambda: cuda_available
        fake_cuda.init = lambda: calls.append("init")
        fake_torch = ModuleType("torch")
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)
    else:
        monkeypatch.delitem(sys.modules, "torch", raising=False)

    status = instrumentation._maybe_init_cuda_on_child_main_thread()

    assert status == expected
    assert calls == (["init"] if expected == "initialized" else [])


def test_instrumented_manager_run_initializes_cuda_before_worker_starts(
    monkeypatch,
) -> None:
    """`run()` executes in the forked/spawned pipeline child (`InferencePipelineManager`
    is itself a `multiprocessing.Process`); the CUDA init must happen there, before
    `super().run()` can start `InferencePipeline`'s inference worker thread, and its
    outcome must land in the exported probe for the fingerprint."""
    instrumentation = harness.instrumentation
    order = []
    monkeypatch.setattr(
        instrumentation,
        "_maybe_init_cuda_on_child_main_thread",
        lambda: order.append("cuda_init") or "initialized",
    )
    monkeypatch.setattr(
        instrumentation, "activate", lambda probe: order.append("activate")
    )
    monkeypatch.setattr(
        instrumentation,
        "_install_pipeline_hooks",
        lambda probe: order.append("install_hooks"),
    )
    monkeypatch.setattr(
        instrumentation.InferencePipelineManager,
        "run",
        lambda self: order.append("super_run"),
    )

    manager = instrumentation.InstrumentedInferencePipelineManager(
        pipeline_id="test-pipeline",
        command_queue=None,
        responses_queue=None,
    )
    manager.run()

    assert order == ["cuda_init", "activate", "install_hooks", "super_run"]
    assert (
        manager._benchmark_probe.export()["cuda_child_main_thread_init"]
        == "initialized"
    )


def test_boundary_probe_times_provider_model_calls_and_names_backend() -> None:
    from inference.core.interfaces.workflows_models_provider import (
        ModelManagerModelsProvider,
    )

    instrumentation = harness.instrumentation
    np = pytest.importorskip("numpy")

    class FakeModel:
        pass

    class FakeInnerManager:
        _models = {"coco/36": FakeModel()}

    class FakeManager:
        model_manager = FakeInnerManager()

        def infer_from_request_sync(self, model_id, request, **kwargs):
            return [{"predictions": []}]

        def run_tensor_native_inference(self, model_id, **kwargs):
            return np.zeros((1, 4))

    class Request:
        image = [{"type": "numpy_object", "value": np.zeros((2, 2, 3), np.uint8)}]

        def model_dump(self):
            return {}

    provider = ModelManagerModelsProvider(FakeManager())
    probe = instrumentation.BoundaryProbe()
    instrumentation.activate(probe)
    try:
        assert provider._infer("coco/36", Request()) == [{"predictions": []}]
        provider.run_tensor_native_inference(
            "coco/36", images=[np.zeros((2, 2, 3)), np.zeros((2, 2, 3))]
        )
    finally:
        instrumentation.deactivate()
    provider._infer("coco/36", Request())  # inactive probe: not recorded

    exported = probe.export()
    assert exported["model_ids"] == ["coco/36"]
    assert exported["model_backends"] == {"coco/36": "FakeModel"}
    methods = [exported["model_call_methods"][c[2]] for c in exported["model_calls"]]
    # `_infer` calls the model manager directly, so exactly one record per call.
    assert methods == ["_infer", "run_tensor_native_inference"]
    assert [c[4] for c in exported["model_calls"]] == [1, 2]
    assert all(c[1] >= 0 for c in exported["model_calls"])
    assert exported["model_input_kinds"] == {"numpy_object:ndarray": 1, "ndarray": 1}
    assert exported["model_output_kinds"] == {"dict": 1, "ndarray": 1}
    assert exported["errors"] == []
    assert instrumentation.public_model_id(os.getcwd()) == "<local package>"


def test_boundary_probe_tracks_status_events_and_sink_entries() -> None:
    from types import SimpleNamespace

    probe = harness.instrumentation.BoundaryProbe()
    delivered = []
    sink = probe.wrap_sink(lambda predictions, frames: delivered.append(frames))

    def event(event_type, **payload):
        probe.on_status_update(SimpleNamespace(event_type=event_type, payload=payload))

    for frame_id in (7, 8):
        event("FRAME_CAPTURED", source_id=0, frame_id=frame_id, frame_timestamp=0.0)
        event("FRAME_CONSUMED", source_id=0, frame_id=frame_id)
    event("INFERENCE_COMPLETED", sources_id=[0, 0], frames_ids=[7, 8])
    event("FRAME_DROPPED", source_id=0, frame_id=9)
    sink(None, [SimpleNamespace(source_id=0, frame_id=7), None])
    probe.note_pending(
        SimpleNamespace(_buffer=[(None, [SimpleNamespace(source_id=0, frame_id=8)])])
    )
    probe.on_status_update(object())  # malformed: recorded, never raised

    exported = probe.export()
    assert exported["consumed_ids"] == {"0": [7, 8]}
    assert exported["completed_ids"] == {"0": [7, 8]}
    [(source_id, frame_id, captured_wall, capture_to_sink, workflow)] = exported[
        "sink_entries"
    ]
    assert (source_id, frame_id) == (0, 7)
    assert captured_wall > 0 and capture_to_sink >= 0 and workflow >= 0
    assert exported["pending_at_termination"] == [[0, 8]]
    assert exported["dropped_events"] == {"0": 1}
    assert len(delivered) == 1
    assert exported["errors"] and "status update probe failed" in exported["errors"][0]


def test_record_retrieval_only_flags_ids_consumed_during_a_discarded_call() -> None:
    """Unit-level: the wrapper's own FRAME_CONSUMED bookkeeping, isolated from any
    real `VideoSourcesManager`."""
    from types import SimpleNamespace

    probe = harness.instrumentation.BoundaryProbe()

    def consume(source_id, frame_id) -> None:
        probe.on_status_update(
            SimpleNamespace(
                event_type="FRAME_CONSUMED",
                payload={"source_id": source_id, "frame_id": frame_id},
            )
        )

    # A successful call (returns a batch, not None): nothing is a discard, even
    # though frames were consumed.
    result = probe.record_retrieval(lambda: (consume(0, 1), [object()])[1])
    assert result is not None
    assert probe.export()["terminate_discarded_ids"] == {}

    # A call that consumes frames from two sources and then discards the whole
    # batch on stop (returns None): both consumed ids for this call are flagged.
    def discarded_call():
        consume(0, 2)
        consume(1, 5)
        return None

    assert probe.record_retrieval(discarded_call) is None
    assert probe.export()["terminate_discarded_ids"] == {"0": [2], "1": [5]}

    # FRAME_CONSUMED ids raised outside any wrapped call are never misattributed
    # as a discard.
    consume(0, 3)
    assert probe.export()["terminate_discarded_ids"] == {"0": [2], "1": [5]}


def _actual_video_sources_manager(sources, should_stop):
    from inference.core.interfaces.camera.utils import VideoSources, VideoSourcesManager

    return VideoSourcesManager.init(
        video_sources=VideoSources(
            all_sources=sources, allow_reconnection=[False] * len(sources), managed_sources=[]
        ),
        should_stop=should_stop,
        on_reconnection_error=lambda *a, **k: None,
    )  # fmt: skip


class _FakeSource:
    """Minimal stand-in for `VideoSource.read_frame`, including its FRAME_CONSUMED
    status update, so the real `VideoSourcesManager` loop can be exercised."""

    def __init__(self, probe, source_id, frame_id, stop_after_read=None):
        self._probe = probe
        self.source_id = source_id
        self.frame_id = frame_id
        self._stop_after_read = stop_after_read

    def read_frame(self, timeout=None):
        from types import SimpleNamespace

        self._probe.on_status_update(
            SimpleNamespace(
                event_type="FRAME_CONSUMED",
                payload={"source_id": self.source_id, "frame_id": self.frame_id},
            )
        )
        if self._stop_after_read is not None:
            self._stop_after_read()
        return SimpleNamespace(source_id=self.source_id, frame_id=self.frame_id)


def test_actual_multi_source_partial_collection_stop_records_precise_discarded_ids() -> (
    None
):
    """The real `VideoSourcesManager.retrieve_frames_from_sources`: source 0 is
    read (FRAME_CONSUMED fires) and then the stop signal arrives before source 1 is
    reached, so the whole batch - including the already-consumed frame - is
    discarded. Only that exact id is a terminate-time discard."""
    probe = harness.instrumentation.BoundaryProbe()
    stop = {"flag": False}
    source_0 = _FakeSource(
        probe, source_id=0, frame_id=11, stop_after_read=lambda: stop.update(flag=True)
    )
    source_1 = _FakeSource(probe, source_id=1, frame_id=21)
    manager = _actual_video_sources_manager(
        [source_0, source_1], should_stop=lambda: stop["flag"]
    )

    result = probe.record_retrieval(
        lambda: manager.retrieve_frames_from_sources(batch_collection_timeout=None)
    )

    assert result is None
    assert probe.export()["terminate_discarded_ids"] == {"0": [11]}


def test_actual_multi_source_partial_collection_stop_records_precise_discarded_ids_with_policy() -> (
    None
):
    """Same real discard as above, through
    `retrieve_frames_from_sources_with_policy` (`VideoProcessingMode.EVERY_FRAME`
    skips the staleness path, so the read goes straight to `source.read_frame`)."""
    from inference.core.interfaces.camera.collection_policy import (
        CollectionPolicy,
        VideoProcessingMode,
    )

    probe = harness.instrumentation.BoundaryProbe()
    stop = {"flag": False}
    source_0 = _FakeSource(
        probe, source_id=0, frame_id=12, stop_after_read=lambda: stop.update(flag=True)
    )
    source_1 = _FakeSource(probe, source_id=1, frame_id=22)
    manager = _actual_video_sources_manager(
        [source_0, source_1], should_stop=lambda: stop["flag"]
    )
    policy = CollectionPolicy(mode=VideoProcessingMode.EVERY_FRAME)

    result = probe.record_retrieval(
        lambda: manager.retrieve_frames_from_sources_with_policy(policy)
    )

    assert result is None
    assert probe.export()["terminate_discarded_ids"] == {"0": [12]}


def test_single_source_missing_tail_remains_loss_without_discard_evidence() -> None:
    """The real single-source collector can return the read frame even when stop
    becomes true during the read (no early per-source loop check discards it), so
    the frame proceeds and any later gap has no retrieval-level discard evidence:
    it must still be classified as loss, never silently exempted."""
    report = harness.build_boundary_report(
        {
            "drained": True,
            "consumed_ids": {"0": [1, 2]},
            "completed_ids": {"0": [1]},
            "terminate_discarded_ids": {},
            "sink_entries": [],
            "pending_at_termination": [],
            "errors": [],
        },
        {0: [1]},
        "in-process",
    )

    source_0 = report["stages"]["consumed_to_completed"]["sources"]["0"]
    assert source_0["terminate_discarded"] == 0
    assert source_0["lost"] == 1


def _trace_event(cat, name, tid, ts, dur=1, **args) -> dict:
    return {"ph": "X", "cat": cat, "name": name, "tid": tid, "ts": ts, "dur": dur,
            "args": args}  # fmt: skip


def _cuda_trace_events() -> list:
    return [
        _trace_event("user_annotation", "benchmark.model_call", 100, 0, 100,
                     **{"External id": 1}),
        _trace_event("cpu_op", "aten::copy_", 100, 10, 5, **{"External id": 2}),
        _trace_event("cuda_runtime", "cudaMemcpyAsync", 100, 11,
                     correlation=50, **{"External id": 2}),
        _trace_event("gpu_memcpy", "Memcpy HtoD (Pageable -> Device)", 7, 20,
                     correlation=50, bytes=1000),
        _trace_event("kernel", "some_kernel", 7, 30),
        # Another thread copies while the model call runs: time overlap only.
        _trace_event("cuda_runtime", "cudaMemcpyAsync", 200, 50, correlation=51),
        _trace_event("gpu_memcpy", "Memcpy DtoH (Device -> Pageable)", 7, 55,
                     correlation=51, bytes=10),
    ]  # fmt: skip


def _summarize_trace(events, **overrides) -> dict:
    meta = dict(
        owner_tid=100,
        profiled_calls=1,
        other_thread_calls=0,
        other_tids=set(),
        cuda_traced=True,
        state="stopped",
        errors=[],
    )
    meta.update(overrides)
    return harness.instrumentation.summarize_cuda_trace(events, **meta)


def test_cuda_trace_attributes_copies_by_launch_not_time_overlap() -> None:
    summary = _summarize_trace(_cuda_trace_events())

    copies = summary["device_copies"]
    assert copies["HtoD"] == {"model_call": {"count": 1, "bytes": 1000}}
    # Overlaps the model range in time, launched by a non-owner thread the trace
    # never registered a model call for. Absent a call registry there is no way to
    # tell "this thread made no model call" from "this thread's model call predates
    # arming and was never registered" apart, so it is unattributed, not asserted
    # outside - which also makes the trace unproven, never falsely complete.
    assert copies["DtoH"] == {"unattributed": {"count": 1, "bytes": 10}}
    assert summary["status"] == "unproven"
    assert any("without launch attribution" in r for r in summary["reasons"])
    assert summary["model_call_host_device_copies"] is None
    assert summary["cpu_copy_ops"] == {
        "aten::copy_": {"inside_model_call": 1, "outside_model_call": 0}
    }
    assert summary["copy_samples"][0] == ["HtoD", 1000, "model_call", "aten::copy_"]
    assert summary["input_host_round_trip"] == "unknown"


def test_cuda_trace_owner_launch_outside_its_model_range_is_outside_model_call() -> (
    None
):
    """`outside_model_call` still applies to the one case it can be proven: a copy
    launched by the owner (profiled) thread itself, outside its recorded model-call
    range - never to a different, unregistered thread."""
    events = [
        _trace_event(
            "user_annotation", "benchmark.model_call", 100, 0, 10, **{"External id": 1}
        ),
        _trace_event("cpu_op", "aten::copy_", 100, 50, 5, **{"External id": 2}),
        _trace_event(
            "cuda_runtime", "cudaMemcpyAsync", 100, 51, correlation=60,
            **{"External id": 2},
        ),
        _trace_event(
            "gpu_memcpy", "Memcpy HtoD (Pageable -> Device)", 7, 60,
            correlation=60, bytes=5,
        ),
        _trace_event("kernel", "some_kernel", 7, 65),
    ]  # fmt: skip

    summary = _summarize_trace(events)

    assert summary["device_copies"]["HtoD"] == {
        "outside_model_call": {"count": 1, "bytes": 5}
    }


def test_cuda_trace_another_thread_model_call_predating_arming_stays_unproven() -> None:
    """Regression: a model call on a non-owner thread that started before the trace
    was armed (so `record_model_call`'s snapshot never routed it through the trace,
    and it is absent from `other_tids`/`other_thread_calls`) must not have its
    correlated device copy misattributed as `outside_model_call`, which would make
    the trace falsely `complete` with a misleadingly exact model-call copy count."""
    events = [
        _trace_event(
            "user_annotation", "benchmark.model_call", 100, 100, 50, **{"External id": 1}
        ),
        _trace_event("cpu_op", "aten::copy_", 100, 110, 5, **{"External id": 2}),
        _trace_event(
            "cuda_runtime", "cudaMemcpyAsync", 100, 111, correlation=70,
            **{"External id": 2},
        ),
        _trace_event(
            "gpu_memcpy", "Memcpy HtoD (Pageable -> Device)", 7, 120,
            correlation=70, bytes=1000,
        ),
        _trace_event("kernel", "some_kernel", 7, 130),
        # Thread 300's model call started before the trace was armed: neither the
        # owner nor a registered "other" thread, but it is genuinely mid-call.
        _trace_event("cuda_runtime", "cudaMemcpyAsync", 300, 115, correlation=71),
        _trace_event(
            "gpu_memcpy", "Memcpy DtoH (Device -> Pageable)", 7, 116,
            correlation=71, bytes=10,
        ),
    ]  # fmt: skip

    summary = _summarize_trace(events, other_thread_calls=0, other_tids=set())

    assert summary["device_copies"]["DtoH"] == {
        "unattributed": {"count": 1, "bytes": 10}
    }
    assert summary["status"] == "unproven"
    assert summary["model_call_host_device_copies"] is None


@pytest.mark.parametrize(
    "events_filter, overrides, reason",
    [
        # Model calls also ran on a thread that was not profiled.
        (None, {"other_thread_calls": 3, "other_tids": {200}}, "unprofiled threads"),
        # The profiled thread recorded no model-call range (the L4 failure).
        ("benchmark.model_call", {}, "no model-call range"),
        (None, {"profiled_calls": 2}, "1 model-call ranges for 2 profiled calls"),
        (None, {"cuda_traced": False}, "CUDA activity not traced"),
        (None, {"state": "not_started"}, "trace not_started"),
        # A copy with no launch record cannot be attributed.
        ("cuda_runtime", {}, "without launch attribution"),
    ],
)
def test_cuda_trace_incomplete_attribution_is_unproven(
    events_filter, overrides, reason
) -> None:
    events = [
        e for e in _cuda_trace_events() if events_filter not in (e["cat"], e["name"])
    ]
    summary = _summarize_trace(events, **overrides)

    assert summary["status"] == "unproven"
    assert any(reason in r for r in summary["reasons"]), summary["reasons"]
    assert summary["model_call_host_device_copies"] is None


def test_cuda_trace_without_gpu_activity_is_unproven() -> None:
    events = [
        e for e in _cuda_trace_events() if e["cat"] not in ("kernel", "gpu_memcpy")
    ]
    summary = _summarize_trace(events)
    assert summary["status"] == "unproven"
    assert "no GPU activity recorded" in summary["reasons"]


def test_cuda_trace_fails_open_when_owner_thread_ids_not_verified() -> None:
    """Owner thread ID verification is required for both status and host-transfer
    coverage, even with zero copies or only DtoD copies that pose no host-transfer
    risk. A regression for when model calls run on thread 999 but owner_tid=100."""
    events = [
        _trace_event(
            "user_annotation", "benchmark.model_call", 999, 0, 100, **{"External id": 1}
        ),
        _trace_event("kernel", "some_kernel", 7, 30),
        _trace_event("gpu_memcpy", "Memcpy DtoD (Device -> Device)", 7, 40, bytes=1000),
    ]

    summary = _summarize_trace(events, owner_tid=100)

    assert summary["launch_thread_ids_verified"] is False
    assert summary["status"] == "unproven"
    assert summary["host_transfer_coverage_status"] == "unproven"
    assert summary["model_call_host_device_copies"] is None
    assert any("launch thread IDs not verified" in r for r in summary["reasons"])
    assert any(
        "launch thread IDs not verified" in r
        for r in summary["host_transfer_coverage_reasons"]
    )


def test_cuda_trace_unattributed_dtod_leaves_host_transfer_coverage_complete() -> None:
    """Regression for the real L4 run: 99 profiled calls/ranges, no other-model
    threads, positive GPU activity, no errors, every HtoD/DtoH launch attributed -
    but 98 DtoD copies (frame-sized `pool[...].clone()` calls on the producer
    thread in `PreallocatedCudaFrameProducer.retrieve()`) have no launch record.
    DtoD never reaches host memory, so it must not block the host-transfer claim,
    even though it still leaves the full device-copy `status` unproven."""
    # Base fixture events, minus the launch-less DtoH copy from the unregistered
    # thread (200): here every HtoD/DtoH launch is attributed, and only a DtoD
    # copy (no correlated launch, like the producer-thread frame clone) is not.
    events = [
        e
        for e in _cuda_trace_events()
        if not (e["name"] == "cudaMemcpyAsync" and e["tid"] == 200)
        and "Memcpy DtoH" not in e.get("name", "")
    ] + [
        _trace_event(
            "gpu_memcpy", "Memcpy DtoD (Device -> Device)", 9, 40, bytes=2764800
        ),
    ]
    summary = _summarize_trace(events)

    assert summary["device_copies"]["DtoD"] == {
        "unattributed": {"count": 1, "bytes": 2764800}
    }
    assert summary["status"] == "unproven"
    assert any("without launch attribution" in r for r in summary["reasons"])
    assert summary["host_transfer_coverage_status"] == "complete"
    assert summary["host_transfer_coverage_reasons"] == []
    assert summary["model_call_host_device_copies"] == {
        "count": 1,
        "bytes": 1000,
        "per_model_call": 1.0,
    }


def test_cuda_trace_unattributed_dtoh_blocks_host_transfer_coverage() -> None:
    """Unlike DtoD, an unattributed DtoH could be a model input/output host
    round-trip, so it must keep `host_transfer_coverage_status` unproven too."""
    summary = _summarize_trace(_cuda_trace_events())

    assert summary["status"] == "unproven"
    assert summary["host_transfer_coverage_status"] == "unproven"
    assert any(
        "without launch attribution" in r
        for r in summary["host_transfer_coverage_reasons"]
    )
    assert summary["model_call_host_device_copies"] is None


def test_cuda_trace_unattributed_unknown_direction_blocks_host_transfer_coverage() -> (
    None
):
    """A copy whose kind could not even be determined ("other") is treated as
    potentially host-facing, so it must block `host_transfer_coverage_status`."""
    events = [
        e
        for e in _cuda_trace_events()
        if e["name"] != "cudaMemcpyAsync" or e["tid"] != 200
    ]
    events = [e for e in events if "Memcpy DtoH" not in e.get("name", "")]
    events.append(
        _trace_event("gpu_memcpy", "Memcpy PtoP (Device -> Device)", 9, 40, bytes=5)
    )

    summary = _summarize_trace(events)

    assert summary["device_copies"]["PtoP"] == {
        "unattributed": {"count": 1, "bytes": 5}
    }
    assert summary["host_transfer_coverage_status"] == "unproven"


@pytest.mark.parametrize(
    "events_filter, overrides, reason",
    [
        (None, {"other_thread_calls": 3, "other_tids": {200}}, "unprofiled threads"),
        ("benchmark.model_call", {}, "no model-call range"),
        (None, {"cuda_traced": False}, "CUDA activity not traced"),
        (None, {"state": "not_started"}, "trace not_started"),
    ],
)
def test_cuda_trace_no_positive_coverage_blocks_host_transfer_coverage_too(
    events_filter, overrides, reason
) -> None:
    """Positive model-range/GPU/thread coverage is a prerequisite for the narrower
    host-transfer claim as well; only DtoD/Memset-only unattribution is exempt."""
    events = [
        e for e in _cuda_trace_events() if events_filter not in (e["cat"], e["name"])
    ]
    summary = _summarize_trace(events, **overrides)

    assert summary["status"] == "unproven"
    assert summary["host_transfer_coverage_status"] == "unproven"
    assert any(reason in r for r in summary["host_transfer_coverage_reasons"]), summary[
        "host_transfer_coverage_reasons"
    ]
    assert summary["model_call_host_device_copies"] is None


def test_cuda_trace_profiles_in_the_model_call_thread() -> None:
    """The profiler must start in the thread issuing model calls: started in
    another thread, it records none of that thread's ranges."""
    import threading

    torch = pytest.importorskip("torch")
    trace = harness.instrumentation.CudaCopyTrace(seconds=0.0)
    results = []

    def worker():
        for _ in range(3):
            results.append(trace.run_model_call(lambda: torch.ones(4).add(1).sum()))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=60)
    assert trace.stopped.is_set() and len(results) == 3

    summary = trace.summarize()
    assert summary["trace_state"] == "stopped"
    assert summary["model_calls_profiled"] == 1  # stopped after the first call
    assert summary["model_call_ranges"] == 1
    assert summary["launch_thread_ids_verified"]
    if not torch.cuda.is_available():
        assert summary["status"] == "unproven"


def test_cuda_trace_counts_model_calls_on_other_threads() -> None:
    import threading

    pytest.importorskip("torch")
    trace = harness.instrumentation.CudaCopyTrace(seconds=60.0)
    inside = threading.Event()
    release = threading.Event()

    def owner_call():
        inside.set()
        release.wait(10)

    owner = threading.Thread(target=lambda: trace.run_model_call(owner_call))
    owner.start()
    assert inside.wait(30)
    other = threading.Thread(target=lambda: trace.run_model_call(lambda: None))
    other.start()
    other.join(10)
    trace.cancel()  # stops at the end of the owner's running call
    release.set()
    owner.join(30)

    summary = trace.summarize()
    assert summary["model_calls_other_threads"] == 1
    assert summary["status"] == "unproven"


def test_cuda_trace_cancelled_before_any_model_call_never_starts() -> None:
    pytest.importorskip("torch")
    trace = harness.instrumentation.CudaCopyTrace(seconds=1.0)
    trace.cancel()
    assert trace.run_model_call(lambda: 5) == 5
    summary = trace.summarize()
    assert summary["trace_state"] == "not_started"
    assert summary["status"] == "unproven" and summary["model_calls_profiled"] == 0


def test_describe_model_boundary_value() -> None:
    describe = harness.instrumentation.describe_model_boundary_value
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")

    assert describe(np.zeros((2, 2, 3))) == ("ndarray", 1)
    assert describe(torch.zeros(2, 3, 4, 4)) == ("torch:cpu", 2)
    assert describe([torch.zeros(3, 4, 4)] * 3) == ("torch:cpu", 3)
    assert describe({"type": "numpy_object", "value": np.zeros(1)}) == (
        "numpy_object:ndarray",
        1,
    )
    assert describe(None) == ("none", 0)
    assert describe([]) == ("empty", 0)


def test_parse_args_cuda_trace_is_in_process_diagnostic_only(tmp_path) -> None:
    workflow, source = tmp_path / "w.json", tmp_path / "s.mp4"
    workflow.write_text("{}")
    source.write_bytes(b"")
    base = [
        "--workflow",
        str(workflow),
        "--source",
        str(source),
        "--output",
        str(tmp_path / "o"),
    ]
    args = harness.parse_args(base + ["--cuda-trace", "--cuda-trace-seconds", "2"])
    assert args.cuda_trace and args.cuda_trace_seconds == 2.0
    with pytest.raises(SystemExit):
        harness.parse_args(base + ["--transport", "manager", "--cuda-trace"])
    with pytest.raises(SystemExit):
        harness.parse_args(
            base + ["--duration", "1", "--cuda-trace", "--cuda-trace-seconds", "5"]
        )


# ---------------------------------------------------------------------------------
# Real runtime smokes (bounded: tiny synthetic video, no model)
# ---------------------------------------------------------------------------------


def _write_smoke_video(path, frames: int) -> str:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
    writer.release()
    return str(path)


def _write_no_model_workflow(path) -> str:
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "inputs": [{"type": "InferenceImage", "name": "image"}],
                "steps": [
                    {
                        "type": "roboflow_core/property_definition@v1",
                        "name": "height",
                        "data": "$inputs.image",
                        "operations": [
                            {"type": "ExtractImageProperty", "property_name": "height"}
                        ],
                    }
                ],
                "outputs": [
                    {
                        "type": "JsonField",
                        "name": "height",
                        "selector": "$steps.height.output",
                    }
                ],
            }
        )
    )
    return str(path)


def _run_harness(arguments, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, harness.__file__, *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "DISABLE_VERSION_CHECK": "True"},
    )


def _manager_leftovers():
    """Scratch dirs and instrumented manager processes that a run must not leave."""
    psutil = pytest.importorskip("psutil")
    scratch = {
        name
        for name in os.listdir(tempfile.gettempdir())
        if name.startswith("stream-manager-benchmark-")
    }
    processes = set()
    for process in psutil.process_iter(["pid", "cmdline"]):
        # Match the actual module argv/executable, not a substring of the whole
        # joined command line: that previously matched an unrelated shell/prompt
        # argument that happened to contain this text (e.g. the parent Claude Code
        # process), not a real leftover instrumented pipeline process.
        argv = process.info["cmdline"] or []
        if any(
            arg == "development.stream_interface.benchmark_instrumentation"
            or arg.endswith(
                os.path.join("stream_interface", "benchmark_instrumentation.py")
            )
            for arg in argv
        ):
            processes.add(process.info["pid"])
    return scratch, processes


def test_real_in_process_pipeline_smoke(tmp_path) -> None:
    """Brief no-model run through the real pipeline: capture, decode, workflow, sink."""
    pytest.importorskip("inference.core.interfaces.stream.inference_pipeline")
    video_path = _write_smoke_video(tmp_path / "smoke.mp4", 6000)
    workflow_path = _write_no_model_workflow(tmp_path / "no_model_workflow.json")
    output = tmp_path / "result.json"

    completed = _run_harness(
        [
            "--workflow", workflow_path, "--source", video_path, "--copies", "2",
            "--warmup-frames", "5", "--duration", "1", "--repeats", "1",
            "--output", str(output),
        ]
    )  # fmt: skip

    assert completed.returncode == 0, (
        completed.stdout[-3000:] + completed.stderr[-3000:]
    )
    result = json.loads(output.read_text())
    repeat = result["repeats"][0]
    metrics = repeat["metrics"]
    assert repeat["errors"] == []
    assert metrics["window_frames"] > 0
    assert metrics["starved_sources"] == 0
    assert metrics["reordered_frames"] == metrics["duplicate_frames"] == 0
    assert metrics["missing_frames"] == 0
    assert metrics["capture_to_sink_count"] == metrics["window_frames"]
    assert set(repeat["sources"]) == {"0", "1"}
    assert all(s["first_frame_id"] == 1 for s in repeat["sources"].values())
    assert set(repeat["frame_image_types"]) == {"ndarray"}
    assert repeat["raw"]["digests"]["0"]
    # Sampled after `start()`: names the actual producer, never the pre-start
    # `NoneType`.
    assert repeat["startup"]["decoders"] and all(
        d != "NoneType" for d in repeat["startup"]["decoders"]
    )
    # No model in the workflow: no model calls, and the harness never fakes them.
    assert "model_call_count" not in metrics
    assert repeat["instrumentation"]["model_calls"] == {"available": False, "calls": 0}
    assert repeat["instrumentation"]["cuda_trace"] is None
    boundary = repeat["boundary"]
    assert boundary["available"] and boundary["lost"] == 0
    assert metrics["boundary_lost_outputs"] == 0
    stage = boundary["stages"]["completed_to_sink"]["sources"]["0"]
    assert stage["expected"] >= stage["delivered"] > 0
    assert harness.compare_results(result, result)["passed"]
    assert "cuda_trace" not in result["fingerprint"]


def test_manager_launch_env_overrides_are_linux_only_and_cuda_fork_safe() -> None:
    """On Linux the manager forks pipeline processes, so it must start with the
    CUDA-safe settings (no CUDA context before the fork); spawn hosts are unchanged."""
    assert harness.manager_launch_env_overrides("linux") == {
        "CORE_MODEL_SAM3_ENABLED": "False",
        "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
    }
    assert harness.manager_launch_env_overrides("darwin") == {}


def test_build_manager_init_command_conserves_null_source_buffer_strategy() -> None:
    """The two source-buffer strategy `None`s must survive `exclude_none=True`
    serialisation and come back out as explicit nulls, not the manager's own
    DROP_OLDEST/EAGER `VideoConfiguration` defaults, which is the exact wire path
    `StreamManagerClient.initialise_pipeline` exercises (`.dict(exclude_none=True)`
    then JSON over the socket).
    """
    pytest.importorskip("inference.core.interfaces.stream_manager.manager_app.entities")
    from inference.core.interfaces.stream_manager.manager_app.entities import (
        InitialisePipelinePayload,
        VideoConfiguration,
        WorkflowConfiguration,
    )

    payload = InitialisePipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration",
            video_reference="/tmp/does-not-need-to-exist.mp4",
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration",
            workflow_specification={
                "version": "1.0",
                "inputs": [],
                "steps": [],
                "outputs": [],
            },
            workflows_parameters={},
        ),
    )

    command = harness.build_manager_init_command(payload)
    # Real wire path: JSON round-trip, then the manager's own model validation,
    # not just inspecting the dict the harness built.
    on_the_wire = json.loads(json.dumps(command))
    parsed = InitialisePipelinePayload.model_validate(
        {k: v for k, v in on_the_wire.items() if k != "type"}
    )
    assert parsed.video_configuration.source_buffer_filling_strategy is None
    assert parsed.video_configuration.source_buffer_consumption_strategy is None


def test_real_manager_pipeline_smoke(tmp_path) -> None:
    """Real stream manager over TCP with the instrumented pipeline child.

    The pipeline runs a file source unpaced, so the fixture is long enough not to
    run out before termination (the manager's health check auto-terminates a
    depleted pipeline and stops answering commands that were in flight).
    """
    pytest.importorskip("inference.core.interfaces.stream_manager.manager_app.app")
    before = _manager_leftovers()
    video_path = _write_smoke_video(tmp_path / "smoke.mp4", 30000)
    workflow_path = _write_no_model_workflow(tmp_path / "no_model_workflow.json")
    output = tmp_path / "result.json"

    completed = _run_harness(
        [
            "--transport", "manager", "--workflow", workflow_path,
            "--source", video_path, "--copies", "1", "--warmup-frames", "5",
            "--duration", "1", "--repeats", "1", "--output", str(output),
        ],
        timeout=420,
    )  # fmt: skip

    assert completed.returncode == 0, (
        completed.stdout[-3000:] + completed.stderr[-3000:]
    )
    result = json.loads(output.read_text())
    repeat = result["repeats"][0]
    metrics = repeat["metrics"]
    assert repeat["errors"] == []
    assert repeat["startup"]["manager_pipeline_class"] == (
        "InstrumentedInferencePipelineManager"
    )
    assert metrics["window_frames"] > 0 and metrics["starved_sources"] == 0
    assert metrics["reordered_frames"] == metrics["duplicate_frames"] == 0
    # Cross-process: consumer receipt minus the child's FRAME_CAPTURED wall clock.
    assert metrics["capture_to_consume_count"] == metrics["window_frames"]
    assert metrics["capture_to_consume_p50_ms"] > 0
    # Measured in the pipeline child for the same frames; must not exceed
    # capture_to_consume for a file source (synthetic frame timestamps unused).
    assert metrics["capture_to_sink_count"] == metrics["window_frames"]
    assert 0 < metrics["capture_to_sink_p50_ms"] <= metrics["capture_to_consume_p50_ms"]
    assert metrics["consume_to_inference_completed_count"] == metrics["window_frames"]
    instrumentation = repeat["instrumentation"]
    assert instrumentation["probe_available"]
    assert instrumentation["probe_process"].startswith("pipeline child")
    assert instrumentation["decoders"] and repeat["startup"]["decoders"] == (
        instrumentation["decoders"]
    )
    # Sampled after drain, from the retained pipeline reference: names the actual
    # producer, never the pre-start `NoneType`.
    assert all(d != "NoneType" for d in instrumentation["decoders"])
    boundary = repeat["boundary"]
    assert boundary["available"] and boundary["lost"] == 0
    assert set(boundary["stages"]) == {
        "consumed_to_completed",
        "completed_to_sink",
        "sink_to_consumer",
    }
    consumer_stage = boundary["stages"]["sink_to_consumer"]["sources"]["0"]
    assert consumer_stage["delivered"] > 0
    assert boundary["sink_overflow_discards"] == consumer_stage["lost"]
    assert "manager_log_excerpt" not in repeat["startup"]
    assert "stream-manager-benchmark" not in json.dumps(result)
    assert _manager_leftovers() == before


def test_real_manager_failed_init_reports_and_cleans_up(tmp_path) -> None:
    pytest.importorskip("inference.core.interfaces.stream_manager.manager_app.app")
    before = _manager_leftovers()
    video_path = _write_smoke_video(tmp_path / "smoke.mp4", 30)
    workflow_path = tmp_path / "broken_workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "inputs": [{"type": "InferenceImage", "name": "image"}],
                "steps": [{"type": "roboflow_core/does_not_exist@v1", "name": "x"}],
                "outputs": [],
            }
        )
    )
    output = tmp_path / "result.json"

    completed = _run_harness(
        [
            "--transport", "manager", "--workflow", str(workflow_path),
            "--source", video_path, "--copies", "1", "--warmup-frames", "1",
            "--duration", "1", "--repeats", "1", "--output", str(output),
        ],
    )  # fmt: skip

    assert completed.returncode == 1, completed.stdout[-3000:]
    repeat = json.loads(output.read_text())["repeats"][0]
    assert repeat["metrics"] == {}
    assert any("manager run failed" in error for error in repeat["errors"])
    assert repeat["instrumentation"]["probe_available"] is False
    # Only the sanitized log excerpt survives the scratch directory's removal.
    assert isinstance(repeat["instrumentation"]["manager_log_excerpt"]["tail"], list)
    assert "stream-manager-benchmark" not in json.dumps(repeat)
    assert _manager_leftovers() == before


def test_real_cuda_trace_smoke_reports_no_proof_without_cuda(tmp_path) -> None:
    """The diagnostic trace runs on CPU (profiler only) and says so in the result."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("inference.core.interfaces.stream.inference_pipeline")
    video_path = _write_smoke_video(tmp_path / "smoke.mp4", 6000)
    workflow_path = _write_no_model_workflow(tmp_path / "no_model_workflow.json")
    output = tmp_path / "result.json"

    completed = _run_harness(
        [
            "--workflow", workflow_path, "--source", video_path, "--copies", "1",
            "--warmup-frames", "5", "--duration", "1", "--repeats", "1",
            "--cuda-trace", "--cuda-trace-seconds", "0.3", "--output", str(output),
        ]
    )  # fmt: skip

    assert completed.returncode == 0, (
        completed.stdout[-3000:] + completed.stderr[-3000:]
    )
    result = json.loads(output.read_text())
    assert result["fingerprint"]["cuda_trace"] is True
    assert result["labels"]["diagnostic"] == "cuda-trace"
    trace = result["repeats"][0]["instrumentation"]["cuda_trace"]
    assert trace["cuda_activity_traced"] == torch.cuda.is_available()
    # No model in the workflow: the profiler never starts, and says so.
    assert trace["trace_state"] == "not_started"
    assert trace["model_call_ranges"] == 0
    assert trace["status"] == "unproven"
    assert trace["model_call_host_device_copies"] is None
    assert result["repeats"][0]["errors"] == []
    # A diagnostic run never compares as a baseline.
    assert not harness.compare_results(_strip_trace(result), result)["passed"]


def _strip_trace(result: dict) -> dict:
    stripped = copy.deepcopy(result)
    del stripped["fingerprint"]["cuda_trace"]
    return stripped
