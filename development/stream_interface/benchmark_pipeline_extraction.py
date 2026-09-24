#!/usr/bin/env python3
"""End-to-end benchmark of the stream runtime for the pipeline/manager extraction.

Unlike `benchmark_engine_throughput.py` (which pre-decodes frames and times only the
Execution Engine), this harness drives the real runtime: source capture and decode,
the workflow run and either the in-process sink (`--transport in-process`) or the
stream manager's memory sink consumed over TCP (`--transport manager`).

Each repeat runs in a fresh child process: the pipeline starts, every source must
deliver `--warmup-frames` results (model loading / TRT build are reported as startup,
bounded by WARMUP_TIMEOUT_S), then results are measured for `--duration` seconds and
the pipeline is terminated. Results are written as JSON with raw per-frame data beside
the summaries; `--compare BASELINE.json` checks frame integrity, content digests and
practical regressions. The `observability` section of every result states which
metrics are measured, how, and which cannot be observed without runtime instrumentation.

Only `--host legacy` (the runtime in this checkout) is implemented. Prepare the
deterministic fixture video first (the binary is generated, never committed):

    python development/stream_interface/benchmark_pipeline_extraction.py --prepare-fixtures

CPU % convention: psutil per-process CPU, 100% == one fully used core, summed over the
harness process tree (manager server and pipeline processes included).
"""

import argparse
import asyncio
import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

import psutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from development.stream_interface import benchmark_instrumentation as instrumentation
from inference.core.interfaces.camera.source_reference_sanitizer import (
    redact_credentials_in_text,
    sanitize_source_reference,
)

FIXTURES_DIR = os.path.join(_THIS_DIR, "fixtures")
MANIFEST_PATH = os.path.join(FIXTURES_DIR, "manifest.json")
SCHEMA_VERSION = 1
WARMUP_TIMEOUT_S = 900.0  # covers model download-free loading and TRT engine builds
RESOURCE_SAMPLE_INTERVAL_S = 0.5
GPU_SAMPLE_INTERVAL_S = 1.0
PREALLOCATED_CUDA_PREFIX = "preallocated-cuda:"
PREALLOCATED_CUDA_POOL_SIZE = 64
# Manager transport: results are consumed for this long after the window closes, so
# outputs still buffered at window end are delivered before termination and the
# boundary report separates them from outputs that were lost.
MANAGER_DRAIN_S = 1.0
MANAGER_STOP_TIMEOUT_S = 20
BOUNDARY_ID_SAMPLE = 20
CUDA_TRACE_STOP_GRACE_S = 30.0
# Serialised output keys that are random per run (uuids) or timing-dependent.
NON_DETERMINISTIC_OUTPUT_KEYS = {"detection_id", "inference_id", "parent_id", "time"}
FLOAT_DIGEST_DECIMALS = 3
MISSING_RATIO_TOLERANCE = 0.01
# (metric, higher_is_better, practical equivalence band) — see section 8 of the plan.
COMPARED_METRICS = [
    ("total_fps", True, 0.03),
    ("min_source_fps", True, 0.03),
    ("capture_to_sink_p50_ms", False, 0.03),
    ("capture_to_sink_p95_ms", False, 0.03),
    ("capture_to_consume_p50_ms", False, 0.03),
    ("capture_to_consume_p95_ms", False, 0.03),
    ("cpu_percent_mean", False, 0.05),
    ("rss_mb_peak", False, 0.05),
    ("gpu_memory_mb_peak", False, 0.05),
    ("model_call_p50_ms", False, 0.03),
    ("model_call_p95_ms", False, 0.03),
]
# Metrics that must carry real, finite per-repeat evidence for an actual workload run
# (not merely skipped when absent on both sides): throughput, latency, CPU/RSS always;
# capture_to_consume only for the manager transport; GPU only when the host/run has GPU
# evidence; model-call timing only when model calls actually occurred. Everything else
# in COMPARED_METRICS stays optional (e.g. an unavailable platform metric).
ALWAYS_MANDATORY_METRIC_NAMES = {
    "total_fps",
    "min_source_fps",
    "capture_to_sink_p50_ms",
    "capture_to_sink_p95_ms",
    "cpu_percent_mean",
    "rss_mb_peak",
}
MANAGER_ONLY_MANDATORY_METRIC_NAMES = {
    "capture_to_consume_p50_ms",
    "capture_to_consume_p95_ms",
}
GPU_METRIC_NAMES = {"gpu_memory_mb_peak"}
MODEL_CALL_METRIC_NAMES = {"model_call_p50_ms", "model_call_p95_ms"}
# Absolute (not relative) allowance for rss_mb_slope_per_min: see compare_results.
RSS_SLOPE_ABS_BAND_MB_PER_MIN = 5.0
# Settings that must match for two results to be comparable.
FINGERPRINT_ENV_VARS = [
    "ENABLE_TENSOR_DATA_REPRESENTATION",
    "WORKFLOWS_IMAGE_TENSOR_DEVICE",
    "USE_INFERENCE_MODELS",
    "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES",
    "ONNXRUNTIME_EXECUTION_PROVIDERS",
    "VIDEO_SOURCE_BUFFER_SIZE",
    "INFERENCE_PIPELINE_PREDICTIONS_QUEUE_SIZE",
    "ENABLE_FRAME_DROP_ON_VIDEO_FILE_RATE_LIMITING",
    "DISABLE_GSTREAMER_VIDEO_SOURCES",
    "MAX_ACTIVE_MODELS",
]
# CUDA-safe launch policy for the manager process on Linux, where it forks pipeline
# processes (default start method, unchanged). Importing `inference.models` there
# initialises CUDA (sam3's class-level `torch.autocast` queries the device, and a
# non-NVML `torch.cuda.is_available()` creates a context), so the forked child's
# `cuInit` fails with "initialization error". Both are existing settings; the only
# side effect is SAM3 missing from the manager's model registry. Recorded in the
# fingerprint of manager results only; in-process runs are unaffected.
MANAGER_LINUX_CUDA_SAFE_ENV = {
    "CORE_MODEL_SAM3_ENABLED": "False",
    "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
}
RECORDED_ENV_VARS = FINGERPRINT_ENV_VARS + [
    "PYTHONPATH",
    "DISABLE_VERSION_CHECK",
    "MODEL_CACHE_DIR",
    "MMP_PERFORMANCE_PROFILING_ENABLED",
    "ENABLE_WORKFLOWS_PROFILING",
    "CUDA_VISIBLE_DEVICES",
]
RECORDED_DISTRIBUTIONS = [
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "torch",
    "torchvision",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-silicon",
    "tensorrt",
    "supervision",
    "pydantic",
    "psutil",
    "inference-models",
    "roboflow-workflows",
    "inference",
]
# Backend/decoder dependency versions recorded in the fingerprint (comparability
# gate): distinct from `inference`/`roboflow-workflows`, whose checkout versions are
# provenance only (see `environment.checkout_versions`) since they are the subject of
# this extraction, not a fixed dependency of it.
DEPENDENCY_FINGERPRINT_KEYS = [
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "torch",
    "torchvision",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-silicon",
    "tensorrt",
    "inference-models",
]
OBSERVABILITY = {
    "per_source_fps": "results delivered to the sink (in-process) or received by the "
    "consumer (manager) inside the measured window / window length",
    "capture_to_sink": "FRAME_CAPTURED status update (source thread, after grab() and "
    "before retrieve()/decode) to entry of the on_prediction sink, same process, "
    "perf_counter_ns. In-process: the harness sink. Manager: the memory sink inside "
    "the pipeline child, observed by the benchmark probe installed in that process "
    "(development/stream_interface/benchmark_instrumentation.py)",
    "capture_to_consume": "manager only, all sources: wall clock (time_ns) of the "
    "FRAME_CAPTURED event in the pipeline child to wall clock of CONSUME_RESULT "
    "receipt in the harness, same host. It is NOT sink latency: it adds memory-sink "
    "residence, polling and TCP/serialisation time. The synthetic file "
    "frame_timestamp is not used",
    "consume_to_inference_completed": "FRAME_CONSUMED to INFERENCE_COMPLETED status "
    "updates in the pipeline process; covers batch collection plus the whole "
    "workflow run (not model-only); ends at result enqueue when the RF-DETR stream "
    "pipeline is enabled",
    "model_call": "perf_counter_ns around the legacy provider call "
    "(ModelManagerModelsProvider._infer / infer_from_request_sync / "
    "run_tensor_native_inference), i.e. the request -> ModelManager -> model boundary: "
    "includes the model's own pre/post-processing, excludes workflow block logic. "
    "model_busy_fraction = summed call time / window length (can exceed 1 with "
    "parallel steps). Recorded in the process that runs the pipeline",
    "model_boundary": "kinds of the values entering and leaving the model call "
    "(ndarray, torch:cuda, torch:cpu, request image dicts). Shows what crosses the "
    "boundary; it does not show copies made inside the model. See cuda_trace",
    "cuda_trace": "--cuda-trace only (diagnostic run, excluded from baselines): "
    "torch.profiler started in the thread issuing model calls for a bounded window; "
    "device copy counts/bytes attributed to model calls by launch correlation and "
    "launching thread (never time overlap). status=unproven unless every copy and "
    "model call is covered; copies are never linked to model inputs/outputs, so it "
    "does not prove the absence of input host round-trips. Adds overhead",
    "boundary_loss": "frame ids per stage (FRAME_CONSUMED, INFERENCE_COMPLETED, sink "
    "entry, delivery to the harness) compared after terminate + join. Once the "
    "pipeline drained, every id missing downstream is lost unless the memory sink "
    "still held it (pending) or it is the last consumed frame per source that the "
    "multiplexer discards on stop (terminate_discarded, reported). Without a "
    "successful drain trailing ids are unresolved (a run error). Manager sink -> "
    "consumer discards are legacy memory-sink overflow, counted separately",
    "decode_time": "UNOBSERVABLE separately without instrumentation (included in "
    "capture_to_sink)",
    "dropped_frames": "frame-id gaps at the sink/consumer plus FRAME_DROPPED status "
    "updates (both transports); manager memory-sink overflow is reported by the "
    "boundary section",
    "batch_sizes": "non-empty frames per sink call (in-process) or per consumed "
    "result (manager)",
    "cpu_percent": "psutil, 100% == one core, summed over the harness process tree",
    "gpu_memory": "the compared metric (gpu_memory_mb_peak) is nvidia-smi "
    "per-process (compute-apps) memory summed over the harness's own process tree, "
    "steady window only; nvidia-smi total device memory is recorded as "
    "resources.gpu.memory_mb_peak for diagnostics only (moves with co-tenant GPU "
    "processes). torch allocator peak is also recorded for the in-process transport "
    "when CUDA was initialised",
    "decoder": "producer class per source read from private pipeline attributes in "
    "the process that runs the pipeline; manager runs also keep decoder "
    "selection/fallback lines from the manager log. model_backends is the loaded "
    "model class per model id",
    "sink_work": "the in-process sink serialises outputs with the manager's "
    "serialiser and hashes them; this cost is identical across compared phases",
}


class SinkRecord(NamedTuple):
    t_ns: int
    source_id: int
    frame_id: int
    latency_ns: Optional[int]
    workflow_ns: Optional[int]
    digest: Optional[str]


# ---------------------------------------------------------------------------------
# Pure statistics (covered by test_benchmark_pipeline_extraction.py)
# ---------------------------------------------------------------------------------


def percentile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolation percentile (numpy's default method)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    lower = math.floor(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def summarize_latencies_ns(values_ns: List[int]) -> Dict[str, Optional[float]]:
    values_ms = [v / 1e6 for v in values_ns]
    return {
        "count": len(values_ms),
        "p50_ms": percentile(values_ms, 50),
        "p95_ms": percentile(values_ms, 95),
        "p99_ms": percentile(values_ms, 99),
        "max_ms": max(values_ms) if values_ms else None,
        "mean_ms": statistics.fmean(values_ms) if values_ms else None,
    }


def analyze_frame_ids(frame_ids: List[int]) -> Dict[str, Optional[int]]:
    """Integrity of one source's delivered frame ids, in delivery order."""
    unique = set()
    duplicates = reordered = 0
    highest = None
    for frame_id in frame_ids:
        if frame_id in unique:
            duplicates += 1
            continue
        unique.add(frame_id)
        if highest is not None and frame_id < highest:
            reordered += 1
        highest = frame_id if highest is None else max(highest, frame_id)
    if not unique:
        return {
            "delivered": 0,
            "first_frame_id": None,
            "last_frame_id": None,
            "missing": 0,
            "reordered": 0,
            "duplicates": 0,
        }
    return {
        "delivered": len(frame_ids),
        "first_frame_id": min(unique),
        "last_frame_id": max(unique),
        "missing": max(unique) - min(unique) + 1 - len(unique),
        "reordered": reordered,
        "duplicates": duplicates,
    }


def summarize_repeat(
    records: List[SinkRecord],
    window_start_ns: int,
    window_end_ns: int,
    source_ids: List[int],
    latency_label: Optional[str],
) -> Dict[str, Any]:
    """Per-source FPS/integrity plus latency percentiles for the measured window.

    Integrity covers every delivered frame (warmup included); throughput and latency
    cover only frames delivered inside [window_start_ns, window_end_ns).
    """
    window_s = (window_end_ns - window_start_ns) / 1e9
    by_source: Dict[int, List[SinkRecord]] = defaultdict(list)
    for record in records:
        by_source[record.source_id].append(record)
    sources = {}
    latencies, workflow_times = [], []
    total_window_frames = total_missing = total_delivered = 0
    for source_id in source_ids:
        source_records = by_source.get(source_id, [])
        integrity = analyze_frame_ids([r.frame_id for r in source_records])
        in_window = [
            r for r in source_records if window_start_ns <= r.t_ns < window_end_ns
        ]
        latencies.extend(r.latency_ns for r in in_window if r.latency_ns is not None)
        workflow_times.extend(
            r.workflow_ns for r in in_window if r.workflow_ns is not None
        )
        total_window_frames += len(in_window)
        total_missing += integrity["missing"]
        total_delivered += integrity["delivered"]
        sources[str(source_id)] = {
            **integrity,
            "window_frames": len(in_window),
            "fps": len(in_window) / window_s if window_s > 0 else None,
        }
    fps_values = [s["fps"] or 0.0 for s in sources.values()]
    metrics: Dict[str, Optional[float]] = {
        "window_s": window_s,
        "window_frames": total_window_frames,
        "total_fps": total_window_frames / window_s if window_s > 0 else None,
        "min_source_fps": min(fps_values) if fps_values else None,
        "max_source_fps": max(fps_values) if fps_values else None,
        "source_fairness": (
            min(fps_values) / max(fps_values)
            if fps_values and max(fps_values) > 0
            else None
        ),
        "missing_frames": total_missing,
        "missing_ratio": (
            total_missing / (total_missing + total_delivered)
            if total_missing + total_delivered
            else 0.0
        ),
        "reordered_frames": sum(s["reordered"] for s in sources.values()),
        "duplicate_frames": sum(s["duplicates"] for s in sources.values()),
        "starved_sources": sum(1 for s in sources.values() if s["window_frames"] == 0),
    }
    if latency_label is not None:
        for key, value in summarize_latencies_ns(latencies).items():
            metrics[f"{latency_label}_{key}"] = value
    if workflow_times:
        for key, value in summarize_latencies_ns(workflow_times).items():
            metrics[f"consume_to_inference_completed_{key}"] = value
    return {"sources": sources, "metrics": metrics}


def summarize_metric_values(values: List[float]) -> Dict[str, Optional[float]]:
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "stdev": stdev,
        "cv": stdev / mean if mean else None,
        "min": min(values),
        "max": max(values),
    }


_T_975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228]


def _paired_relative_ci(
    baseline: List[float], candidate: List[float]
) -> Optional[Tuple[float, float]]:
    pairs = [(b, c) for b, c in zip(baseline, candidate) if b and c is not None]
    if len(pairs) < 2:
        return None
    deltas = [c / b - 1.0 for b, c in pairs]
    t_value = _T_975[len(deltas) - 2] if len(deltas) - 2 < len(_T_975) else 1.96
    half_width = t_value * statistics.stdev(deltas) / math.sqrt(len(deltas))
    mean = statistics.fmean(deltas)
    return mean - half_width, mean + half_width


def _expected_source_ids(fingerprint: dict) -> List[str]:
    """Source ids the run was configured to open, from declared sources/copies.

    Independent of which ids actually show up in raw digests, so a run that silently
    drops coverage of a configured source cannot pass by omission.
    """
    sources = fingerprint.get("sources") or [None]
    copies = fingerprint.get("copies") or 1
    return [str(i) for i in range(len(sources) * copies)]


def _has_model_calls(result: dict) -> bool:
    return any(
        r.get("instrumentation", {}).get("model_calls", {}).get("available")
        for r in result.get("repeats", [])
    )


def _has_gpu_evidence(result: dict) -> bool:
    return any(
        isinstance(r.get("resources", {}).get("gpu"), dict)
        for r in result.get("repeats", [])
    )


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _mandatory_metric_names(baseline: dict, candidate: dict) -> Set[str]:
    """Metrics that must carry real per-repeat evidence for this pair of runs."""
    names = set(ALWAYS_MANDATORY_METRIC_NAMES)
    if (
        baseline.get("labels", {}).get("transport") == "manager"
        or candidate.get("labels", {}).get("transport") == "manager"
    ):
        names |= MANAGER_ONLY_MANDATORY_METRIC_NAMES
    if _has_gpu_evidence(baseline) or _has_gpu_evidence(candidate):
        names |= GPU_METRIC_NAMES
    if _has_model_calls(baseline) or _has_model_calls(candidate):
        names |= MODEL_CALL_METRIC_NAMES
    return names


def compare_results(baseline: dict, candidate: dict) -> Dict[str, Any]:
    """Frame integrity, content and practical-regression check of candidate vs baseline."""
    failures, warnings, inconclusive, metrics = [], [], [], {}
    base_fp, cand_fp = baseline["fingerprint"], candidate["fingerprint"]
    for key in sorted(set(base_fp) | set(cand_fp)):
        if base_fp.get(key) != cand_fp.get(key):
            failures.append(
                f"not comparable: fingerprint '{key}' differs "
                f"({base_fp.get(key)!r} vs {cand_fp.get(key)!r})"
            )
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        # Fail closed: a result with no `model` section or no `weights_verified` key
        # is treated as unverified, not as an implicit pass.
        if _has_model_calls(result) and not result.get("model", {}).get(
            "weights_verified", False
        ):
            failures.append(
                f"{label}: model calls occurred but the loaded weights could not be "
                "verified against a local artifact hash (pass --model-id pointing at "
                "the cached package directory); a no-model run has no weights to verify"
            )
    base_repeats, cand_repeats = baseline["repeats"], candidate["repeats"]
    if len(base_repeats) != len(cand_repeats):
        inconclusive.append(
            f"repeat count differs: baseline={len(base_repeats)} "
            f"candidate={len(cand_repeats)}"
        )
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        for repeat in result["repeats"]:
            for error in repeat.get("errors", []):
                failures.append(f"{label} repeat {repeat['repeat']} invalid: {error}")
    base_missing = [r["metrics"].get("missing_ratio", 0.0) for r in base_repeats]
    for repeat in cand_repeats:
        m, index = repeat["metrics"], repeat["repeat"]
        if m.get("reordered_frames"):
            failures.append(f"repeat {index}: {m['reordered_frames']} reordered frames")
        if m.get("duplicate_frames"):
            failures.append(f"repeat {index}: {m['duplicate_frames']} duplicate frames")
        if m.get("starved_sources"):
            failures.append(f"repeat {index}: {m['starved_sources']} starved sources")
        if m.get("missing_frames") and not any(base_missing):
            failures.append(
                f"repeat {index}: {m['missing_frames']} missing frames "
                "(baseline delivered every frame)"
            )
    if base_missing and cand_repeats:
        cand_missing = [r["metrics"].get("missing_ratio", 0.0) for r in cand_repeats]
        if (
            statistics.fmean(cand_missing)
            > statistics.fmean(base_missing) + MISSING_RATIO_TOLERANCE
        ):
            failures.append(
                f"missing-frame ratio increased from {statistics.fmean(base_missing):.4f} "
                f"to {statistics.fmean(cand_missing):.4f}"
            )
    expected_sources = _expected_source_ids(base_fp)
    content = _compare_digests(base_repeats, cand_repeats, expected_sources)
    base_has_digests = bool(content["baseline_sources"])
    cand_has_digests = bool(content["candidate_sources"])
    if content["mismatched"]:
        failures.append(
            f"content digest mismatch on {content['mismatched']} of "
            f"{content['compared']} common frames"
        )
    if not base_has_digests and not cand_has_digests:
        # Output content evidence is mandatory: a pass must be backed by a digest
        # comparison, not merely the absence of a mismatch.
        failures.append(
            "content digests not captured on either side: output content is not "
            "verified"
        )
    elif base_has_digests != cand_has_digests:
        inconclusive.append(
            "content digests recorded on only one side "
            f"(baseline={base_has_digests}, candidate={cand_has_digests})"
        )
    else:
        if content["missing_sources"]:
            inconclusive.append(
                "missing source coverage: candidate has no content digests for "
                f"source(s) {', '.join(content['missing_sources'])}"
            )
        for label, missing_by_repeat in (
            ("baseline", content["missing_sources_per_repeat"]["baseline"]),
            ("candidate", content["missing_sources_per_repeat"]["candidate"]),
        ):
            for index, missing in enumerate(missing_by_repeat):
                if missing:
                    failures.append(
                        f"{label} repeat {index}: no content digests for expected "
                        f"source(s) {', '.join(missing)}"
                    )
        if not content["compared"]:
            inconclusive.append(
                "no common content digests despite both sides recording them: "
                "content not verified"
            )
    if content["baseline_nondeterministic"]:
        warnings.append(
            f"{content['baseline_nondeterministic']} baseline frames had differing "
            "digests across repeats and were excluded"
        )
    mandatory_metric_names = _mandatory_metric_names(baseline, candidate)
    for name, higher_is_better, band in COMPARED_METRICS:
        # Null/NaN readings are not evidence: treat them as absent, same as a missing
        # key, rather than letting them count towards n or pull the mean/CV.
        base_values = [
            v if _is_finite_number(v) else None
            for v in (r["metrics"].get(name) for r in base_repeats)
        ]
        cand_values = [
            v if _is_finite_number(v) else None
            for v in (r["metrics"].get(name) for r in cand_repeats)
        ]
        mandatory = name in mandatory_metric_names
        if mandatory:
            # A mandatory metric absent in the same repeat on both sides must not be
            # silently skipped by the aggregate-n check below: that repeat contributes
            # no real evidence for either side.
            both_missing = [
                index
                for index, (bv, cv) in enumerate(zip(base_values, cand_values))
                if bv is None and cv is None
            ]
            if both_missing:
                failures.append(
                    f"{name}: mandatory metric for this workload has no finite "
                    f"evidence on either side in repeat(s) {both_missing}"
                )
        base_summary = summarize_metric_values(base_values)
        cand_summary = summarize_metric_values(cand_values)
        base_supported, cand_supported = bool(base_summary["n"]), bool(
            cand_summary["n"]
        )
        if not base_supported and not cand_supported:
            continue
        # Per-repeat parity: aggregate n>0 on both sides can hide a single paired
        # repeat where the metric is missing on only one side.
        repeat_mismatches = [
            index
            for index, (bv, cv) in enumerate(zip(base_values, cand_values))
            if (bv is not None) != (cv is not None)
        ]
        if base_supported != cand_supported or repeat_mismatches:
            detail = f" (repeats {repeat_mismatches})" if repeat_mismatches else ""
            inconclusive.append(
                f"{name}: metric present on only one side "
                f"(baseline n={base_summary['n']}, candidate n={cand_summary['n']})"
                f"{detail}"
            )
            continue
        if not base_summary["mean"]:
            continue
        delta = cand_summary["mean"] / base_summary["mean"] - 1.0
        worse = -delta if higher_is_better else delta
        entry = {
            "baseline_mean": base_summary["mean"],
            "candidate_mean": cand_summary["mean"],
            "relative_delta": delta,
            "band": band,
            "paired_relative_ci95": _paired_relative_ci(base_values, cand_values),
            "baseline_cv": base_summary["cv"],
        }
        metrics[name] = entry
        if worse > band:
            failures.append(
                f"{name} regressed by {worse:.1%} (band {band:.0%}): "
                f"{base_summary['mean']:.3f} -> {cand_summary['mean']:.3f}"
            )
        if base_summary["cv"] is not None and base_summary["cv"] > band:
            inconclusive.append(
                f"{name}: baseline CV {base_summary['cv']:.1%} exceeds the {band:.0%} "
                "band; extend runs or stabilise the host"
            )
        if cand_summary["cv"] is not None and cand_summary["cv"] > band:
            inconclusive.append(
                f"{name}: candidate CV {cand_summary['cv']:.1%} exceeds the {band:.0%} "
                "band; extend runs or stabilise the host"
            )
    base_slopes = [r["metrics"].get("rss_mb_slope_per_min") for r in base_repeats]
    cand_slopes = [r["metrics"].get("rss_mb_slope_per_min") for r in cand_repeats]
    base_slope_summary = summarize_metric_values(base_slopes)
    cand_slope_summary = summarize_metric_values(cand_slopes)
    if base_slope_summary["n"] and cand_slope_summary["n"]:
        # Absolute, not relative: both sides accrue comparable RSS growth from the
        # harness's own retained per-frame records over a fixed-duration window, so a
        # ratio would flag that shared growth as a regression. Only a materially
        # faster climb on the candidate side (beyond what retention alone explains)
        # indicates a leak introduced by the change under test.
        slope_delta = cand_slope_summary["mean"] - base_slope_summary["mean"]
        metrics["rss_mb_slope_per_min"] = {
            "baseline_mean": base_slope_summary["mean"],
            "candidate_mean": cand_slope_summary["mean"],
            "absolute_delta_mb_per_min": slope_delta,
            "band_mb_per_min": RSS_SLOPE_ABS_BAND_MB_PER_MIN,
        }
        if slope_delta > RSS_SLOPE_ABS_BAND_MB_PER_MIN:
            failures.append(
                f"rss_mb_slope_per_min grew {slope_delta:.2f} MB/min faster than "
                f"baseline (band {RSS_SLOPE_ABS_BAND_MB_PER_MIN:.1f} MB/min): "
                f"{base_slope_summary['mean']:.2f} -> {cand_slope_summary['mean']:.2f}"
            )
    result = {
        "passed": not failures and not inconclusive,
        "failures": failures,
        "warnings": warnings,
        "inconclusive": inconclusive,
        "metrics": metrics,
        "content": content,
    }
    # Diagnostics boundary: baseline/candidate artifacts may be old or user-provided
    # and can carry unredacted references (e.g. a fingerprint mismatch interpolates
    # both sides' raw values); sanitize the assembled comparison, not just freshly
    # collected data, before it is printed or saved.
    return _sanitize_nested(result)


def _valid_digest(digest: Any) -> bool:
    """A digest is content evidence only if it is a real, non-empty hash string; an
    empty source list or a null digest proves nothing about the frame's content."""
    return isinstance(digest, str) and bool(digest)


def _compare_digests(
    base_repeats: List[dict], cand_repeats: List[dict], expected_sources: List[str]
) -> dict:
    reference: Dict[Tuple[str, int], set] = defaultdict(set)
    base_sources, cand_sources = set(), set()
    base_repeat_sources: List[set] = []
    for repeat in base_repeats:
        sources_here = set()
        for source_id, frames in repeat.get("raw", {}).get("digests", {}).items():
            valid_frames = [(fid, d) for fid, d in frames if _valid_digest(d)]
            if not valid_frames:
                continue
            base_sources.add(source_id)
            sources_here.add(source_id)
            for frame_id, digest in valid_frames:
                reference[(source_id, frame_id)].add(digest)
        base_repeat_sources.append(sources_here)
    unstable = {key for key, digests in reference.items() if len(digests) > 1}
    compared = mismatched = 0
    cand_repeat_sources: List[set] = []
    for repeat in cand_repeats:
        sources_here = set()
        for source_id, frames in repeat.get("raw", {}).get("digests", {}).items():
            valid_frames = [(fid, d) for fid, d in frames if _valid_digest(d)]
            if not valid_frames:
                continue
            cand_sources.add(source_id)
            sources_here.add(source_id)
            for frame_id, digest in valid_frames:
                key = (source_id, frame_id)
                if key not in reference or key in unstable:
                    continue
                compared += 1
                if digest not in reference[key]:
                    mismatched += 1
        cand_repeat_sources.append(sources_here)
    return {
        "compared": compared,
        "mismatched": mismatched,
        "baseline_nondeterministic": len(unstable),
        "baseline_sources": sorted(base_sources),
        "candidate_sources": sorted(cand_sources),
        "missing_sources": sorted(base_sources - cand_sources),
        "expected_sources": list(expected_sources),
        "missing_sources_per_repeat": {
            "baseline": [
                sorted(set(expected_sources) - s) for s in base_repeat_sources
            ],
            "candidate": [
                sorted(set(expected_sources) - s) for s in cand_repeat_sources
            ],
        },
    }


def output_digest(serialised_output: Any) -> str:
    canonical = json.dumps(
        _canonicalize(serialised_output), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _canonicalize(v)
            for k, v in value.items()
            if k not in NON_DETERMINISTIC_OUTPUT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v) for v in value]
    if isinstance(value, float):
        return round(value, FLOAT_DIGEST_DECIMALS)
    return value


# ---------------------------------------------------------------------------------
# Metadata, fixtures and sources
# ---------------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def sanitize_reference(reference: str) -> str:
    """Redact credentials from a free-text token (argv entry, log line, error
    message); delegates to the camera source sanitizer so unknown query
    parameters (auth, signature, ...) never survive alongside userinfo."""
    return redact_credentials_in_text(reference)


# argv flags whose value is a known source/model reference, not free text: these get
# the stricter `sanitize_source_reference` (which also redacts schemeless credentials
# like `user:pass@host:554/live`, invisible to the free-text scheme-URL scan).
_COMMAND_REFERENCE_FLAGS = {"--source", "--model-id"}


def _sanitize_command(argv: List[str]) -> List[str]:
    sanitized = []
    reference_next = False
    for arg in argv:
        if reference_next:
            sanitized.append(sanitize_source_reference(arg))
            reference_next = False
        else:
            # Check if arg is a reference flag with = form (e.g., --source=value)
            sanitized_arg = None
            for flag in _COMMAND_REFERENCE_FLAGS:
                if arg.startswith(flag + "="):
                    value = arg[len(flag) + 1 :]
                    sanitized_arg = flag + "=" + sanitize_source_reference(value)
                    break

            if sanitized_arg is not None:
                sanitized.append(sanitized_arg)
            else:
                # Regular sanitization for other args
                sanitized.append(sanitize_reference(arg))
                # Check if this arg is a reference flag without = form
                reference_next = arg in _COMMAND_REFERENCE_FLAGS
    return sanitized


def _sanitize_nested(value: Any) -> Any:
    """Recursively redact credentials from every string in a saved structure
    (errors, decoder log lines, provenance) so they never leak nested in JSON."""
    if isinstance(value, str):
        return redact_credentials_in_text(value)
    if isinstance(value, dict):
        return {k: _sanitize_nested(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_nested(v) for v in value]
    return value


def prepare_fixture_video(output_path: str) -> dict:
    """Render the manifest's deterministic panning video from committed test images."""
    import cv2
    import numpy as np

    spec = load_manifest()["video"]
    width, height, fps = spec["width"], spec["height"], float(spec["fps"])
    tiles = []
    for asset in spec["generator"]["assets"]:
        image = cv2.imread(os.path.join(_REPO_ROOT, asset["path"]))
        if image is None:
            raise SystemExit(f"Cannot read fixture asset {asset['path']}")
        tiles.append(
            cv2.resize(
                image,
                (width, spec["generator"]["panorama_height"]),
                interpolation=cv2.INTER_LINEAR,
            )
        )
    panorama = np.concatenate(tiles, axis=1)
    x_range = panorama.shape[1] - width
    y_range = panorama.shape[0] - height
    step_x, step_y = spec["generator"]["step_x"], spec["generator"]["step_y"]
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*spec["codec"]), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"cv2 cannot encode '{spec['codec']}' on this host")
    raw_digest = hashlib.sha256()
    for index in range(spec["frames"]):
        x = (index * step_x) % x_range
        y = abs((index * step_y) % (2 * y_range) - y_range)
        frame = np.ascontiguousarray(panorama[y : y + height, x : x + width])
        raw_digest.update(frame.tobytes())
        writer.write(frame)
    writer.release()
    return {
        "path": os.path.relpath(output_path, _REPO_ROOT),
        "sha256": sha256_file(output_path),
        "raw_frames_sha256": raw_digest.hexdigest(),
        "expected_sha256": spec.get("sha256"),
        "expected_raw_frames_sha256": spec.get("raw_frames_sha256"),
        "opencv": cv2.__version__,
    }


def describe_source(reference: str, manifest: dict) -> dict:
    import cv2

    path = reference
    if reference.startswith(PREALLOCATED_CUDA_PREFIX):
        path = reference[len(PREALLOCATED_CUDA_PREFIX) :]
    if not os.path.isfile(path):
        return {"reference": sanitize_source_reference(reference), "kind": "stream"}
    capture = cv2.VideoCapture(path)
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    description = {
        "reference": os.path.relpath(os.path.abspath(path), _REPO_ROOT),
        "kind": (
            "preallocated-cuda" if path != reference else "file (unpaced, every frame)"
        ),
        "sha256": sha256_file(path),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec": "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)),
    }
    capture.release()
    description["matches_manifest"] = description["sha256"] == manifest["video"].get(
        "sha256"
    )
    return description


def _hash_package_files(model_id: str) -> Dict[str, str]:
    """Recursively hashes every regular file in a local package directory, keyed by
    its path relative to the package root (real packages keep weights in nested
    directories, e.g. `base/weights.onnx`, not only at the top level)."""
    files = {}
    for root, _, names in os.walk(model_id):
        for name in sorted(names):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                files[os.path.relpath(path, model_id)] = sha256_file(path)
    return files


def describe_model(model_id: str) -> dict:
    if not os.path.isdir(model_id):
        # Not a local package directory: no file to hash, so the actual weights
        # loaded at runtime cannot be verified from here. Pass a cached package
        # directory as --model-id (see results/README.md) to get a verified run.
        return {
            "model_id": sanitize_source_reference(model_id),
            "kind": "registry id",
            "weights_verified": False,
        }
    files = _hash_package_files(model_id)
    config = {}
    config_path = os.path.join(model_id, "model_config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            config = json.load(f)
    return {
        # Host-specific cache paths stay out of results; the file digests identify it.
        "model_id": "<local package>",
        "kind": "local package",
        "package_sha256": (
            hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
            if files
            else None
        ),
        "files_sha256": files,
        # Verified only when actual file content was hashed above (recursively, so
        # nested weights directories count); an empty directory proves nothing.
        "weights_verified": bool(files),
        **{
            key: config.get(key)
            for key in (
                "canonical_model_id",
                "model_architecture",
                "backend_type",
                "quantization",
                "static_batch_size",
                "dynamic_batch_size_supported",
            )
        },
    }


def _run_text(command: List[str]) -> Optional[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=_REPO_ROOT,
            check=True,
        ).stdout.rstrip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict:
    versions = {}
    for name in RECORDED_DISTRIBUTIONS:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            pass
    dirty = _run_text(["git", "status", "--porcelain", "--untracked-files=no"])
    cpu_model = platform.processor()
    if sys.platform == "darwin":
        cpu_model = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu_model
    elif os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            names = re.findall(r"model name\s*:\s*(.+)", f.read())
        cpu_model = names[0] if names else cpu_model
    gpus = None
    if shutil.which("nvidia-smi"):
        gpus = _run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        )
    return {
        "git_sha": _run_text(["git", "rev-parse", "HEAD"]),
        "git_tracked_changes": dirty.splitlines() if dirty else [],
        "checkout_versions": {
            "inference": _read_version("inference/core/version.py", "__version__"),
            "roboflow-workflows": _read_version("workflows/pyproject.toml", "version"),
            "inference-models": _read_version(
                "inference_models/pyproject.toml", "version"
            ),
        },
        "installed_distributions": versions,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_mb": psutil.virtual_memory().total / 2**20,
        "gpus": gpus,
        "env": {k: os.environ[k] for k in RECORDED_ENV_VARS if k in os.environ},
    }


def _read_version(relative_path: str, key: str) -> Optional[str]:
    try:
        with open(os.path.join(_REPO_ROOT, relative_path)) as f:
            match = re.search(rf'^{key}\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    except OSError:
        return None
    return match.group(1) if match else None


def make_preallocated_cuda_producer(path: str):
    """Factory for a producer serving CUDA frames uploaded once before the run.

    Frames are CHW RGB uint8 tensors on WORKFLOWS_IMAGE_TENSOR_DEVICE, cloned per frame
    like the PyNvVideoCodec producer (device-to-device, no host copy). This isolates
    GPU residency downstream of the source; it does not measure hardware decoding.
    """
    import cv2
    import torch

    from inference.core.env import (
        ENABLE_TENSOR_DATA_REPRESENTATION,
        WORKFLOWS_IMAGE_TENSOR_DEVICE,
    )
    from inference.core.interfaces.camera.entities import (
        SourceProperties,
        VideoFrameProducer,
    )

    if not ENABLE_TENSOR_DATA_REPRESENTATION:
        raise SystemExit(
            f"{PREALLOCATED_CUDA_PREFIX} sources need ENABLE_TENSOR_DATA_REPRESENTATION=True"
        )
    if WORKFLOWS_IMAGE_TENSOR_DEVICE.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit(
            f"{PREALLOCATED_CUDA_PREFIX} sources need CUDA and "
            f"WORKFLOWS_IMAGE_TENSOR_DEVICE=cuda:N (got {WORKFLOWS_IMAGE_TENSOR_DEVICE})"
        )
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    pool = []
    while len(pool) < PREALLOCATED_CUDA_POOL_SIZE:
        ok, frame = capture.read()
        if not ok:
            break
        rgb = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1)
        pool.append(rgb.contiguous().to(WORKFLOWS_IMAGE_TENSOR_DEVICE))
    capture.release()
    if not pool:
        raise SystemExit(f"Could not decode frames from {path}")
    torch.cuda.synchronize()
    height, width = pool[0].shape[1:]

    class PreallocatedCudaFrameProducer(VideoFrameProducer):
        def __init__(self):
            self._served = 0

        def isOpened(self) -> bool:
            return True

        def grab(self) -> bool:
            if self._served >= total_frames:
                return False
            self._served += 1
            return True

        def retrieve(self):
            return True, pool[(self._served - 1) % len(pool)].clone()

        def release(self) -> None:
            return None

        def initialize_source_properties(self, properties: Dict[str, float]) -> None:
            return None

        def discover_source_properties(self) -> SourceProperties:
            return SourceProperties(
                width=width,
                height=height,
                total_frames=total_frames,
                is_file=True,
                fps=fps,
                is_reconnectable=False,
                timestamp_created=None,
            )

        def connection_error_message(self) -> str:
            return ""

    return PreallocatedCudaFrameProducer


# ---------------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------------


class ResourceSampler:
    """Samples CPU/RSS of the process tree and GPU memory in a background thread."""

    def __init__(self):
        self._root = psutil.Process(os.getpid())
        self._tracked: Dict[int, psutil.Process] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._gpu_available = shutil.which("nvidia-smi") is not None
        self._last_gpu_sample = 0.0
        self.samples: List[dict] = []
        self.gpu_samples: List[dict] = []
        self.processes: Dict[int, dict] = {}

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.wait(RESOURCE_SAMPLE_INTERVAL_S):
            self._sample_tree()
            now = time.monotonic()
            if (
                self._gpu_available
                and now - self._last_gpu_sample >= GPU_SAMPLE_INTERVAL_S
            ):
                self._last_gpu_sample = now
                self._sample_gpu()

    def _sample_tree(self) -> None:
        try:
            processes = [self._root] + self._root.children(recursive=True)
        except psutil.Error:
            return
        cpu = rss = 0.0
        for process in processes:
            tracked = self._tracked.setdefault(process.pid, process)
            try:
                with tracked.oneshot():
                    process_cpu = tracked.cpu_percent(None)
                    process_rss = tracked.memory_info().rss / 2**20
                    info = self.processes.setdefault(
                        tracked.pid,
                        {"name": tracked.name(), "cmdline": tracked.cmdline()[:4]},
                    )
            except psutil.Error:
                continue
            info["rss_mb_peak"] = max(info.get("rss_mb_peak", 0.0), process_rss)
            cpu += process_cpu
            rss += process_rss
        self.samples.append(
            {
                "t_ns": time.perf_counter_ns(),
                "processes": len(processes),
                "cpu_percent": cpu,
                "rss_mb": rss,
            }
        )

    def _sample_gpu(self) -> None:
        total = _run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        apps = _run_text(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        if total is None:
            return
        tree_pids = set(self._tracked)
        per_process = {}
        # NVML lists a CUDA context created on a non-main thread under that thread's
        # id (not the process id), with 0 MiB used (driver 550, L4: the manager's
        # pipeline processes create theirs on a worker thread). Such ids are kept
        # only if /proc says their thread group is in our tree.
        thread_contexts = {}
        for line in (apps or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            if int(parts[0]) in tree_pids:
                per_process[parts[0]] = _to_float(parts[1])
                continue
            tgid = _thread_group_id(int(parts[0]))
            if tgid is not None and tgid != int(parts[0]) and tgid in tree_pids:
                thread_contexts[parts[0]] = {
                    "tgid": tgid,
                    "memory_mb": _to_float(parts[1]),
                }
        # Each NVML entry is its own context, so a reported thread context adds to
        # its process's entry; a 0/None one is left for `summarize` to reject.
        for context in thread_contexts.values():
            if context["memory_mb"]:
                key = str(context["tgid"])
                per_process[key] = (per_process.get(key) or 0.0) + context["memory_mb"]
        gpus = []
        for line in total.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpus.append(
                    {
                        "index": parts[0],
                        "memory_used_mb": _to_float(parts[1]),
                        "utilization_percent": _to_float(parts[2]),
                    }
                )
        self.gpu_samples.append(
            {
                "t_ns": time.perf_counter_ns(),
                "gpus": gpus,
                "tree_process_memory_mb": per_process,
                # Only present when NVML reported thread ids, so in-process raw
                # samples keep their earlier shape.
                **(
                    {"tree_thread_contexts": thread_contexts} if thread_contexts else {}
                ),
            }
        )

    def summarize(self, window_start_ns: int, window_end_ns: int) -> dict:
        window = [
            s for s in self.samples if window_start_ns <= s["t_ns"] <= window_end_ns
        ]
        summary: Dict[str, Any] = {
            "cpu_percent_mean": None,
            # Peaks use the same steady measurement window as CPU/FPS: samples taken
            # during startup/warmup (allocator warm-up, model load) would otherwise
            # inflate a peak that is supposed to describe the measured window.
            "rss_mb_peak": max((s["rss_mb"] for s in window), default=None),
            "rss_mb_window_mean": None,
            "rss_mb_slope_per_min": None,
            "processes": self.processes,
            "gpu": "unavailable: nvidia-smi not found",
        }
        if window:
            summary["cpu_percent_mean"] = statistics.fmean(
                s["cpu_percent"] for s in window
            )
            summary["rss_mb_window_mean"] = statistics.fmean(
                s["rss_mb"] for s in window
            )
            summary["rss_mb_slope_per_min"] = _slope_per_minute(
                [(s["t_ns"], s["rss_mb"]) for s in window]
            )
        if self._gpu_available:
            gpu_window = [
                s
                for s in self.gpu_samples
                if window_start_ns <= s["t_ns"] <= window_end_ns
            ]
            # A tree-owned thread context that NVML reports as 0/None is memory the
            # driver does not attribute: any owned sum in this window would
            # under-count, so the owned peak is unsupported (None).
            unreported_thread_contexts = {
                tid: context["tgid"]
                for s in gpu_window
                for tid, context in s.get("tree_thread_contexts", {}).items()
                if not context["memory_mb"]
            }
            tree_window_samples = [
                sum(s["tree_process_memory_mb"].values())
                for s in gpu_window
                if s["tree_process_memory_mb"] and not unreported_thread_contexts
            ]
            summary["gpu"] = {
                # Total device memory: diagnostics only (other processes on a shared
                # GPU move this independently of the harness under test).
                "memory_mb_peak": max(
                    (
                        sum(g["memory_used_mb"] or 0 for g in s["gpus"])
                        for s in gpu_window
                    ),
                    default=None,
                ),
                "utilization_percent_mean": (
                    statistics.fmean(
                        statistics.fmean(
                            g["utilization_percent"] or 0 for g in s["gpus"]
                        )
                        for s in gpu_window
                        if s["gpus"]
                    )
                    if any(s["gpus"] for s in gpu_window)
                    else None
                ),
                # Owned process-tree memory: used for the compared metric below, since
                # it isolates the harness's own GPU usage from co-tenant processes.
                # None (not 0) when nvidia-smi never mapped a pid in this window.
                "tree_process_memory_mb_peak": max(tree_window_samples, default=None),
                "note": "per-process memory is empty when nvidia-smi cannot map pids "
                "(containers, Jetson)",
            }
            if unreported_thread_contexts:
                summary["gpu"]["unreported_tree_thread_contexts"] = {
                    "thread_to_process": unreported_thread_contexts,
                    "note": "NVML listed these tree-owned thread ids (CUDA context "
                    "created off the main thread) with 0 MiB, so owned memory is "
                    "unobservable; memory_mb_peak is device-wide, not owned",
                }
        return summary


def _thread_group_id(task_id: int) -> Optional[int]:
    """Process id (Tgid) of a Linux task id, from /proc; None if unreadable."""
    try:
        with open(f"/proc/{task_id}/status") as f:
            for line in f:
                if line.startswith("Tgid:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


def _slope_per_minute(points: List[Tuple[int, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    xs = [(t - points[0][0]) / 60e9 for t, _ in points]
    ys = [v for _, v in points]
    x_mean, y_mean = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if not denominator:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


class Recorder:
    """Collects sink/consumer deliveries and status-update timestamps (thread-safe)."""

    def __init__(self, source_count: int, disable_content_hash: bool = False):
        # Manager transport: frame_timestamp is wall clock only for non-file sources.
        self.wall_clock_frame_timestamps = False
        # --cuda-trace only: content hashing (_digest_prediction) runs on the sink
        # thread and adds artificial CPU/host transfers that would corrupt a CUDA
        # copy trace. Timed baselines must always keep content hashing; a diagnostic
        # run with this set makes no content-parity claim.
        self.disable_content_hash = disable_content_hash
        self._lock = threading.Lock()
        self.records: List[SinkRecord] = []
        self.delivered = [0] * source_count
        self.batch_sizes: Counter = Counter()
        self.image_types: Counter = Counter()
        self.dropped_events: Counter = Counter()
        self.errors: List[str] = []
        self.first_result_ns: Optional[int] = None
        # Manager transport: wall clock of CONSUME_RESULT receipt per frame, joined
        # with the pipeline child's FRAME_CAPTURED wall clock (file sources included).
        self.consumed_wall_ns: Dict[Tuple[int, int], int] = {}
        self._captured: Dict[Tuple[int, int], int] = {}
        self._consumed: Dict[Tuple[int, int], int] = {}
        self._workflow: Dict[Tuple[int, int], int] = {}

    def on_status_update(self, update) -> None:
        now = time.perf_counter_ns()
        event, payload = update.event_type, update.payload or {}
        with self._lock:
            if event == "FRAME_CAPTURED":
                self._captured[(payload.get("source_id") or 0, payload["frame_id"])] = (
                    now
                )
            elif event == "FRAME_CONSUMED":
                self._consumed[(payload.get("source_id") or 0, payload["frame_id"])] = (
                    now
                )
            elif event == "INFERENCE_COMPLETED":
                for source_id, frame_id in zip(
                    payload.get("sources_id", []), payload.get("frames_ids", [])
                ):
                    key = (source_id or 0, frame_id)
                    consumed = self._consumed.pop(key, None)
                    if consumed is not None:
                        self._workflow[key] = now - consumed
            elif event == "FRAME_DROPPED":
                # A dropped frame never reaches the sink, so its FRAME_CAPTURED
                # timestamp would otherwise sit in `_captured` for the rest of the
                # run: retiring it here is the only cleanup this state needs, there
                # is no drain/pending evidence to keep it for.
                self._captured.pop(
                    (payload.get("source_id") or 0, payload.get("frame_id")), None
                )
                self.dropped_events[str(payload.get("source_id"))] += 1
            if getattr(update.severity, "name", "") == "ERROR":
                self.errors.append(f"{update.context} {event}: {payload}")

    def on_prediction(self, predictions, video_frames) -> None:
        now = time.perf_counter_ns()
        if not isinstance(video_frames, list):
            predictions, video_frames = [predictions], [video_frames]
        entries = []
        for prediction, frame in zip(predictions, video_frames):
            if frame is None:
                continue
            image = frame.image
            # numpy>=2 arrays also expose `.device`, so check the type's module.
            image_type = type(image).__name__
            if type(image).__module__.startswith("torch"):
                image_type = f"torch:{image.device}"
            entries.append(
                (
                    frame.source_id or 0,
                    frame.frame_id,
                    (
                        None
                        if self.disable_content_hash
                        else _digest_prediction(prediction)
                    ),
                    image_type,
                )
            )
        with self._lock:
            self._add(
                now, [(s, f, d) for s, f, d, _ in entries], latency_from_capture=True
            )
            self.image_types.update(t for *_, t in entries)

    def on_consumed(self, now: int, outputs: List, frames_metadata: List) -> None:
        entries = []
        wall_now = datetime.now()
        wall_now_ns = time.time_ns()
        for output, metadata in zip(outputs, frames_metadata):
            if metadata is None:
                continue
            self.consumed_wall_ns[(metadata.source_id or 0, metadata.frame_id)] = (
                wall_now_ns
            )
            latency = None
            if self.wall_clock_frame_timestamps:
                latency = int(
                    (
                        wall_now - metadata.frame_timestamp.replace(tzinfo=None)
                    ).total_seconds()
                    * 1e9
                )
            entries.append(
                (
                    metadata.source_id or 0,
                    metadata.frame_id,
                    output_digest(output),
                    latency,
                )
            )
        with self._lock:
            self._add(now, entries, latency_from_capture=False)

    def _add(self, now: int, entries: List[tuple], latency_from_capture: bool) -> None:
        if not entries:
            return
        if self.first_result_ns is None:
            self.first_result_ns = now
        self.batch_sizes[str(len(entries))] += 1
        for entry in entries:
            source_id, frame_id, digest = entry[:3]
            key = (source_id, frame_id)
            if latency_from_capture:
                captured = self._captured.pop(key, None)
                latency = now - captured if captured is not None else None
            else:
                latency = entry[3]
            self.records.append(
                SinkRecord(
                    now,
                    source_id,
                    frame_id,
                    latency,
                    self._workflow.pop(key, None),
                    digest,
                )
            )
            if 0 <= source_id < len(self.delivered):
                self.delivered[source_id] += 1

    def all_sources_warmed_up(self, warmup_frames: int) -> bool:
        with self._lock:
            return min(self.delivered) >= warmup_frames

    def snapshot(self) -> List[SinkRecord]:
        """Records with workflow time joined by frame identity: the sink can run
        before INFERENCE_COMPLETED (the result is enqueued before that event)."""
        with self._lock:
            return [
                (
                    r._replace(workflow_ns=self._workflow[(r.source_id, r.frame_id)])
                    if r.workflow_ns is None
                    and (r.source_id, r.frame_id) in self._workflow
                    else r
                )
                for r in self.records
            ]


def _digest_prediction(prediction: Any) -> Optional[str]:
    if prediction is None:
        return None
    from inference.core.interfaces.http.orjson_utils import (
        serialise_single_workflow_result_element,
    )

    return output_digest(serialise_single_workflow_result_element(prediction, []))


def _wait_for_measurement(
    recorder: Recorder, args: argparse.Namespace, alive, during_window=None
) -> Tuple[int, int, List[str]]:
    """Waits for warmup, then the measured window: (window_start_ns, window_end_ns, errors).

    `during_window`, when given, runs once right after the window opens (the CUDA
    copy trace); a failure is reported as an error and does not end the window.
    """
    errors = []
    deadline = time.monotonic() + WARMUP_TIMEOUT_S
    while not recorder.all_sources_warmed_up(args.warmup_frames):
        if recorder.errors:
            return 0, 0, [f"runtime error during warmup: {recorder.errors[0]}"]
        if not alive():
            return 0, 0, ["pipeline ended before warmup completed"]
        if time.monotonic() > deadline:
            return 0, 0, [f"warmup not reached in {WARMUP_TIMEOUT_S}s"]
        time.sleep(0.01)
    window_start = time.perf_counter_ns()
    window_end = window_start + int(args.duration * 1e9)
    if during_window is not None:
        try:
            during_window()
        except Exception as error:  # noqa: BLE001 - diagnostic only
            errors.append(f"cuda trace failed: {error!r}")
    while time.perf_counter_ns() < window_end:
        if not alive():
            errors.append(
                "pipeline ended before the measured window closed (source exhausted? "
                "regenerate a longer fixture or shorten --duration)"
            )
            window_end = time.perf_counter_ns()
            break
        time.sleep(0.01)
    if recorder.errors:
        errors.append(f"runtime error during measurement: {recorder.errors[0]}")
    return window_start, window_end, errors


def run_in_process(args, sources, specification, parameters, recorder) -> dict:
    from inference.core.interfaces.stream.inference_pipeline import InferencePipeline

    # Copies of one preallocated-cuda path share its frame pool (one upload per path).
    producers = {
        s: make_preallocated_cuda_producer(s[len(PREALLOCATED_CUDA_PREFIX) :])
        for s in set(sources)
        if s.startswith(PREALLOCATED_CUDA_PREFIX)
    }
    references = [producers.get(s, s) for s in sources]
    probe = instrumentation.BoundaryProbe()
    instrumentation.activate(probe)
    started = time.perf_counter_ns()
    pipeline = InferencePipeline.init_with_workflow(
        video_reference=references if len(references) > 1 else references[0],
        workflow_specification=specification,
        workflows_parameters=parameters,
        on_prediction=probe.wrap_sink(recorder.on_prediction),
        status_update_handlers=[probe.on_status_update, recorder.on_status_update],
    )
    initialised = time.perf_counter_ns()
    pipeline.start(use_main_thread=False)
    finished = threading.Event()
    joiner = threading.Thread(
        target=lambda: (pipeline.join(), finished.set()), daemon=True
    )
    joiner.start()
    cuda_trace: Dict[str, Any] = {}
    try:
        window_start, window_end, errors = _wait_for_measurement(
            recorder,
            args,
            alive=lambda: not finished.is_set(),
            during_window=(
                (
                    lambda: cuda_trace.update(
                        _run_cuda_trace(probe, args.cuda_trace_seconds)
                    )
                )
                if args.cuda_trace
                else None
            ),
        )
    finally:
        # `video_source.start()` runs lazily inside the inference thread's frame
        # generator, not synchronously inside `pipeline.start()`, so sampling right
        # after `start()` still raced `NoneType`. By the end of the measurement
        # window every source has actually initialised.
        probe.note_decoders(pipeline)
        pipeline.terminate()
        joiner.join(timeout=60)
        instrumentation.deactivate()
        # `finished` is set only when join() returned: all results dispatched.
        probe.note_drained(finished.is_set())
    if joiner.is_alive():
        errors.append("pipeline did not stop within 60s of terminate()")
    torch_peak = None
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        torch_peak = torch_module.cuda.max_memory_allocated() / 2**20
    return {
        "init_s": (initialised - started) / 1e9,
        "started_ns": started,
        "window_start_ns": window_start,
        "window_end_ns": window_end,
        "errors": errors,
        "decoders": probe.decoders,
        "torch_cuda_max_memory_allocated_mb": torch_peak,
        "latency_label": "capture_to_sink",
        "probe": probe.export(),
        "cuda_trace": cuda_trace or None,
    }


def _run_cuda_trace(probe, seconds: float) -> dict:
    """Diagnostic only: profiles device copies for ~`seconds` inside the window.

    The profiler starts and stops inside model calls, in the thread issuing them,
    so a workflow without model calls yields a `not_started` (unproven) trace.
    """
    trace = instrumentation.CudaCopyTrace(seconds)
    probe.cuda_trace = trace
    if not trace.started.wait(seconds):
        trace.cancel()
    if not trace.stopped.wait(seconds + CUDA_TRACE_STOP_GRACE_S):
        trace.cancel()  # the owner thread stops it at the end of its next call
    if trace.stopped.is_set():
        probe.cuda_trace = None
    return trace.summarize()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def manager_launch_env_overrides(platform_name: str = sys.platform) -> Dict[str, str]:
    """Environment forced on the manager process (see `MANAGER_LINUX_CUDA_SAFE_ENV`)."""
    return (
        dict(MANAGER_LINUX_CUDA_SAFE_ENV) if platform_name.startswith("linux") else {}
    )


def build_manager_init_command(payload) -> dict:
    """Wire command for `StreamManagerClient`'s INIT, with the two source-buffer
    strategy fields forced back to `None` after serialisation.

    `StreamManagerClient.initialise_pipeline` builds this same command via
    `payload.dict(exclude_none=True)`, so setting `source_buffer_filling_strategy`
    / `source_buffer_consumption_strategy` to `None` on the `VideoConfiguration`
    passed into the payload has no effect: `exclude_none` drops them from the wire
    command entirely, and the manager then falls back to `VideoConfiguration`'s own
    DROP_OLDEST/EAGER defaults (tuned for RTSP-style live sources) instead of the
    WAIT/sequential defaults a `None` resolves to for file sources (see
    `video_source.py`) -- which is exactly what `InferencePipeline.init_with_workflow`
    leaves in place in-process. Restoring the `None`s into the already-encoded dict,
    after `exclude_none` has run, is what lets them survive JSON serialisation and
    reach `VideoConfiguration.model_validate` on the manager side as explicit nulls.
    """
    from inference.core.interfaces.stream_manager.manager_app.entities import (
        TYPE_KEY,
        CommandType,
    )

    command = payload.dict(exclude_none=True)
    command[TYPE_KEY] = CommandType.INIT
    command["video_configuration"]["source_buffer_filling_strategy"] = None
    command["video_configuration"]["source_buffer_consumption_strategy"] = None
    return command


def run_manager(args, sources, specification, parameters, recorder) -> dict:
    from inference.core.interfaces.stream_manager.api.stream_manager_client import (
        StreamManagerClient,
        build_response,
    )
    from inference.core.interfaces.stream_manager.manager_app.entities import (
        InitialisePipelinePayload,
        VideoConfiguration,
        WorkflowConfiguration,
    )

    if any(s.startswith(PREALLOCATED_CUDA_PREFIX) for s in sources):
        raise SystemExit(
            f"{PREALLOCATED_CUDA_PREFIX} sources are in-process only (the manager "
            "accepts serialisable references)"
        )
    recorder.wall_clock_frame_timestamps = not any(os.path.isfile(s) for s in sources)
    port = _free_port()
    # Everything the manager writes lives in one temporary directory that the
    # `finally` below removes, whether initialisation succeeded or not. Only the
    # sanitized log excerpt and the probe's content survive in the result.
    scratch = tempfile.mkdtemp(prefix="stream-manager-benchmark-")
    log_path = os.path.join(scratch, "manager.log")
    probe_dir = os.path.join(scratch, "probe")
    os.mkdir(probe_dir)
    log_file = open(log_path, "wb")
    env = {
        **os.environ,
        "STREAM_MANAGER_HOST": "127.0.0.1",
        "STREAM_MANAGER_PORT": str(port),
        instrumentation.PROBE_DIR_ENV: probe_dir,
        **manager_launch_env_overrides(),
    }
    started = time.perf_counter_ns()
    # The real manager (`manager_app.app.start`) with the pipeline process class
    # replaced by the instrumented subclass; the child installs the probe in run().
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "development.stream_interface.benchmark_instrumentation",
        ],
        cwd=_REPO_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    errors: List[str] = []
    result: Dict[str, Any] = {
        "started_ns": started,
        "latency_label": None,
        "probe": None,
        "manager_pipeline_class": instrumentation.InstrumentedInferencePipelineManager.__name__,
        # The manager never sets a start method, so it uses this interpreter's default.
        "manager_start_method": multiprocessing.get_start_method(),
        "manager_env_overrides": manager_launch_env_overrides(),
    }
    pipeline_id = None
    pipeline_processes: List[psutil.Process] = []
    server_responsive = True
    try:
        client = StreamManagerClient.init(
            host="127.0.0.1", port=port, operations_timeout=30.0
        )
        _wait_for_manager(client, server)
        result["server_ready_s"] = (time.perf_counter_ns() - started) / 1e9
        payload = InitialisePipelinePayload(
            video_configuration=VideoConfiguration(
                type="VideoConfiguration",
                video_reference=sources if len(sources) > 1 else sources[0],
            ),
            processing_configuration=WorkflowConfiguration(
                type="WorkflowConfiguration",
                workflow_specification=specification,
                workflows_parameters=parameters,
            ),
        )
        command = build_manager_init_command(payload)
        response = build_response(asyncio.run(client._handle_command(command)))
        pipeline_id = response.context.pipeline_id
        result["init_s"] = (time.perf_counter_ns() - started) / 1e9
        try:
            pipeline_processes = psutil.Process(server.pid).children(recursive=True)
        except psutil.Error:
            pipeline_processes = []
        stop_consuming = threading.Event()
        consumer_failure: List[str] = []

        async def consume() -> None:
            while not stop_consuming.is_set():
                consumed = await client.consume_pipeline_result(
                    pipeline_id, excluded_fields=[]
                )
                if not consumed.outputs:
                    await asyncio.sleep(0.001)
                    continue
                recorder.on_consumed(
                    time.perf_counter_ns(), consumed.outputs, consumed.frames_metadata
                )

        def consume_in_thread() -> None:
            try:
                asyncio.run(consume())
            except Exception as error:  # noqa: BLE001 - reported as run failure
                consumer_failure.append(repr(error))

        consumer = threading.Thread(target=consume_in_thread, daemon=True)
        consumer.start()
        window_start, window_end, errors = _wait_for_measurement(
            recorder,
            args,
            alive=lambda: server.poll() is None and consumer.is_alive(),
        )
        # Bounded drain: keep consuming so results buffered at window end reach the
        # harness; whatever is still buffered at termination is reported as pending.
        drain_deadline = time.monotonic() + MANAGER_DRAIN_S
        while consumer.is_alive() and time.monotonic() < drain_deadline:
            time.sleep(0.01)
        stop_consuming.set()
        consumer.join(timeout=30)
        errors.extend(f"consumer failed: {e}" for e in consumer_failure)
        if any("ConnectivityError" in failure for failure in consumer_failure):
            # The manager stopped answering (legacy behaviour when its health check
            # auto-terminates a depleted pipeline while a command is in flight): no
            # further commands, SIGTERM below still joins the child and its probe.
            server_responsive = False
            errors.append(
                "manager stopped answering commands; skipping status and terminate "
                "(file source exhausted? regenerate a longer fixture)"
            )
        if errors and server_responsive:
            try:
                status = asyncio.run(client.get_status(pipeline_id))
                errors.append(f"pipeline status report: {str(status.report)[:2000]}")
            except Exception as error:  # noqa: BLE001 - diagnostics only
                errors.append(f"pipeline status unavailable: {error!r}")
        result.update(window_start_ns=window_start, window_end_ns=window_end)
        result["latency_label"] = "capture_to_consume"
    except (Exception, SystemExit) as error:  # noqa: BLE001 - reported per repeat
        errors.append(f"manager run failed: {sanitize_reference(repr(error))}")
    finally:
        if server.poll() is None and server_responsive:
            # TERMINATE joins the pipeline child before answering, so the probe file
            # is complete once this returns. A pipeline whose INIT failed stays in
            # the manager's table without a known id, so every listed pipeline is
            # terminated (the port is private to this run). The manager's own
            # SIGTERM handler does not cope with live pipelines (it would hang
            # the harness on the child), hence the explicit terminates first.
            try:
                pipeline_ids = list(asyncio.run(client.list_pipelines()).pipelines)
            except Exception as error:  # noqa: BLE001 - the server is stopped below
                pipeline_ids = [pipeline_id] if pipeline_id else []
                errors.append(f"list_pipelines failed: {error!r}")
            for listed in pipeline_ids:
                try:
                    asyncio.run(client.terminate_pipeline(listed))
                except Exception as error:  # noqa: BLE001 - the server is stopped below
                    errors.append(f"terminate_pipeline failed: {error!r}")
        if server.poll() is None:
            server.send_signal(signal.SIGTERM)
            try:
                server.wait(timeout=MANAGER_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                errors.append(
                    f"manager did not stop within {MANAGER_STOP_TIMEOUT_S}s of "
                    "SIGTERM; killed"
                )
        _kill_tree(server.pid)
        for process in pipeline_processes:  # orphaned by a crashed manager
            try:
                if process.is_running():
                    process.kill()
                    errors.append(f"killed orphaned pipeline process {process.pid}")
            except psutil.Error:
                pass
        log_file.close()
        result["probe"] = _read_probe(probe_dir, pipeline_id, errors)
        result["manager_log_excerpt"] = _read_manager_log(log_path)
        shutil.rmtree(scratch, ignore_errors=True)
    result["decoders"] = (
        result["probe"]["decoders"]
        if result["probe"] and result["probe"].get("decoders")
        else result["manager_log_excerpt"]["decoder_lines"]
    )
    result["errors"] = errors
    return result


def _read_probe(probe_dir: str, pipeline_id: Optional[str], errors: List[str]):
    if pipeline_id is None:
        return None
    path = os.path.join(probe_dir, f"{pipeline_id}.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as error:
        errors.append(f"pipeline child probe unavailable: {type(error).__name__}")
        return None


def _read_manager_log(log_path: str) -> Dict[str, List[str]]:
    """Sanitized excerpt of the manager log: decoder lines and the tail."""
    try:
        with open(log_path, errors="replace") as f:
            lines = [sanitize_reference(line.rstrip())[:300] for line in f]
    except OSError:
        lines = []
    return {
        "decoder_lines": [line for line in lines if "decoder" in line.lower()][:20],
        "tail": lines[-20:],
    }


def _wait_for_manager(client, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"stream manager exited with code {server.returncode}")
        try:
            asyncio.run(client.list_pipelines())
            return
        except Exception:  # noqa: BLE001 - not listening yet
            time.sleep(0.5)
    raise SystemExit("stream manager did not answer LIST_PIPELINES within 120s")


def _kill_tree(pid: int) -> None:
    """Kills what is left of a process tree (a no-op for an exited, reaped root)."""
    try:
        root = psutil.Process(pid)
        processes = root.children(recursive=True) + [root]
    except psutil.Error:
        return
    for process in processes:
        try:
            process.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(processes, timeout=10)


def run_single_repeat(args: argparse.Namespace, repeat: int) -> dict:
    with open(args.workflow) as f:
        specification = json.load(f)
    parameters = {"model_id": args.model_id} if args.model_id else None
    sources = [s for s in args.source for _ in range(args.copies)]
    recorder = Recorder(source_count=len(sources), disable_content_hash=args.cuda_trace)
    sampler = ResourceSampler()
    sampler.start()
    try:
        runner = run_in_process if args.transport == "in-process" else run_manager
        outcome = runner(args, sources, specification, parameters, recorder)
    finally:
        sampler.stop()
    # perf_counter (harness timestamps) -> wall clock, to join the probe's time_ns.
    wall_offset_ns = time.time_ns() - time.perf_counter_ns()
    records = recorder.snapshot()
    errors = list(outcome.pop("errors"))
    window_start, window_end = outcome.pop("window_start_ns", 0), outcome.pop(
        "window_end_ns", 0
    )
    started = outcome.pop("started_ns")
    probe = outcome.pop("probe", None)
    cuda_trace = outcome.pop("cuda_trace", None)
    manager_log_excerpt = outcome.pop("manager_log_excerpt", None)
    if probe and probe.get("errors"):
        errors.extend(f"probe: {e}" for e in probe["errors"][:5])
    if not window_end:
        return {
            "repeat": repeat,
            "errors": errors,
            "metrics": {},
            "startup": outcome,
            "instrumentation": {
                "probe_available": probe is not None,
                "manager_log_excerpt": manager_log_excerpt,
            },
        }
    capture_to_sink: Dict[Tuple[int, int], int] = {}
    if args.transport == "manager":
        records, capture_to_sink = merge_probe_latencies(records, probe, recorder)
    summary = summarize_repeat(
        records,
        window_start,
        window_end,
        list(range(len(sources))),
        outcome.pop("latency_label"),
    )
    if capture_to_sink:
        in_window = [
            capture_to_sink[(r.source_id, r.frame_id)]
            for r in records
            if window_start <= r.t_ns < window_end
            and (r.source_id, r.frame_id) in capture_to_sink
        ]
        for key, value in summarize_latencies_ns(in_window).items():
            summary["metrics"][f"capture_to_sink_{key}"] = value
    resources = sampler.summarize(window_start, window_end)
    summary["metrics"].update(
        cpu_percent_mean=resources["cpu_percent_mean"],
        rss_mb_peak=resources["rss_mb_peak"],
        rss_mb_slope_per_min=resources["rss_mb_slope_per_min"],
        gpu_memory_mb_peak=(
            resources["gpu"]["tree_process_memory_mb_peak"]
            if isinstance(resources["gpu"], dict)
            else None
        ),
    )
    model_calls = summarize_model_calls(
        probe, window_start + wall_offset_ns, window_end + wall_offset_ns
    )
    if model_calls["available"]:
        summary["metrics"].update(
            model_call_count=model_calls["count"],
            model_call_p50_ms=model_calls["p50_ms"],
            model_call_p95_ms=model_calls["p95_ms"],
            model_call_mean_ms=model_calls["mean_ms"],
            model_busy_fraction=model_calls["busy_fraction"],
        )
    delivered_by_source: Dict[int, List[int]] = defaultdict(list)
    for record in records:
        delivered_by_source[record.source_id].append(record.frame_id)
    boundary = build_boundary_report(probe, delivered_by_source, args.transport)
    if boundary.get("lost"):
        errors.append(
            f"boundary loss: {boundary['lost']} completed outputs never reached the "
            f"sink ({_describe_boundary_loss(boundary)})"
        )
    if boundary.get("unresolved"):
        errors.append(
            f"boundary accounting unproven: pipeline not drained "
            f"(drained={boundary.get('drained')}), {boundary['unresolved']} trailing "
            f"outputs neither delivered, pending nor proven lost"
        )
    summary["metrics"]["boundary_lost_outputs"] = boundary.get("lost")
    summary["metrics"]["sink_overflow_discards"] = boundary.get(
        "sink_overflow_discards"
    )
    if summary["metrics"]["starved_sources"]:
        errors.append(f"{summary['metrics']['starved_sources']} sources starved")
    return {
        "repeat": repeat,
        "errors": errors,
        "startup": {
            **outcome,
            "first_result_s": (
                (recorder.first_result_ns - started) / 1e9
                if recorder.first_result_ns
                else None
            ),
            "warmup_done_s": (window_start - started) / 1e9,
        },
        **summary,
        "batch_sizes": dict(recorder.batch_sizes),
        "frame_image_types": dict(recorder.image_types),
        "frame_dropped_events": dict(recorder.dropped_events),
        "resources": resources,
        "instrumentation": {
            "probe_available": probe is not None,
            "probe_process": (
                "pipeline child (spawned by the stream manager)"
                if args.transport == "manager"
                else "harness process"
            ),
            "model_calls": model_calls,
            "model_backends": probe.get("model_backends", {}) if probe else {},
            "decoders": probe.get("decoders", []) if probe else [],
            "pipeline_dropped_events": probe.get("dropped_events", {}) if probe else {},
            "cuda_child_main_thread_init": (
                probe.get("cuda_child_main_thread_init") if probe else None
            ),
            "cuda_trace": cuda_trace,
            "content_hash_disabled": recorder.disable_content_hash,
            "manager_log_excerpt": manager_log_excerpt,
        },
        "boundary": boundary,
        "raw": {
            "window_start_ns": window_start,
            "window_end_ns": window_end,
            # [source_id, frame_id, t_ns - window_start, latency_ns, workflow_ns]
            "deliveries": [
                [
                    r.source_id,
                    r.frame_id,
                    r.t_ns - window_start,
                    r.latency_ns,
                    r.workflow_ns,
                ]
                for r in records
            ],
            "digests": _digests_by_source(records),
            # [start_wall_ns - window_start_wall_ns, duration_ns, method, model, batch, thread]
            "model_calls": (
                [
                    [c[0] - window_start - wall_offset_ns, *c[1:]]
                    for c in probe["model_calls"]
                ]
                if probe
                else []
            ),
            "model_call_methods": probe.get("model_call_methods") if probe else None,
            "model_ids": probe.get("model_ids") if probe else None,
            "capture_to_sink_ns": (
                [[s, f, ns] for (s, f), ns in capture_to_sink.items()]
                if capture_to_sink
                else None
            ),
            "resource_samples": sampler.samples,
            "gpu_samples": sampler.gpu_samples,
        },
    }


def _describe_boundary_loss(boundary: dict) -> str:
    parts = []
    for stage, entry in boundary.get("stages", {}).items():
        for source, source_entry in entry["sources"].items():
            if source_entry["lost"]:
                parts.append(
                    f"{stage} source {source}: leading={source_entry['leading_lost']} "
                    f"interior={source_entry['interior_lost']} "
                    f"trailing={source_entry['trailing_lost']} "
                    f"ids={source_entry['lost_ids_sample']}"
                )
    return "; ".join(parts)


def _digests_by_source(records: List[SinkRecord]) -> Dict[str, List[list]]:
    digests: Dict[str, List[list]] = defaultdict(list)
    for record in records:
        if record.digest is not None:
            digests[str(record.source_id)].append([record.frame_id, record.digest])
    return dict(digests)


# ---------------------------------------------------------------------------------
# Boundary-loss evidence and model-call timing (pure; covered by the tests)
# ---------------------------------------------------------------------------------


def analyze_boundary_loss(
    expected_ids,
    delivered_ids,
    pending_ids=(),
    drained: Optional[bool] = False,
    terminate_discard_ids=(),
) -> Dict[str, Any]:
    """Classifies ids that reached an upstream stage but not the downstream one.

    `expected_ids` reached the upstream stage (e.g. INFERENCE_COMPLETED), `delivered_ids`
    the downstream one (e.g. the sink), `pending_ids` are known to be buffered at
    termination. Ids need not start at 1 and the two stages need not have equal
    counts: only ids missing downstream are classified. Pending ids are never lost.
    Missing ids before or between delivered ids are lost. Missing ids after the last
    delivered id depend on `drained` (terminate + join returned, so upstream work
    has been handed downstream): lost if drained, except ids in `terminate_discard_ids`
    (explicitly observed terminate-time discard, reported as `terminate_discarded`);
    `trailing_unresolved` (not proven either way) if not drained. There is no
    heuristic allowance: an id is only exempt from loss when it was actually observed
    being discarded.
    """
    expected, delivered, pending = (
        set(expected_ids),
        set(delivered_ids),
        set(pending_ids),
    )
    undelivered = expected - delivered
    held = undelivered & pending
    missing = undelivered - pending
    last = max(delivered) if delivered else None
    if delivered:
        first = min(delivered)
        leading = {i for i in missing if i < first}
        trailing = {i for i in missing if i > last}
    else:
        leading, trailing = set(), set(missing)
    interior = missing - leading - trailing
    discarded: Set[int] = set()
    if drained:
        discarded = trailing & set(terminate_discard_ids)
        trailing_lost, unresolved = trailing - discarded, set()
    else:
        trailing_lost, unresolved = set(), trailing
    lost = leading | interior | trailing_lost
    trailing_pending = {i for i in held if last is None or i > last}
    return {
        "expected": len(expected),
        "delivered": len(delivered),
        "delivered_unexpected": len(delivered - expected),
        "leading_lost": len(leading),
        "interior_lost": len(interior),
        "interior_pending": len(held - trailing_pending),
        "trailing_pending": len(trailing_pending),
        "trailing_lost": len(trailing_lost),
        "terminate_discarded": len(discarded),
        "trailing_unresolved": len(unresolved),
        "lost": len(lost),
        "lost_ids_sample": sorted(lost)[:BOUNDARY_ID_SAMPLE],
    }


def build_boundary_report(
    probe: Optional[dict], delivered_by_source: Dict[int, List[int]], transport: str
) -> Dict[str, Any]:
    """Per-source stage chain from the probe plus what the harness received.

    `probe["drained"]` (terminate + join returned in the pipeline process) decides
    whether trailing gaps are loss or unresolved. Every source seen at any stage is
    accounted, including one that was only ever consumed.
    """
    if not probe:
        return {"available": False, "reason": "no probe data", "lost": None}
    drained = probe.get("drained")
    # (name, upstream, downstream, is_sink_overflow)
    stages = [("consumed_to_completed", "consumed_ids", "completed_ids", False)]
    terminate_discard_ids: Dict[str, List[int]] = {
        str(s): ids for s, ids in probe.get("terminate_discarded_ids", {}).items()
    }
    sink_entered: Dict[str, List[int]] = defaultdict(list)
    for source_id, frame_id, *_ in probe.get("sink_entries", []):
        sink_entered[str(source_id)].append(frame_id)
    pending: Dict[str, List[int]] = defaultdict(list)
    for source_id, frame_id in probe.get("pending_at_termination", []):
        pending[str(source_id)].append(frame_id)
    delivered = {str(s): ids for s, ids in delivered_by_source.items()}
    stage_ids: Dict[str, Dict[str, List[int]]] = {
        "consumed_ids": probe.get("consumed_ids", {}),
        "completed_ids": probe.get("completed_ids", {}),
        "sink_entered": dict(sink_entered),
        "delivered": delivered,
    }
    if transport == "manager":
        stages.append(("completed_to_sink", "completed_ids", "sink_entered", False))
        # The legacy memory sink is a bounded deque: whatever entered it and was
        # neither consumed nor still buffered at termination was discarded as the
        # oldest entry. Reported, not an error (missing_ratio covers it).
        stages.append(("sink_to_consumer", "sink_entered", "delivered", True))
    else:
        stages.append(("completed_to_sink", "completed_ids", "delivered", False))
    report: Dict[str, Any] = {
        "available": True,
        "drained": drained,
        "stages": {},
        "lost": 0,
        "sink_overflow_discards": 0,
        "terminate_discarded": 0,
        "unresolved": 0,
        "probe_errors": list(probe.get("errors", [])),
    }
    source_keys = sorted(
        set().union(*(ids.keys() for ids in stage_ids.values())), key=lambda s: int(s)
    )
    for name, upstream, downstream, is_sink_overflow in stages:
        per_source = {}
        for source in source_keys:
            per_source[source] = analyze_boundary_loss(
                stage_ids[upstream].get(source, []),
                stage_ids[downstream].get(source, []),
                pending.get(source, []) if downstream == "delivered" else (),
                drained=drained,
                terminate_discard_ids=(
                    terminate_discard_ids.get(source, [])
                    if name == "consumed_to_completed"
                    else ()
                ),
            )
        lost = sum(entry["lost"] for entry in per_source.values())
        report["stages"][name] = {"sources": per_source, "lost": lost}
        report["terminate_discarded"] += sum(
            entry["terminate_discarded"] for entry in per_source.values()
        )
        report["unresolved"] += sum(
            entry["trailing_unresolved"] for entry in per_source.values()
        )
        if is_sink_overflow:
            report["sink_overflow_discards"] += lost
        else:
            report["lost"] += lost
    return report


def summarize_model_calls(
    probe: Optional[dict], window_start_wall_ns: int, window_end_wall_ns: int
) -> Dict[str, Any]:
    """Model-call timing inside the window, per model id / provider method."""
    if not probe or not probe.get("model_calls"):
        return {"available": False, "calls": 0}
    methods, model_ids = probe["model_call_methods"], probe["model_ids"]
    in_window = [
        call
        for call in probe["model_calls"]
        if window_start_wall_ns <= call[0] < window_end_wall_ns
    ]
    durations = [call[1] for call in in_window]
    window_ns = window_end_wall_ns - window_start_wall_ns
    by_key: Dict[str, List[list]] = defaultdict(list)
    for call in in_window:
        by_key[f"{model_ids[call[3]]} via {methods[call[2]]}"].append(call)
    return {
        "available": True,
        "calls": len(in_window),
        "calls_total": len(probe["model_calls"]),
        **summarize_latencies_ns(durations),
        "busy_fraction": sum(durations) / window_ns if window_ns > 0 else None,
        "threads": len({call[5] for call in in_window}),
        "by_model": {
            key: {
                **summarize_latencies_ns([c[1] for c in calls]),
                "batch_mean": statistics.fmean(c[4] for c in calls),
                "backend": probe.get("model_backends", {}).get(key.split(" via ")[0]),
            }
            for key, calls in by_key.items()
        },
        "input_kinds": probe.get("model_input_kinds", {}),
        "output_kinds": probe.get("model_output_kinds", {}),
        "backends": probe.get("model_backends", {}),
    }


def merge_probe_latencies(
    records: List[SinkRecord], probe: Optional[dict], recorder: "Recorder"
) -> Tuple[List[SinkRecord], Dict[Tuple[int, int], int]]:
    """Manager transport: fills capture_to_consume (wall clock across processes) and
    consume_to_inference_completed from the child's probe; returns the per-frame
    capture_to_sink map measured in the child for the same records."""
    if not probe:
        return records, {}
    by_key = {(entry[0], entry[1]): entry for entry in probe.get("sink_entries", [])}
    merged, capture_to_sink = [], {}
    for record in records:
        key = (record.source_id, record.frame_id)
        entry = by_key.get(key)
        if entry is None:
            merged.append(record)
            continue
        captured_wall, sink_latency, workflow = entry[2], entry[3], entry[4]
        consumed_wall = recorder.consumed_wall_ns.get(key)
        latency = record.latency_ns
        if captured_wall is not None and consumed_wall is not None:
            latency = consumed_wall - captured_wall
        if sink_latency is not None:
            capture_to_sink[key] = sink_latency
        merged.append(record._replace(latency_ns=latency, workflow_ns=workflow))
    return merged, capture_to_sink


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end stream pipeline benchmark (capture -> workflow -> sink).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        choices=["legacy", "nextgen-direct", "nextgen-gateway"],
        default="legacy",
    )
    parser.add_argument(
        "--transport", choices=["in-process", "manager"], default="in-process"
    )
    parser.add_argument("--workflow", help="Path to a workflow specification JSON.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Video reference (file, stream URL) or "
        f"'{PREALLOCATED_CUDA_PREFIX}PATH'; repeatable.",
    )
    parser.add_argument(
        "--copies", type=int, default=1, help="Independent sources opened per --source."
    )
    parser.add_argument("--warmup-frames", type=int, default=200)
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Seconds measured."
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", help="Results JSON path.")
    parser.add_argument(
        "--compare",
        help="Baseline results JSON. Without --workflow, compares --output to it.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Value for the workflow's `model_id` parameter (registry id or local "
        "package directory); defaults to the id pinned in the workflow.",
    )
    parser.add_argument(
        "--prepare-fixtures",
        action="store_true",
        help="Generate fixtures/benchmark.mp4 from the manifest and exit.",
    )
    parser.add_argument(
        "--cuda-trace",
        action="store_true",
        help="Diagnostic run (in-process only, never a baseline): profile device "
        "copies with torch.profiler for --cuda-trace-seconds inside the window and "
        "attribute them to model calls. Adds overhead; the result is fingerprinted "
        "as a trace so it cannot be compared with baselines.",
    )
    parser.add_argument("--cuda-trace-seconds", type=float, default=3.0)
    parser.add_argument("--child-repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.prepare_fixtures:
        return args
    if args.host != "legacy":
        parser.error(
            f"--host {args.host} is not implemented yet: the next-gen hosts arrive in "
            "WP-P401/P403; only the legacy runtime of this checkout can be measured"
        )
    if args.compare and not args.workflow:
        if not args.output:
            parser.error("--compare without --workflow needs --output CANDIDATE.json")
        return args
    if not args.workflow or not args.source or not args.output:
        parser.error("--workflow, --source and --output are required for a run")
    if (
        args.copies < 1
        or args.repeats < 1
        or args.warmup_frames < 1
        or args.duration <= 0
    ):
        parser.error(
            "--copies, --repeats, --warmup-frames and --duration must be positive"
        )
    if args.cuda_trace and args.transport != "in-process":
        parser.error("--cuda-trace profiles the harness process: in-process only")
    if args.cuda_trace and not 0 < args.cuda_trace_seconds <= args.duration:
        parser.error("--cuda-trace-seconds must be positive and within --duration")
    for source in args.source:
        path = (
            source[len(PREALLOCATED_CUDA_PREFIX) :]
            if source.startswith(PREALLOCATED_CUDA_PREFIX)
            else source
        )
        if path.endswith(".mp4") and not os.path.isfile(path):
            parser.error(f"{path} does not exist; generate it with --prepare-fixtures")
    return args


def build_result(args: argparse.Namespace, repeats: List[dict]) -> dict:
    repeats = [_sanitize_nested(repeat) for repeat in repeats]
    manifest = load_manifest()
    with open(args.workflow, "rb") as f:
        workflow_sha = hashlib.sha256(f.read()).hexdigest()
    workflow_name = os.path.basename(args.workflow)
    expected_workflow_sha = manifest["workflows"].get(workflow_name, {}).get("sha256")
    sources = [describe_source(s, manifest) for s in args.source]
    model = (
        describe_model(args.model_id)
        if args.model_id
        else {
            "model_id": "<workflow default>",
            "kind": "pinned in workflow",
            # Unlike --model-id pointing at a cache directory, a workflow-pinned
            # registry id cannot be hashed from argv alone.
            "weights_verified": False,
        }
    )
    environment = collect_environment()
    # Actual backend classes and decoders observed at runtime (instrumentation), not
    # just the env vars that influence them: a dependency swap invisible to
    # FINGERPRINT_ENV_VARS (e.g. a fallback to a different provider) still needs to
    # mark the run not comparable. Source revision and inference/Workflows/streams
    # package versions are provenance (see `environment`), not part of this gate,
    # since the extraction is what's being measured.
    observed_backends = sorted(
        {
            backend
            for r in repeats
            for backend in r.get("instrumentation", {})
            .get("model_backends", {})
            .values()
        }
    )
    observed_decoders = sorted(
        {d for r in repeats for d in r.get("startup", {}).get("decoders") or []}
    )
    observed_cuda_child_main_thread_init = sorted(
        {
            status
            for r in repeats
            for status in [
                r.get("instrumentation", {}).get("cuda_child_main_thread_init")
            ]
            if status
        }
    )
    dependency_versions = {
        key: environment["installed_distributions"][key]
        for key in DEPENDENCY_FINGERPRINT_KEYS
        if key in environment["installed_distributions"]
    }
    fingerprint = {
        "workflow_sha256": workflow_sha,
        "sources": [s.get("sha256") or s["reference"] for s in sources],
        "source_kinds": [s["kind"] for s in sources],
        "copies": args.copies,
        "warmup_frames": args.warmup_frames,
        "duration_s": args.duration,
        "model": model.get("package_sha256") or model["model_id"],
        "transport": args.transport,
        "python_version": environment["python"].split()[0],
        "dependency_versions": dependency_versions,
        "cpu_model": environment["cpu_model"],
        "gpus": environment["gpus"],
        **{f"env.{k}": environment["env"].get(k) for k in FINGERPRINT_ENV_VARS},
        **({"model_backends": observed_backends} if observed_backends else {}),
        **({"decoders": observed_decoders} if observed_decoders else {}),
        # Only present for diagnostic runs, so earlier baselines stay comparable.
        **({"cuda_trace": True} if args.cuda_trace else {}),
        # Only present for Linux manager runs, so Mac and in-process baselines stay
        # comparable.
        **(
            {"manager_env_overrides": manager_launch_env_overrides()}
            if args.transport == "manager" and manager_launch_env_overrides()
            else {}
        ),
        # Only present for Linux manager runs (same condition as
        # `manager_env_overrides` above): the child records a status on every
        # platform (including "skipped:not-linux" on Mac), so gating on transport
        # alone would still break Mac baselines recorded before this field existed.
        **(
            {"cuda_child_main_thread_init": observed_cuda_child_main_thread_init}
            if args.transport == "manager"
            and manager_launch_env_overrides()
            and observed_cuda_child_main_thread_init
            else {}
        ),
    }
    # Physical hardware (cpu_model, gpus, driver) must match for performance
    # comparisons: benchmark results are only valid on the same CPU, GPU, and driver
    # versions. Software --host labels (legacy/nextgen) are independent and allowed
    # to differ. Source/package version checkouts stay as provenance only.
    metric_names = sorted({k for r in repeats for k in r.get("metrics", {})})
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": _sanitize_command(sys.argv),
        "labels": {
            "host": args.host,
            "transport": args.transport,
            **({"diagnostic": "cuda-trace"} if args.cuda_trace else {}),
        },
        "protocol": {
            "warmup_frames": args.warmup_frames,
            "warmup_timeout_s": WARMUP_TIMEOUT_S,
            "duration_s": args.duration,
            "repeats": args.repeats,
            "copies": args.copies,
            "repeat_isolation": "fresh child process per repeat",
            "multi_source_shape": "one pipeline with all sources (both transports)",
            "cpu_percent_convention": "100% == one core, summed over the process tree",
        },
        "fingerprint": fingerprint,
        "workflow": {
            "path": os.path.relpath(os.path.abspath(args.workflow), _REPO_ROOT),
            "sha256": workflow_sha,
            "matches_manifest": workflow_sha == expected_workflow_sha,
        },
        "sources": sources,
        "model": model,
        "environment": environment,
        "observability": OBSERVABILITY,
        "summary": {
            name: summarize_metric_values(
                [r.get("metrics", {}).get(name) for r in repeats]
            )
            for name in metric_names
        },
        "repeats": repeats,
    }
    # Model ids and env values (not just repeats, already sanitized above) can carry
    # URL references (a mistyped --model-id, a credentialed MODEL_CACHE_DIR): sanitize
    # the whole assembled result, the final form persisted to disk, not only its parts.
    return _sanitize_nested(result)


def run_repeat_in_child(args: argparse.Namespace, repeat: int) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        output = os.path.join(directory, "repeat.json")
        command = [sys.executable, os.path.abspath(__file__), "--host", args.host]
        command += ["--transport", args.transport, "--workflow", args.workflow]
        for source in args.source:
            command += ["--source", source]
        command += [
            "--copies",
            str(args.copies),
            "--warmup-frames",
            str(args.warmup_frames),
        ]
        command += [
            "--duration",
            str(args.duration),
            "--repeats",
            "1",
            "--output",
            output,
        ]
        command += ["--child-repeat", str(repeat)]
        if args.model_id:
            command += ["--model-id", args.model_id]
        if args.cuda_trace:
            command += [
                "--cuda-trace",
                "--cuda-trace-seconds",
                str(args.cuda_trace_seconds),
            ]
        child = subprocess.Popen(command, cwd=_REPO_ROOT)
        try:
            child.wait(timeout=WARMUP_TIMEOUT_S + args.duration + 300)
        except subprocess.TimeoutExpired:
            _kill_tree(child.pid)
            return {
                "repeat": repeat,
                "errors": ["repeat timed out and was killed"],
                "metrics": {},
            }
        if child.returncode != 0 or not os.path.isfile(output):
            return {
                "repeat": repeat,
                "errors": [f"repeat process exited with code {child.returncode}"],
                "metrics": {},
            }
        with open(output) as f:
            return json.load(f)


def print_summary(result: dict) -> None:
    print(f"\n=== {result['labels']} copies={result['protocol']['copies']} ===")
    for name in (
        "total_fps",
        "min_source_fps",
        "source_fairness",
        "capture_to_sink_p50_ms",
        "capture_to_sink_p95_ms",
        "capture_to_consume_p50_ms",
        "capture_to_consume_p95_ms",
        "consume_to_inference_completed_p50_ms",
        "model_call_p50_ms",
        "model_call_p95_ms",
        "model_busy_fraction",
        "missing_frames",
        "reordered_frames",
        "boundary_lost_outputs",
        "sink_overflow_discards",
        "cpu_percent_mean",
        "rss_mb_peak",
        "rss_mb_slope_per_min",
        "gpu_memory_mb_peak",
    ):
        stats = result["summary"].get(name)
        if stats and stats.get("n"):
            print(
                f"{name:40s} mean={stats['mean']:.3f} stdev={stats['stdev']:.3f} "
                f"n={stats['n']}"
            )
    for repeat in result["repeats"]:
        for error in repeat.get("errors", []):
            print(f"repeat {repeat['repeat']} ERROR: {error[:500]}")


def print_comparison(comparison: dict) -> None:
    print(f"\n=== comparison: {'PASSED' if comparison['passed'] else 'FAILED'} ===")
    for name, entry in comparison["metrics"].items():
        if "absolute_delta_mb_per_min" in entry:
            print(
                f"{name:40s} delta={entry['absolute_delta_mb_per_min']:+.2f}MB/min "
                f"band={entry['band_mb_per_min']:.1f}MB/min"
            )
            continue
        print(
            f"{name:40s} delta={entry['relative_delta']:+.2%} band={entry['band']:.0%} "
            f"ci95={entry['paired_relative_ci95']}"
        )
    for failure in comparison["failures"]:
        print(f"FAIL: {failure}")
    for reason in comparison["inconclusive"]:
        print(f"INCONCLUSIVE: {reason}")
    for warning in comparison["warnings"]:
        print(f"WARN: {warning}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.prepare_fixtures:
        report = prepare_fixture_video(os.path.join(FIXTURES_DIR, "benchmark.mp4"))
        print(json.dumps(report, indent=2))
        if report["expected_sha256"] and report["sha256"] != report["expected_sha256"]:
            print(
                "WARNING: encoded bytes differ from the manifest (encoder build differs); "
                "copy the reference file to keep fixture checksums identical across hosts"
            )
        return 0
    if args.child_repeat is not None:
        # Sanitize before writing: this file is read back and folded into the parent
        # result verbatim, so redacting only at the parent (build_result) is too late.
        repeat = _sanitize_nested(run_single_repeat(args, args.child_repeat))
        with open(args.output, "w") as f:
            json.dump(repeat, f)
        return 0
    if args.compare and not args.workflow:
        with open(args.compare) as f:
            baseline = json.load(f)
        with open(args.output) as f:
            candidate = json.load(f)
        comparison = compare_results(baseline, candidate)
        print_comparison(comparison)
        return 0 if comparison["passed"] else 1
    repeats = [run_repeat_in_child(args, index) for index in range(args.repeats)]
    result = build_result(args, repeats)
    exit_code = 0 if all(not r.get("errors") for r in repeats) else 1
    if args.compare:
        with open(args.compare) as f:
            result["comparison"] = compare_results(json.load(f), result)
        exit_code = exit_code or (0 if result["comparison"]["passed"] else 1)
    with open(args.output, "w") as f:
        json.dump(result, f)
    print_summary(result)
    if "comparison" in result:
        print_comparison(result["comparison"])
    print(f"\nresults: {args.output}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
