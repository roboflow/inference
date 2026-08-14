"""Bounded-cardinality Prometheus metrics for the video processor.

The processor can run jobs from several workspaces in one process. Metrics must
therefore describe aggregate worker behavior without job, source, workspace,
workflow, or model identifiers. Detailed benchmark attribution belongs in the
benchmark result manifest, not Prometheus labels.
"""

import math
import re
import threading
from collections import defaultdict

VALID_MODES = ("stream", "batch", "unknown")
VALID_OUTCOMES = ("completed", "error", "cancelled", "stopped")
VALID_TRANSPORTS = ("whip", "rtsp")
VALID_CLAIM_REJECTION_REASONS = (
    "execution_cell_mismatch",
    "implicit_cross_cell",
    "invalid_placement",
    "processor_cell_missing",
)
CELL_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def _bounded(value, allowed, fallback):
    value = str(value or "").lower()
    return value if value in allowed else fallback


def _escape_label(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(**labels):
    if not labels:
        return ""
    rendered = ",".join(
        f'{key}="{_escape_label(value)}"' for key, value in sorted(labels.items())
    )
    return "{" + rendered + "}"


class _Histogram:
    def __init__(self, buckets):
        self.buckets = tuple(float(bucket) for bucket in buckets)
        self.values = {}

    def observe(self, label, value):
        value = max(0.0, float(value))
        record = self.values.setdefault(
            label,
            {"buckets": [0] * len(self.buckets), "count": 0, "sum": 0.0},
        )
        for index, upper_bound in enumerate(self.buckets):
            if value <= upper_bound:
                record["buckets"][index] += 1
        record["count"] += 1
        record["sum"] += value

    def render(self, name, help_text, label_name):
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        for label in sorted(self.values):
            record = self.values[label]
            for upper_bound, count in zip(self.buckets, record["buckets"]):
                lines.append(
                    f"{name}_bucket"
                    f"{_labels(**{label_name: label, 'le': _format_number(upper_bound)})} "
                    f"{count}"
                )
            lines.append(
                f"{name}_bucket"
                f"{_labels(**{label_name: label, 'le': '+Inf'})} "
                f"{record['count']}"
            )
            lines.append(
                f"{name}_sum{_labels(**{label_name: label})} "
                f"{_format_number(record['sum'])}"
            )
            lines.append(
                f"{name}_count{_labels(**{label_name: label})} {record['count']}"
            )
        return lines


def _format_number(value):
    value = float(value)
    if math.isfinite(value) and value.is_integer():
        return str(int(value))
    return format(value, ".12g")


class ProcessorMetrics:
    """Thread-safe process-lifetime counters and latency histograms."""

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs_started = defaultdict(int)
        self._jobs_finished = defaultdict(int)
        self._frames_processed = defaultdict(int)
        self._claim_rejections = defaultdict(int)
        self._job_start_duration = _Histogram((0.5, 1, 2, 5, 10, 30, 60, 120, 300))
        self._time_to_first_result = _Histogram((0.5, 1, 2, 5, 10, 30, 60, 120, 300))
        self._decode_to_result_latency = _Histogram(
            (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
        )

    def job_started(self, mode):
        mode = _bounded(mode, VALID_MODES, "unknown")
        with self._lock:
            self._jobs_started[mode] += 1

    def job_finished(self, mode, outcome):
        mode = _bounded(mode, VALID_MODES, "unknown")
        outcome = _bounded(outcome, VALID_OUTCOMES, "error")
        with self._lock:
            self._jobs_finished[(mode, outcome)] += 1

    def pipeline_started(self, mode, duration_seconds):
        mode = _bounded(mode, VALID_MODES, "unknown")
        with self._lock:
            self._job_start_duration.observe(mode, duration_seconds)

    def frame_processed(self, mode, latency_seconds, first_result_seconds=None):
        mode = _bounded(mode, VALID_MODES, "unknown")
        with self._lock:
            self._frames_processed[mode] += 1
            self._decode_to_result_latency.observe(mode, latency_seconds)
            if first_result_seconds is not None:
                self._time_to_first_result.observe(mode, first_result_seconds)

    def claim_rejected(self, reason):
        reason = _bounded(
            reason,
            VALID_CLAIM_REJECTION_REASONS,
            "invalid_placement",
        )
        with self._lock:
            self._claim_rejections[reason] += 1

    def render(
        self,
        *,
        active_jobs,
        capacity,
        tier,
        retiring,
        cell=None,
        active_publishers=None,
    ):
        active_jobs = max(0, int(active_jobs))
        capacity = max(1, int(capacity))
        tier = _bounded(tier, ("gpu", "cpu"), "unknown")
        cell = str(cell or "legacy")
        if cell != "legacy" and not CELL_ID_RE.fullmatch(cell):
            cell = "unknown"
        publishers = active_publishers or {}

        with self._lock:
            jobs_started = dict(self._jobs_started)
            jobs_finished = dict(self._jobs_finished)
            frames_processed = dict(self._frames_processed)
            claim_rejections = dict(self._claim_rejections)
            job_start_duration = self._job_start_duration.render(
                "video_processor_job_start_duration_seconds",
                "Seconds from job receipt until the workflow pipeline is initialized",
                "mode",
            )
            time_to_first_result = self._time_to_first_result.render(
                "video_processor_time_to_first_result_seconds",
                "Seconds from job receipt until the first workflow result",
                "mode",
            )
            decode_latency = self._decode_to_result_latency.render(
                "video_processor_decode_to_result_latency_seconds",
                "Frame decode timestamp to workflow result latency in seconds",
                "mode",
            )

        lines = [
            "# HELP video_processor_info Static video processor identity",
            "# TYPE video_processor_info gauge",
            f"video_processor_info{_labels(cell=cell, tier=tier)} 1",
            "# HELP video_processor_busy Number of active jobs assigned to this worker",
            "# TYPE video_processor_busy gauge",
            f"video_processor_busy {active_jobs}",
            "# HELP video_processor_active_jobs Number of active jobs assigned to this worker",
            "# TYPE video_processor_active_jobs gauge",
            f"video_processor_active_jobs {active_jobs}",
            "# HELP video_processor_capacity Maximum concurrent jobs accepted by this worker",
            "# TYPE video_processor_capacity gauge",
            f"video_processor_capacity {capacity}",
            "# HELP video_processor_available_slots Unoccupied concurrent job slots",
            "# TYPE video_processor_available_slots gauge",
            f"video_processor_available_slots {max(0, capacity - active_jobs)}",
            "# HELP video_processor_retiring Whether this worker is draining and retiring",
            "# TYPE video_processor_retiring gauge",
            f"video_processor_retiring {1 if retiring else 0}",
            "# HELP video_processor_jobs_started_total Jobs accepted by processing mode",
            "# TYPE video_processor_jobs_started_total counter",
        ]
        for mode in VALID_MODES:
            lines.append(
                f"video_processor_jobs_started_total{_labels(mode=mode)} "
                f"{jobs_started.get(mode, 0)}"
            )

        lines.extend(
            [
                "# HELP video_processor_jobs_finished_total Jobs finished by mode and outcome",
                "# TYPE video_processor_jobs_finished_total counter",
            ]
        )
        for mode in VALID_MODES:
            for outcome in VALID_OUTCOMES:
                lines.append(
                    "video_processor_jobs_finished_total"
                    f"{_labels(mode=mode, outcome=outcome)} "
                    f"{jobs_finished.get((mode, outcome), 0)}"
                )

        lines.extend(
            [
                "# HELP video_processor_frames_processed_total Workflow results emitted by mode",
                "# TYPE video_processor_frames_processed_total counter",
            ]
        )
        for mode in VALID_MODES:
            lines.append(
                f"video_processor_frames_processed_total{_labels(mode=mode)} "
                f"{frames_processed.get(mode, 0)}"
            )

        lines.extend(
            [
                "# HELP video_processor_claim_rejections_total Claims rejected before job execution by bounded reason",
                "# TYPE video_processor_claim_rejections_total counter",
            ]
        )
        for reason in VALID_CLAIM_REJECTION_REASONS:
            lines.append(
                "video_processor_claim_rejections_total"
                f"{_labels(reason=reason)} {claim_rejections.get(reason, 0)}"
            )

        lines.extend(
            [
                "# HELP video_processor_output_publishers Active annotated output publishers",
                "# TYPE video_processor_output_publishers gauge",
            ]
        )
        for transport in VALID_TRANSPORTS:
            lines.append(
                "video_processor_output_publishers"
                f"{_labels(transport=transport)} "
                f"{max(0, int(publishers.get(transport, 0)))}"
            )

        lines.extend(job_start_duration)
        lines.extend(time_to_first_result)
        lines.extend(decode_latency)
        return "\n".join(lines) + "\n"
