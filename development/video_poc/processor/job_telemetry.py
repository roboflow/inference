"""Bounded, job-local telemetry for video processor status reports.

Unlike ``processor_metrics.py``, this data is returned only for an authorized job
and can therefore be used by the staging benchmark harness without introducing
job/workspace labels into Prometheus.  The schema is additive: the original
``frames``, ``fps`` and timing fields remain present for older clients.
"""

import math
import os
import socket
import threading
import time
from datetime import datetime


STATS_SCHEMA_VERSION = 2
LATENCY_BOUNDS_MS = (5, 10, 20, 35, 50, 75, 100, 150, 250, 500, 1000, 2000, 5000)


def _safe_env(*names):
    """Return one short, printable value from an explicit environment allowlist."""
    for name in names:
        value = os.getenv(name)
        if value:
            value = "".join(c for c in str(value) if c.isprintable()).strip()
            if value:
                return value[:256]
    return None


def build_runtime_identity(processor_id=None, cell=None):
    """Build non-secret process/image identity once at worker startup.

    Container images cannot discover their own registry reference, so deployers
    should inject ``VIDEO_PROC_IMAGE`` and ``VIDEO_PROC_GIT_SHA``.  The fallbacks
    are deliberately limited to well-known, non-secret names; the environment is
    never enumerated or returned wholesale.
    """
    identity = {
        "schemaVersion": 1,
        "processorId": str(processor_id)[:256] if processor_id else None,
        "cell": str(cell)[:63] if cell else None,
        "hostname": socket.gethostname()[:256],
        "processId": os.getpid(),
        "image": _safe_env("VIDEO_PROC_IMAGE", "VIDEO_PROCESSOR_IMAGE", "IMAGE_URI"),
        "revision": _safe_env(
            "VIDEO_PROC_GIT_SHA", "VIDEO_PROCESSOR_REVISION", "GIT_SHA", "COMMIT_SHA"
        ),
        "variant": _safe_env("VIDEO_PROC_RUNTIME_VARIANT"),
        "podUid": _safe_env("POD_UID"),
        "gpuVisibleDevices": _safe_env(
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"
        ),
    }
    return {key: value for key, value in identity.items() if value is not None}


class JobTelemetry:
    """Thread-safe counters and a fixed-memory decode-to-result histogram."""

    def __init__(self, monotonic_clock=time.time):
        self.lock = threading.Lock()
        self._clock = monotonic_clock
        self.reset()

    def reset(self):
        with getattr(self, "lock", threading.Lock()):
            self.job_received_at = None
            self.pipeline_started_at = None
            self.first_result_at = None
            self.frames = 0
            self.last_latency_ms = None
            self.ema_latency_ms = None
            self._last_frame_time = None
            self._counters = {
                "captured": 0,
                "decoded": 0,
                "dropped": 0,
                "inferred": 0,
                "rendered": 0,
                "published": 0,
                "imageOutputHostMaterializations": 0,
            }
            self._latency_count = 0
            self._latency_sum_ms = 0.0
            self._latency_min_ms = None
            self._latency_max_ms = None
            # One non-cumulative bucket per bound plus an overflow bucket.
            self._latency_bucket_counts = [0] * (len(LATENCY_BOUNDS_MS) + 1)

    def on_job(self):
        self.reset()
        with self.lock:
            self.job_received_at = self._clock()

    def on_pipeline_start(self):
        now = self._clock()
        with self.lock:
            self.pipeline_started_at = now
            return max(0.0, now - self.job_received_at) if self.job_received_at else 0.0

    def on_source_event(self, event_type):
        # FRAME_CONSUMED is the closest stable pipeline event to "decoded": it
        # is emitted only after a decoded VideoFrame has been selected from the
        # source buffer for inference. Captured frames that are later dropped
        # are tracked separately.
        counter = {
            "FRAME_CAPTURED": "captured",
            "FRAME_CONSUMED": "decoded",
            "FRAME_DROPPED": "dropped",
        }.get(str(event_type))
        if counter:
            with self.lock:
                self._counters[counter] += 1

    def on_result(self, video_frame):
        now = self._clock()
        with self.lock:
            first_result = self.first_result_at is None
            if first_result:
                self.first_result_at = now
            self.frames += 1
            self._counters["inferred"] += 1
            self._last_frame_time = now
            frame_timestamp = video_frame.frame_timestamp
            latency_ms = (
                datetime.now(tz=frame_timestamp.tzinfo) - frame_timestamp
            ).total_seconds() * 1000.0
            if not math.isfinite(latency_ms):
                latency_ms = 0.0
            latency_ms = max(0.0, latency_ms)
            self.last_latency_ms = latency_ms
            self.ema_latency_ms = (
                latency_ms
                if self.ema_latency_ms is None
                else 0.9 * self.ema_latency_ms + 0.1 * latency_ms
            )
            self._observe_latency_locked(latency_ms)
            first_result_seconds = (
                max(0.0, now - self.job_received_at)
                if first_result and self.job_received_at
                else None
            )
            return latency_ms / 1000.0, first_result_seconds

    def record_latency_ms(self, latency_ms):
        """Test/adapter seam for recording a known decode-to-result sample."""
        latency_ms = float(latency_ms)
        if not math.isfinite(latency_ms):
            return
        with self.lock:
            self._observe_latency_locked(max(0.0, latency_ms))

    def _observe_latency_locked(self, latency_ms):
        self._latency_count += 1
        self._latency_sum_ms += latency_ms
        self._latency_min_ms = (
            latency_ms
            if self._latency_min_ms is None
            else min(self._latency_min_ms, latency_ms)
        )
        self._latency_max_ms = (
            latency_ms
            if self._latency_max_ms is None
            else max(self._latency_max_ms, latency_ms)
        )
        bucket = len(LATENCY_BOUNDS_MS)
        for index, upper_bound in enumerate(LATENCY_BOUNDS_MS):
            if latency_ms <= upper_bound:
                bucket = index
                break
        self._latency_bucket_counts[bucket] += 1

    def on_rendered(self):
        with self.lock:
            self._counters["rendered"] += 1

    def on_published(self):
        with self.lock:
            self._counters["published"] += 1

    def on_image_output_materialized(self):
        with self.lock:
            self._counters["imageOutputHostMaterializations"] += 1

    def _approx_quantile_locked(self, quantile):
        if not self._latency_count:
            return None
        target = max(1, math.ceil(self._latency_count * quantile))
        seen = 0
        for index, count in enumerate(self._latency_bucket_counts):
            seen += count
            if seen >= target:
                if index < len(LATENCY_BOUNDS_MS):
                    return LATENCY_BOUNDS_MS[index]
                # The overflow bucket has no finite upper bound. Its observed
                # max is truthful and more useful than inventing one.
                return round(self._latency_max_ms, 1)
        return round(self._latency_max_ms, 1)

    def snapshot(self, runtime=None):
        now = self._clock()
        with self.lock:
            delivered_fps = None
            if (
                self.frames > 1
                and self.first_result_at is not None
                and self._last_frame_time > self.first_result_at
            ):
                # Report delivered throughput, not an EMA of instantaneous
                # 1/dt samples. The latter explodes when an RTSP decoder emits
                # several frames in a burst: the uploaded-file control really
                # delivered 4.91 FPS while its old stats.fps ranged as high as
                # 82.5 and finished at 27.62. End-to-end count/time remains
                # truthful for both evenly paced and bursty delivery.
                delivered_fps = (self.frames - 1) / (
                    self._last_frame_time - self.first_result_at
                )
            counters = dict(self._counters)
            cumulative_counts = []
            cumulative = 0
            for count in self._latency_bucket_counts:
                cumulative += count
                cumulative_counts.append(cumulative)
            latency = {
                "unit": "ms",
                "count": self._latency_count,
                "sum": round(self._latency_sum_ms, 1),
                "mean": (
                    round(self._latency_sum_ms / self._latency_count, 1)
                    if self._latency_count
                    else None
                ),
                "min": (
                    round(self._latency_min_ms, 1)
                    if self._latency_min_ms is not None
                    else None
                ),
                "max": (
                    round(self._latency_max_ms, 1)
                    if self._latency_max_ms is not None
                    else None
                ),
                "p50Approx": self._approx_quantile_locked(0.50),
                "p95Approx": self._approx_quantile_locked(0.95),
                "p99Approx": self._approx_quantile_locked(0.99),
                "histogram": {
                    "bounds": list(LATENCY_BOUNDS_MS) + [None],
                    "cumulativeCounts": cumulative_counts,
                },
            }
            timing = {
                "runningS": (
                    round(max(0.0, now - self.job_received_at), 2)
                    if self.job_received_at
                    else None
                ),
                "pipelineStartS": (
                    round(self.pipeline_started_at - self.job_received_at, 2)
                    if self.job_received_at and self.pipeline_started_at
                    else None
                ),
                "timeToFirstResultS": (
                    round(self.first_result_at - self.job_received_at, 2)
                    if self.job_received_at and self.first_result_at
                    else None
                ),
            }
            out = {
                "schemaVersion": STATS_SCHEMA_VERSION,
                # Backward-compatible fields used by existing UI/benchmarks.
                "frames": self.frames,
                "fps": round(delivered_fps, 2) if delivered_fps else None,
                "decodeToResultLatencyMs": (
                    round(self.ema_latency_ms, 1)
                    if self.ema_latency_ms is not None
                    else None
                ),
                "counters": counters,
                "decodeToResultLatency": latency,
                "timing": timing,
            }
            if timing["pipelineStartS"] is not None:
                out["pipelineStartS"] = timing["pipelineStartS"]
            if timing["timeToFirstResultS"] is not None:
                out["timeToFirstResultS"] = timing["timeToFirstResultS"]
            if runtime:
                out["runtime"] = dict(runtime)
            return out
