from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from inference_models.logger import LOGGER
from inference_models.utils.environment import (
    get_boolean_from_env,
    get_float_from_env,
    get_integer_from_env,
)


class _MetricState:
    def __init__(self, max_samples: int, unit: str) -> None:
        self.observed_count = 0
        self.sampled_count = 0
        self.unit = unit
        self.samples: Deque[float] = deque(maxlen=max_samples)


class PerformanceProfiler:
    """Opt-in, sampled wall-clock profiler for the MMP hot path."""

    def __init__(self) -> None:
        self.enabled = get_boolean_from_env(
            "MMP_PERFORMANCE_PROFILING_ENABLED", default=False
        )
        self.sample_every_n = get_integer_from_env(
            "MMP_PERFORMANCE_PROFILING_SAMPLE_EVERY_N", default=1
        )
        self.warmup_calls = get_integer_from_env(
            "MMP_PERFORMANCE_PROFILING_WARMUP_CALLS", default=30
        )
        self.log_interval_s = get_float_from_env(
            "MMP_PERFORMANCE_PROFILING_LOG_INTERVAL_S", default=10.0
        )
        self.max_samples = get_integer_from_env(
            "MMP_PERFORMANCE_PROFILING_MAX_SAMPLES", default=1000
        )
        if self.sample_every_n < 1:
            raise ValueError("MMP_PERFORMANCE_PROFILING_SAMPLE_EVERY_N must be >= 1")
        if self.warmup_calls < 0:
            raise ValueError("MMP_PERFORMANCE_PROFILING_WARMUP_CALLS must be >= 0")
        if self.log_interval_s <= 0:
            raise ValueError("MMP_PERFORMANCE_PROFILING_LOG_INTERVAL_S must be > 0")
        if self.max_samples < 1:
            raise ValueError("MMP_PERFORMANCE_PROFILING_MAX_SAMPLES must be >= 1")

        self._lock = threading.Lock()
        self._metrics: Dict[str, _MetricState] = {}
        self._counters: Dict[str, int] = {}
        self._metadata: Dict[str, Any] = {}
        self._summary_sequence = 0
        self._last_flush_ns = time.perf_counter_ns() if self.enabled else 0

    def start(self) -> Optional[int]:
        if not self.enabled:
            return None
        return time.perf_counter_ns()

    def stop(
        self,
        metric: str,
        start_ns: Optional[int],
        end_ns: Optional[int] = None,
    ) -> None:
        if not self.enabled or start_ns is None:
            return
        if end_ns is None:
            end_ns = time.perf_counter_ns()
        self.record(metric, (end_ns - start_ns) / 1_000_000, "ms")

    def record(self, metric: str, value: float, unit: str) -> None:
        if not self.enabled:
            return
        numeric_value = float(value)
        with self._lock:
            state = self._metrics.get(metric)
            if state is None:
                state = _MetricState(max_samples=self.max_samples, unit=unit)
                self._metrics[metric] = state
            elif state.unit != unit:
                return
            state.observed_count += 1
            eligible_count = state.observed_count - self.warmup_calls
            if eligible_count > 0 and eligible_count % self.sample_every_n == 0:
                state.sampled_count += 1
                state.samples.append(numeric_value)

    def increment(self, counter: str, value: int = 1) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._counters[counter] = self._counters.get(counter, 0) + value

    def set_metadata(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._metadata[key] = value

    def flush(self, force: bool = False) -> Optional[dict]:
        if not self.enabled:
            return None
        now_ns = time.perf_counter_ns()
        with self._lock:
            interval_ns = int(self.log_interval_s * 1_000_000_000)
            if not force and now_ns - self._last_flush_ns < interval_ns:
                return None
            if not (self._metrics or self._counters or self._metadata):
                return None
            self._summary_sequence += 1
            summary = self._build_summary()
            self._last_flush_ns = now_ns
        LOGGER.warning(
            "[MMP-PERF] %s",
            json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str),
        )
        return summary

    def _build_summary(self) -> dict:
        metrics = {}
        for name, state in sorted(self._metrics.items()):
            samples = sorted(state.samples)
            metrics[name] = {
                "observed_count": state.observed_count,
                "sampled_count": state.sampled_count,
                "count": len(samples),
                "mean": sum(samples) / len(samples) if samples else None,
                "p50": _percentile(samples, 50),
                "p95": _percentile(samples, 95),
                "p99": _percentile(samples, 99),
                "max": max(samples) if samples else None,
                "unit": state.unit,
            }
        return {
            "pid": os.getpid(),
            "sequence": self._summary_sequence,
            "metrics": metrics,
            "counters": dict(sorted(self._counters.items())),
            "metadata": dict(sorted(self._metadata.items())),
            "config": {
                "enabled": self.enabled,
                "sample_every_n": self.sample_every_n,
                "warmup_calls": self.warmup_calls,
                "log_interval_s": self.log_interval_s,
                "max_samples": self.max_samples,
            },
        }


def _percentile(values: list[float], percentile: int) -> Optional[float]:
    if not values:
        return None
    position = (len(values) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


performance_profiler = PerformanceProfiler()


__all__ = ["PerformanceProfiler", "performance_profiler"]
