import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from inference_models.utils import performance as performance_module
from inference_models.utils.performance import PerformanceProfiler


@pytest.fixture
def profiler_env(monkeypatch):
    def set_env(**overrides):
        values = {
            "MMP_PERFORMANCE_PROFILING_ENABLED": "true",
            "MMP_PERFORMANCE_PROFILING_SAMPLE_EVERY_N": "1",
            "MMP_PERFORMANCE_PROFILING_WARMUP_CALLS": "0",
            "MMP_PERFORMANCE_PROFILING_LOG_INTERVAL_S": "1000",
            "MMP_PERFORMANCE_PROFILING_MAX_SAMPLES": "1000",
        }
        values.update({key: str(value) for key, value in overrides.items()})
        for key, value in values.items():
            monkeypatch.setenv(key, value)

    return set_env


def test_disabled_profiler_is_a_noop(monkeypatch):
    monkeypatch.setenv("MMP_PERFORMANCE_PROFILING_ENABLED", "false")
    log = Mock()
    monkeypatch.setattr(performance_module.LOGGER, "warning", log)
    profiler = PerformanceProfiler()

    assert profiler.start() is None
    profiler.stop("stage", None)
    profiler.record("stage", 1, "ms")
    profiler.increment("calls")
    profiler.set_metadata("model", "example")

    assert profiler.flush(force=True) is None
    log.assert_not_called()


def test_warmup_sampling_and_bounded_retention(profiler_env):
    profiler_env(
        MMP_PERFORMANCE_PROFILING_WARMUP_CALLS=1,
        MMP_PERFORMANCE_PROFILING_SAMPLE_EVERY_N=2,
        MMP_PERFORMANCE_PROFILING_MAX_SAMPLES=2,
    )
    profiler = PerformanceProfiler()

    for value in range(1, 8):
        profiler.record("stage", value, "ms")

    metric = profiler.flush(force=True)["metrics"]["stage"]
    assert metric == {
        "observed_count": 7,
        "sampled_count": 3,
        "count": 2,
        "mean": 6.0,
        "p50": 6.0,
        "p95": 6.9,
        "p99": 6.98,
        "max": 7.0,
        "unit": "ms",
    }


def test_stop_records_elapsed_milliseconds(monkeypatch, profiler_env):
    profiler_env()
    monkeypatch.setattr(performance_module.time, "perf_counter_ns", lambda: 1_000_000)
    profiler = PerformanceProfiler()

    start_ns = profiler.start()
    profiler.stop("stage", start_ns, end_ns=6_000_000)

    metric = profiler.flush(force=True)["metrics"]["stage"]
    assert metric["count"] == 1
    assert metric["mean"] == 5.0
    assert metric["unit"] == "ms"


def test_ignores_mixed_units_for_a_metric(profiler_env):
    profiler_env()
    profiler = PerformanceProfiler()
    profiler.record("stage", 1, "ms")
    profiler.record("stage", 2, "s")

    metric = profiler.flush(force=True)["metrics"]["stage"]
    assert metric["observed_count"] == 1
    assert metric["count"] == 1
    assert metric["mean"] == 1
    assert metric["unit"] == "ms"


def test_flush_reports_context_and_respects_interval(monkeypatch, profiler_env):
    profiler_env(MMP_PERFORMANCE_PROFILING_LOG_INTERVAL_S=10)
    now = [0]
    monkeypatch.setattr(performance_module.time, "perf_counter_ns", lambda: now[0])
    log = Mock()
    monkeypatch.setattr(performance_module.LOGGER, "warning", log)
    profiler = PerformanceProfiler()
    profiler.record("stage", 2, "ms")
    profiler.increment("calls", 3)
    profiler.set_metadata("backend", "direct")

    assert profiler.flush() is None
    now[0] = 10_000_000_000
    summary = profiler.flush()

    assert summary["counters"] == {"calls": 3}
    assert summary["metadata"] == {"backend": "direct"}
    assert summary["pid"] == os.getpid()
    assert summary["sequence"] == 1
    assert summary["config"] == {
        "enabled": True,
        "sample_every_n": 1,
        "warmup_calls": 0,
        "log_interval_s": 10.0,
        "max_samples": 1000,
    }
    assert log.call_count == 1
    assert log.call_args.args[0].startswith("[MMP-PERF]")

    assert profiler.flush(force=True)["sequence"] == 2


def test_rejects_zero_log_interval(profiler_env):
    profiler_env(MMP_PERFORMANCE_PROFILING_LOG_INTERVAL_S=0)

    with pytest.raises(ValueError, match="must be > 0"):
        PerformanceProfiler()


def test_updates_are_thread_safe(profiler_env):
    profiler_env()
    profiler = PerformanceProfiler()

    def observe() -> None:
        for _ in range(100):
            profiler.record("stage", 1, "ms")
            profiler.increment("calls")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: observe(), range(8)))

    summary = profiler.flush(force=True)
    assert summary["metrics"]["stage"]["count"] == 800
    assert summary["counters"] == {"calls": 800}
