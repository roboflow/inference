from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Token
from time import perf_counter
from typing import Iterator, Optional


_current_profiler: ContextVar[Optional["RequestStageProfiler"]] = ContextVar(
    "request_stage_profiler",
    default=None,
)


class RequestStageProfiler:
    def __init__(self) -> None:
        self._stage_ms: dict[str, float] = {}
        self._stage_counts: dict[str, int] = {}

    @contextmanager
    def profile(self, stage: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.record(stage=stage, duration_ms=(perf_counter() - started) * 1000.0)

    def record(self, *, stage: str, duration_ms: float) -> None:
        self._stage_ms[stage] = self._stage_ms.get(stage, 0.0) + duration_ms
        self._stage_counts[stage] = self._stage_counts.get(stage, 0) + 1

    def record_derived(self, *, stage: str, duration_ms: float) -> None:
        self._stage_ms[stage] = max(0.0, duration_ms)
        self._stage_counts[stage] = 1

    def snapshot_ms(self) -> dict[str, float]:
        snapshot = {
            stage: round(duration_ms, 3)
            for stage, duration_ms in sorted(self._stage_ms.items())
        }

        return snapshot

    def snapshot_counts(self) -> dict[str, int]:
        counts = {
            stage: count
            for stage, count in sorted(self._stage_counts.items())
            if count != 1
        }

        return counts


def start_request_stage_profiling() -> tuple[RequestStageProfiler, Token]:
    profiler = RequestStageProfiler()
    token = _current_profiler.set(profiler)

    return profiler, token


def stop_request_stage_profiling(token: Token) -> None:
    _current_profiler.reset(token)


def get_request_stage_profiler() -> Optional[RequestStageProfiler]:
    profiler = _current_profiler.get()

    return profiler


def profile_request_stage(stage: str):
    profiler = get_request_stage_profiler()
    if profiler is None:
        return nullcontext()

    profile_context = profiler.profile(stage=stage)

    return profile_context
