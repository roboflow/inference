"""Benchmark the exact batched tracker instance cache on NVIDIA GPUs.

This harness exercises the persistent FIFO/hash state directly, validates its
decisions against independent ``deque`` references, and reports synchronized
end-to-end p50/p95 latency. The Triton kernel has not yet been compiled or
measured on Jetson Thor; keep that qualification attached to pre-Thor results.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections import deque
from dataclasses import dataclass

import torch

from inference.core.workflows.core_steps.trackers._base_tensor import (
    InstanceCache,
    _InstanceCacheBatchArena,
)
from inference.core.workflows.core_steps.trackers.instance_cache_kernels import (
    has_triton_instance_cache,
)


@dataclass(frozen=True)
class Percentiles:
    """Hold synchronized wall-clock latency percentiles in milliseconds."""

    p50_ms: float
    p95_ms: float


def _percentiles(samples: list[float]) -> Percentiles:
    """Return median and nearest-rank p95 for non-empty samples."""
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return Percentiles(
        p50_ms=statistics.median(ordered),
        p95_ms=ordered[p95_index],
    )


def _frame_ids(
    stream: int,
    frame: int,
    object_count: int,
    working_set: int,
    device: torch.device,
) -> torch.Tensor:
    """Build deterministic churn with duplicates and periodic FIFO evictions."""
    positions = torch.arange(object_count, dtype=torch.long, device=device)
    tracker_ids = torch.remainder(positions + frame * 3, working_set)
    return tracker_ids + stream * (working_set + 1)


def _reference_record(history: deque[int], tracker_ids: list[int]) -> list[bool]:
    """Apply the exact non-refreshing sequential FIFO contract on the CPU."""
    decisions = []
    for tracker_id in tracker_ids:
        seen = tracker_id in history
        decisions.append(seen)
        if not seen:
            history.append(tracker_id)
    return decisions


def _validate(
    *,
    stream_count: int,
    object_count: int,
    cache_size: int,
    working_set: int,
    frames: int,
    device: torch.device,
) -> None:
    """Validate batched device decisions against exact independent deques."""
    caches = [InstanceCache(size=cache_size) for _ in range(stream_count)]
    arena = _InstanceCacheBatchArena(size=cache_size, device=device)
    references = [deque(maxlen=cache_size) for _ in range(stream_count)]
    for frame in range(frames):
        tracker_ids = [
            _frame_ids(stream, frame, object_count, working_set, device)
            for stream in range(stream_count)
        ]
        result = arena.record_instances(caches=caches, tracker_ids=tracker_ids)
        actual = result.seen.detach().cpu().tolist()
        expected = []
        for history, ids in zip(references, tracker_ids):
            expected.extend(_reference_record(history, ids.cpu().tolist()))
        if actual != expected:
            raise AssertionError(f"FIFO/hash parity failed at frame {frame}")


def _measure(
    *,
    stream_count: int,
    object_count: int,
    cache_size: int,
    working_set: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> Percentiles:
    """Measure synchronized host-observed latency with persistent cache state."""
    caches = [InstanceCache(size=cache_size) for _ in range(stream_count)]
    arena = _InstanceCacheBatchArena(size=cache_size, device=device)
    samples = []
    for frame in range(warmup + iterations):
        tracker_ids = [
            _frame_ids(stream, frame, object_count, working_set, device)
            for stream in range(stream_count)
        ]
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = arena.record_instances(caches=caches, tracker_ids=tracker_ids)
        result.partition_counts.detach().cpu()
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if frame >= warmup:
            samples.append(elapsed_ms)
    return _percentiles(samples)


def main() -> None:
    """Validate and benchmark requested stream and object-count shapes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--objects", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--cache-size", type=int, default=16384)
    parser.add_argument("--working-set", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--validation-frames", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not has_triton_instance_cache():
        raise SystemExit("Triton instance-cache kernel is unavailable")
    if args.working_set <= args.cache_size:
        raise SystemExit("working-set must exceed cache-size to exercise eviction")
    device = torch.device("cuda")
    print("streams objects total_ids p50_ms p95_ms ids_per_second")
    for stream_count in args.streams:
        for object_count in args.objects:
            _validate(
                stream_count=stream_count,
                object_count=object_count,
                cache_size=args.cache_size,
                working_set=args.working_set,
                frames=args.validation_frames,
                device=device,
            )
            latency = _measure(
                stream_count=stream_count,
                object_count=object_count,
                cache_size=args.cache_size,
                working_set=args.working_set,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
            total_ids = stream_count * object_count
            ids_per_second = total_ids / (latency.p50_ms / 1_000.0)
            print(
                f"{stream_count} {object_count} {total_ids} "
                f"{latency.p50_ms:.4f} {latency.p95_ms:.4f} "
                f"{ids_per_second:.0f}"
            )


if __name__ == "__main__":
    main()
