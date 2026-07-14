"""Benchmark scalar-scheduled and direct SIMD tensor tracker workflow paths.

The harness keeps every numeric detection field on CUDA and reports synchronized
p50/p95 latency for the complete scalar and direct workflow calls. It also
decomposes the direct path into Tracktors execution, native-output recovery,
tensor InstanceCache/result construction, and the explicit serialization boundary.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import math
import statistics
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tracktors

import inference.core.workflows.core_steps.trackers.batch_scheduler as scheduler_module
from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_sv_detections,
)
from inference.core.workflows.core_steps.trackers._base_tensor import OUTPUT_KEY
from inference.core.workflows.core_steps.trackers.batch_scheduler import (
    TrackerBatchScheduler,
)
from inference.core.workflows.core_steps.trackers.bytetrack.v1_tensor import (
    ByteTrackBlockV1,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference_models.models.base.object_detection import Detections

_TRACKER_KWARGS = {
    "lost_track_buffer": 30,
    "minimum_iou_threshold": 0.1,
    "minimum_consecutive_frames": 1,
    "track_activation_threshold": 0.7,
    "high_conf_det_threshold": 0.6,
}
_CACHE_SIZE = 16384


@dataclass(frozen=True)
class Percentiles:
    """Hold synchronized p50 and p95 wall-clock latency in milliseconds."""

    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class ExecutionCounters:
    """Hold mean persistent-executor path counters for one workflow frame."""

    update_batches: float
    whole_frame_batches: float
    bytetrack_whole_frame_batches: float
    bytetrack_canonical_batches: float
    packed_batches: float

    @property
    def path(self) -> str:
        """Name the observed executor path without inferring from call count."""
        if self.bytetrack_whole_frame_batches > 0:
            return "bytetrack-whole-frame"
        if self.packed_batches > 0:
            return "ragged-packed"
        return "exact-fallback"


class CountingScheduler(TrackerBatchScheduler):
    """Count persistent executor invocations without changing scheduler behavior."""

    def __init__(
        self,
        *,
        batch_window_ms: float,
        max_batch_size: int,
    ) -> None:
        """Initialize the scheduler and its invocation counter."""
        super().__init__(
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
        )
        self.execute_calls = 0

    def execute_batch(
        self,
        trackers: list[Any],
        detections: list[Any],
        *,
        frames: list[Any | None],
        timestamps: list[float | None],
    ) -> list[Any]:
        """Count and forward one persistent-executor invocation."""
        self.execute_calls += 1
        return super().execute_batch(
            trackers,
            detections,
            frames=frames,
            timestamps=timestamps,
        )


def _percentiles(samples: list[float]) -> Percentiles:
    """Return deterministic nearest-rank p50 and p95 values."""
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return Percentiles(
        p50_ms=statistics.median(ordered),
        p95_ms=ordered[p95_index],
    )


def _synchronized_call(operation: Callable[[], Any]) -> tuple[Any, float]:
    """Run one operation between CUDA synchronization boundaries."""
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - started) * 1_000.0


def _executor_snapshot(scheduler: CountingScheduler) -> tuple[int, int, int, int]:
    """Read path-specific counters from the persistent CUDA executor."""
    executor = scheduler._executor
    if executor is None:
        return (0, 0, 0, 0)
    return (
        executor.whole_frame_batches,
        executor.bytetrack_whole_frame_batches,
        executor.bytetrack_canonical_batches,
        executor.packed_batches,
    )


def _execution_counters(
    update_batches: list[int],
    snapshots: list[tuple[int, int, int, int]],
) -> ExecutionCounters:
    """Aggregate per-frame deltas into path counters for one benchmark shape."""
    return ExecutionCounters(
        update_batches=statistics.mean(update_batches),
        whole_frame_batches=statistics.mean(item[0] for item in snapshots),
        bytetrack_whole_frame_batches=statistics.mean(item[1] for item in snapshots),
        bytetrack_canonical_batches=statistics.mean(item[2] for item in snapshots),
        packed_batches=statistics.mean(item[3] for item in snapshots),
    )


def _boxes(
    object_count: int,
    stream_index: int,
    device: torch.device,
) -> torch.Tensor:
    """Build stable non-overlapping boxes for persistent full-match updates."""
    rows = torch.arange(object_count, dtype=torch.float32, device=device)
    left = torch.remainder(rows, 16) * 24.0 + stream_index * 512.0
    top = torch.div(rows, 16, rounding_mode="floor") * 24.0
    return torch.stack((left, top, left + 12.0, top + 12.0), dim=1)


def _detections(
    object_count: int,
    stream_index: int,
    device: torch.device,
    metadata_mode: str,
) -> Detections:
    """Build serializable native detections with CUDA-resident numeric fields."""
    bboxes_metadata = None
    if metadata_mode == "strings":
        bboxes_metadata = [
            {DETECTION_ID_KEY: f"stream-{stream_index}-object-{row}"}
            for row in range(object_count)
        ]
    return Detections(
        xyxy=_boxes(object_count, stream_index, device),
        class_id=torch.zeros(object_count, dtype=torch.long, device=device),
        confidence=torch.full(
            (object_count,),
            0.95,
            dtype=torch.float32,
            device=device,
        ),
        image_metadata={
            CLASS_NAMES_KEY: {0: "object"},
            IMAGE_DIMENSIONS_KEY: (1080, 1920),
        },
        bboxes_metadata=bboxes_metadata,
    )


def _image(stream_index: int) -> WorkflowImageData:
    """Build one minimal workflow image for independent video state."""
    video_id = f"stream-{stream_index}"
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=1,
        frame_timestamp=datetime.datetime.fromtimestamp(
            1,
            tz=datetime.timezone.utc,
        ),
        fps=30,
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=video_id),
        numpy_image=np.zeros((1, 1, 3), dtype=np.uint8),
        video_metadata=metadata,
    )


def _inputs(
    stream_count: int,
    object_count: int,
    device: torch.device,
    metadata_mode: str,
) -> tuple[list[WorkflowImageData], list[Detections]]:
    """Build one aligned multi-stream input set."""
    images = [_image(index) for index in range(stream_count)]
    detections = [
        _detections(object_count, index, device, metadata_mode)
        for index in range(stream_count)
    ]
    return images, detections


def _batch(
    values: list[Any],
) -> Batch[Any]:
    """Wrap aligned values in the exact execution-engine Batch carrier."""
    return Batch.init(values, [(index,) for index in range(len(values))])


def _install_scheduler(scheduler: CountingScheduler) -> None:
    """Install one isolated scheduler behind the workflow's global accessor."""
    existing = scheduler_module._GLOBAL_SCHEDULER
    if existing is not None and existing is not scheduler:
        existing.close()
    scheduler_module._GLOBAL_SCHEDULER = scheduler


def _close_scheduler(scheduler: CountingScheduler) -> None:
    """Close and detach one isolated benchmark scheduler."""
    scheduler.close()
    if scheduler_module._GLOBAL_SCHEDULER is scheduler:
        scheduler_module._GLOBAL_SCHEDULER = None


def _run_scalar_samples(
    stream_count: int,
    object_count: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    batch_window_ms: float,
    metadata_mode: str,
) -> tuple[Percentiles, ExecutionCounters]:
    """Measure concurrent scalar calls entering the Future-based scheduler."""
    scheduler = CountingScheduler(
        batch_window_ms=batch_window_ms,
        max_batch_size=stream_count,
    )
    _install_scheduler(scheduler)
    block = ByteTrackBlockV1()
    images, detections = _inputs(
        stream_count,
        object_count,
        device,
        metadata_mode,
    )
    barrier = threading.Barrier(stream_count)

    def update(index: int) -> Any:
        """Release all scalar callers together before entering scheduler.update."""
        barrier.wait()
        return block.run(
            image=images[index],
            detections=detections[index],
            instances_cache_size=_CACHE_SIZE,
            **_TRACKER_KWARGS,
        )

    samples: list[float] = []
    call_counts: list[int] = []
    path_deltas: list[tuple[int, int, int, int]] = []
    try:
        with ThreadPoolExecutor(max_workers=stream_count) as pool:
            for sample_index in range(warmup + repeats):
                before = scheduler.execute_calls
                path_before = _executor_snapshot(scheduler)

                def run_all() -> list[Any]:
                    """Submit and join one scalar workflow frame per stream."""
                    futures = [
                        pool.submit(update, index) for index in range(stream_count)
                    ]
                    return [future.result() for future in futures]

                _, elapsed_ms = _synchronized_call(run_all)
                if sample_index >= warmup:
                    samples.append(elapsed_ms)
                    call_counts.append(scheduler.execute_calls - before)
                    path_after = _executor_snapshot(scheduler)
                    path_deltas.append(
                        tuple(
                            after - prior
                            for prior, after in zip(path_before, path_after)
                        )
                    )
    finally:
        _close_scheduler(scheduler)
    return _percentiles(samples), _execution_counters(call_counts, path_deltas)


def _run_direct_samples(
    stream_count: int,
    object_count: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    metadata_mode: str,
) -> tuple[Percentiles, ExecutionCounters]:
    """Measure one direct execution-engine Batch call per workflow frame."""
    scheduler = CountingScheduler(batch_window_ms=0.0, max_batch_size=stream_count)
    _install_scheduler(scheduler)
    block = ByteTrackBlockV1()
    images, detections = _inputs(
        stream_count,
        object_count,
        device,
        metadata_mode,
    )
    image_batch = _batch(images)
    detection_batch = _batch(detections)
    samples: list[float] = []
    call_counts: list[int] = []
    path_deltas: list[tuple[int, int, int, int]] = []
    try:
        for sample_index in range(warmup + repeats):
            before = scheduler.execute_calls
            path_before = _executor_snapshot(scheduler)
            _, elapsed_ms = _synchronized_call(
                lambda: block.run(
                    image=image_batch,
                    detections=detection_batch,
                    instances_cache_size=_CACHE_SIZE,
                    **_TRACKER_KWARGS,
                )
            )
            if sample_index >= warmup:
                samples.append(elapsed_ms)
                call_counts.append(scheduler.execute_calls - before)
                path_after = _executor_snapshot(scheduler)
                path_deltas.append(
                    tuple(
                        after - prior for prior, after in zip(path_before, path_after)
                    )
                )
    finally:
        _close_scheduler(scheduler)
    return _percentiles(samples), _execution_counters(call_counts, path_deltas)


def _serialize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cross the real workflow serialization boundary for all tracker outputs."""
    return [
        {
            key: serialise_sv_detections(result[key])
            for key in (
                OUTPUT_KEY,
                "new_instances",
                "already_seen_instances",
            )
        }
        for result in results
    ]


def _attach_serialization_metadata(results: list[dict[str, Any]]) -> None:
    """Add required string IDs only at the tensor-only serialization boundary."""
    for stream_index, result in enumerate(results):
        for key in (
            OUTPUT_KEY,
            "new_instances",
            "already_seen_instances",
        ):
            prediction = result[key]
            if prediction.bboxes_metadata is not None:
                continue
            prediction.bboxes_metadata = [
                {DETECTION_ID_KEY: (f"stream-{stream_index}-{key}-object-{row}")}
                for row in range(len(prediction))
            ]


def _run_phase_samples(
    stream_count: int,
    object_count: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    metadata_mode: str,
) -> dict[str, Percentiles]:
    """Measure synchronized direct-path phases without changing core code."""
    scheduler = CountingScheduler(batch_window_ms=0.0, max_batch_size=stream_count)
    _install_scheduler(scheduler)
    block = ByteTrackBlockV1()
    images, detections = _inputs(
        stream_count,
        object_count,
        device,
        metadata_mode,
    )
    prepared = [
        block._prepare_tracker_input(
            image=image,
            detections=prediction,
            tracker_kwargs=_TRACKER_KWARGS,
        )
        for image, prediction in zip(images, detections)
    ]
    trackers = [item[1] for item in prepared]
    sv_inputs = [item[3] for item in prepared]
    phase_samples = {
        "tracktors_ms": [],
        "native_output_recovery_ms": [],
        "instance_cache_ms": [],
        "serialization_three_output_worst_case_ms": [],
    }
    try:
        for sample_index in range(warmup + repeats):
            tracked_outputs, tracktors_ms = _synchronized_call(
                lambda: scheduler.execute_batch(
                    trackers,
                    sv_inputs,
                    frames=[None] * stream_count,
                    timestamps=[None] * stream_count,
                )
            )

            def recover() -> list[tuple[Any, torch.Tensor]]:
                """Recover native rows and same-device IDs for every stream."""
                return [
                    block._recover_tracker_output(
                        detections=prediction,
                        bbox=prepared_item[2],
                        tracked_sv=tracked_output,
                    )
                    for prediction, prepared_item, tracked_output in zip(
                        detections,
                        prepared,
                        tracked_outputs,
                    )
                ]

            recovered, recovery_ms = _synchronized_call(recover)

            def build_results() -> list[dict[str, Any]]:
                """Run tensor InstanceCache classification and result slicing."""
                return [
                    block._build_tracker_result(
                        video_id=prepared_item[0],
                        tracked_detections=tracked_detections,
                        tracker_ids=tracker_ids,
                        instances_cache_size=_CACHE_SIZE,
                    )
                    for prepared_item, (tracked_detections, tracker_ids) in zip(
                        prepared,
                        recovered,
                    )
                ]

            results, cache_ms = _synchronized_call(build_results)
            if metadata_mode == "tensor-only":
                _attach_serialization_metadata(results)
            _, serialization_ms = _synchronized_call(
                lambda: _serialize_results(results)
            )
            if sample_index >= warmup:
                phase_samples["tracktors_ms"].append(tracktors_ms)
                phase_samples["native_output_recovery_ms"].append(recovery_ms)
                phase_samples["instance_cache_ms"].append(cache_ms)
                phase_samples["serialization_three_output_worst_case_ms"].append(
                    serialization_ms
                )
    finally:
        _close_scheduler(scheduler)
    return {name: _percentiles(samples) for name, samples in phase_samples.items()}


def _result(
    stream_count: int,
    object_count: int,
    repeats: int,
    warmup: int,
    device: torch.device,
    batch_window_ms: float,
    metadata_mode: str,
) -> dict[str, Any]:
    """Benchmark one stream/object shape and return a JSON-safe record."""
    scalar, scalar_counters = _run_scalar_samples(
        stream_count,
        object_count,
        repeats,
        warmup,
        device,
        batch_window_ms,
        metadata_mode,
    )
    direct, direct_counters = _run_direct_samples(
        stream_count,
        object_count,
        repeats,
        warmup,
        device,
        metadata_mode,
    )
    phases = _run_phase_samples(
        stream_count,
        object_count,
        repeats,
        warmup,
        device,
        metadata_mode,
    )
    return {
        "streams": stream_count,
        "objects": object_count,
        "metadata_mode": metadata_mode,
        "scalar_scheduler_p50_ms": scalar.p50_ms,
        "scalar_scheduler_p95_ms": scalar.p95_ms,
        "scalar_update_batch_calls_per_frame": scalar_counters.update_batches,
        "scalar_executor_path": scalar_counters.path,
        "scalar_whole_frame_batches_per_frame": scalar_counters.whole_frame_batches,
        "scalar_bytetrack_whole_frame_batches_per_frame": (
            scalar_counters.bytetrack_whole_frame_batches
        ),
        "scalar_bytetrack_canonical_batches_per_frame": (
            scalar_counters.bytetrack_canonical_batches
        ),
        "scalar_packed_batches_per_frame": scalar_counters.packed_batches,
        "direct_batch_p50_ms": direct.p50_ms,
        "direct_batch_p95_ms": direct.p95_ms,
        "direct_update_batch_calls_per_frame": direct_counters.update_batches,
        "direct_executor_path": direct_counters.path,
        "direct_whole_frame_batches_per_frame": direct_counters.whole_frame_batches,
        "direct_bytetrack_whole_frame_batches_per_frame": (
            direct_counters.bytetrack_whole_frame_batches
        ),
        "direct_bytetrack_canonical_batches_per_frame": (
            direct_counters.bytetrack_canonical_batches
        ),
        "direct_packed_batches_per_frame": direct_counters.packed_batches,
        "direct_speedup": scalar.p50_ms / direct.p50_ms,
        "scalar_aggregate_fps": stream_count * 1_000.0 / scalar.p50_ms,
        "direct_aggregate_fps": stream_count * 1_000.0 / direct.p50_ms,
        **{
            f"{name}_{percentile}": getattr(timing, f"{percentile}_ms")
            for name, timing in phases.items()
            for percentile in ("p50", "p95")
        },
    }


def main() -> None:
    """Run the requested CUDA matrix and emit one JSON record per shape."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument(
        "--objects",
        nargs="+",
        type=int,
        default=[8, 32, 64, 128],
    )
    parser.add_argument(
        "--metadata-mode",
        nargs="+",
        choices=("tensor-only", "strings"),
        default=["tensor-only", "strings"],
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-window-ms", type=float, default=2.0)
    parser.add_argument("--tracktors-commit", default="unknown")
    parser.add_argument("--inference-commit", default="unknown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires an NVIDIA CUDA device")
    device = torch.device("cuda")
    records: list[dict[str, Any]] = []
    for metadata_mode in args.metadata_mode:
        for object_count in args.objects:
            for stream_count in args.streams:
                record = _result(
                    stream_count,
                    object_count,
                    args.repeats,
                    args.warmup,
                    device,
                    args.batch_window_ms,
                    metadata_mode,
                )
                record["tracktors_source"] = tracktors.__file__
                record["tracktors_distribution_version"] = importlib.metadata.version(
                    "tracktors"
                )
                record["tracktors_commit"] = args.tracktors_commit
                record["inference_commit"] = args.inference_commit
                record["torch_version"] = torch.__version__
                record["cuda_device"] = torch.cuda.get_device_name(device)
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    if args.output is not None:
        args.output.write_text(
            "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records)
        )


if __name__ == "__main__":
    main()
