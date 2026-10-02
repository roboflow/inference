"""Timing records, raw CSV output and the run summary.

Timestamps per frame (``time.monotonic_ns`` unless noted):

    reader_wallclock  VideoFrame.frame_timestamp: wall clock taken by the
                      VideoSource capture thread just BEFORE grab(). Not a
                      source PTS: the native bridge (ABI 7) exposes no PTS.
    arrival           the reader thread received the frame from VideoSource
                      (decoder handoff)
    admit             the runner handed the frame to the backend
    ready             process() returned / the backend Future completed;
                      the backend guarantees its GPU work is done by then

    queue_ms   admit - arrival     time waiting in the runner slot
    service_ms ready - admit       backend time, including pipeline overlap
    age_ms     ready - arrival     receive-to-ready frame age

``age_ms`` is not camera-to-display latency: capture, network, RTSP jitter
buffer and decode happen before ``arrival``.

Measurement window: a frame is measured when ``admit`` falls inside
[warmup end, warmup end + duration) and ``ready`` is before the window end.
Frames admitted in the window that finish after it are counted as ``tail``.
"""

import csv
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

QUANTILES = (50, 90, 95, 99)
LATENCY_NAMES = ("queue_ms", "service_ms", "age_ms")
RAW_COLUMNS = (
    "phase",
    "source",
    "frame_id",
    "reader_wallclock_s",
    "arrival_ns",
    "admit_ns",
    "ready_ns",
    "queue_ms",
    "service_ms",
    "age_ms",
    "result",
)


@dataclass(frozen=True)
class FrameTiming:
    """Timing of one completed frame.

    Attributes:
        source_index: Source of the frame.
        frame_id: VideoSource grab counter of the frame.
        reader_wallclock_s: Epoch seconds before the capture thread's grab().
        arrival_ns: Reader handoff, monotonic.
        admit_ns: Backend admission, monotonic.
        ready_ns: Backend completion (GPU-ready), monotonic.
        result: Small JSON-serialisable result summary, or None.
    """

    source_index: int
    frame_id: int
    reader_wallclock_s: float
    arrival_ns: int
    admit_ns: int
    ready_ns: int
    result: Optional[dict] = None


class RunRecorder:
    """Writes every completion to ``frames.csv`` and keeps measured latencies.

    Memory grows with measured frames only (three floats per frame).

    Args:
        output_dir: Directory for ``frames.csv``.
        source_count: Number of sources.
        window_start_ns: Monotonic start of the measurement window.
        window_end_ns: Monotonic end of the measurement window.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        source_count: int,
        window_start_ns: int,
        window_end_ns: int,
    ):
        self.window_start_ns = window_start_ns
        self.window_end_ns = window_end_ns
        self.phase_counts = {"warmup": 0, "measured": 0, "tail": 0}
        self._latencies: List[Dict[str, List[float]]] = [
            {name: [] for name in LATENCY_NAMES} for _ in range(source_count)
        ]
        self._file = (output_dir / "frames.csv").open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(RAW_COLUMNS)

    def record(self, timing: FrameTiming) -> str:
        """Store one completion.

        Args:
            timing: The completed frame.

        Returns:
            The frame's phase: ``warmup``, ``measured`` or ``tail``.
        """
        phase = self._phase(timing)
        self.phase_counts[phase] += 1

        latencies = {
            "queue_ms": (timing.admit_ns - timing.arrival_ns) / 1e6,
            "service_ms": (timing.ready_ns - timing.admit_ns) / 1e6,
            "age_ms": (timing.ready_ns - timing.arrival_ns) / 1e6,
        }
        if phase == "measured":
            for name, value in latencies.items():
                self._latencies[timing.source_index][name].append(value)

        self._writer.writerow(
            (
                phase,
                timing.source_index,
                timing.frame_id,
                f"{timing.reader_wallclock_s:.6f}",
                timing.arrival_ns,
                timing.admit_ns,
                timing.ready_ns,
                f"{latencies['queue_ms']:.3f}",
                f"{latencies['service_ms']:.3f}",
                f"{latencies['age_ms']:.3f}",
                "" if timing.result is None else json.dumps(timing.result),
            )
        )

        return phase

    def measured_counts(self) -> List[int]:
        """Measured completions per source."""
        counts = [len(source["age_ms"]) for source in self._latencies]

        return counts

    def latency_summary(self) -> dict:
        """Exact quantiles of measured latencies, aggregate and per source."""
        aggregate = {
            name: _distribution(
                [value for source in self._latencies for value in source[name]]
            )
            for name in LATENCY_NAMES
        }
        per_source = [
            {name: _distribution(source[name]) for name in LATENCY_NAMES}
            for source in self._latencies
        ]
        summary = {
            "aggregate": aggregate,
            "per_source": per_source,
        }

        return summary

    def close(self) -> None:
        """Flush and close ``frames.csv``."""
        self._file.close()

    def _phase(self, timing: FrameTiming) -> str:
        if timing.admit_ns < self.window_start_ns:
            return "warmup"
        if timing.ready_ns < self.window_end_ns:
            return "measured"

        return "tail"


class ProgressPrinter:
    """Concise periodic progress line: completions per second since last print.

    Args:
        interval_seconds: Seconds between lines.
        source_count: Number of sources.
    """

    def __init__(self, *, interval_seconds: float, source_count: int):
        self._interval_ns = int(interval_seconds * 1e9)
        self._source_count = source_count
        self._last_ns = time.monotonic_ns()
        self._started_ns = self._last_ns
        self._per_source = [0] * source_count

    def completed(self, source_index: int) -> None:
        """Count one completion.

        Args:
            source_index: Source of the completed frame.
        """
        self._per_source[source_index] += 1

    def maybe_print(
        self,
        now_ns: int,
        *,
        phase: str,
        in_flight: int,
        capacity: int,
    ) -> None:
        """Print a line when the interval elapsed.

        Args:
            now_ns: Current monotonic time.
            phase: ``warmup`` or ``measure``.
            in_flight: Frames currently admitted and not ready.
            capacity: Maximum frames in flight.
        """
        elapsed_ns = now_ns - self._last_ns
        if elapsed_ns < self._interval_ns:
            return

        seconds = elapsed_ns / 1e9
        rates = [count / seconds for count in self._per_source]
        print(
            f"[{(now_ns - self._started_ns) / 1e9:7.1f}s {phase:7s}] "
            f"{sum(rates):7.1f} fps total | per source "
            f"{min(rates):5.1f}..{max(rates):5.1f} | "
            f"in flight {in_flight}/{capacity}",
            flush=True,
        )
        self._per_source = [0] * self._source_count
        self._last_ns = now_ns


def process_cpu_seconds() -> float:
    """User plus system CPU seconds of this process (all threads)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_seconds = usage.ru_utime + usage.ru_stime

    return cpu_seconds


def summarize_sources(
    *,
    start: Sequence[dict],
    end: Sequence[dict],
    measured_counts: Sequence[int],
    declared_fps: Sequence[Optional[float]],
    duration_seconds: float,
) -> dict:
    """Per-source throughput, drops and fairness over the window.

    ``upstream_unobserved_estimate`` = declared source FPS x window - frames the
    native bridge converted. It estimates frames lost before the bridge
    (network, RTSP jitter buffer, decoder); it is None without a declared FPS.

    Args:
        start: ``JetsonSourceSet.snapshot()`` at window start.
        end: ``JetsonSourceSet.snapshot()`` at window end.
        measured_counts: Measured completions per source.
        declared_fps: FPS each source announces, or None.
        duration_seconds: Window length.

    Returns:
        Per-source rows, totals and fairness figures.
    """
    rows = []
    for index, (before, after, completed, fps) in enumerate(
        zip(start, end, measured_counts, declared_fps)
    ):
        counters = _delta(before["counters"], after["counters"])
        native = None
        upstream_unobserved = None
        if before["native"] is not None and after["native"] is not None:
            native = _delta(before["native"], after["native"])
            if fps:
                upstream_unobserved = fps * duration_seconds - native["frames"]
        rows.append(
            {
                "source": index,
                "completed_fps": completed / duration_seconds,
                "completed": completed,
                "runner": counters,
                "native_bridge": native,
                "declared_fps": fps,
                "upstream_unobserved_estimate": upstream_unobserved,
            }
        )

    rates = [row["completed_fps"] for row in rows]
    totals = {
        "completed_fps": sum(rates),
        "completed": sum(measured_counts),
        "arrived": sum(row["runner"]["arrived"] for row in rows),
        "replaced_in_slot": sum(row["runner"]["replaced_in_slot"] for row in rows),
        "video_source_dropped": sum(
            row["runner"]["video_source_dropped"] for row in rows
        ),
        "native_dropped_by_consumer": _sum_native(rows, "frames_dropped_by_consumer"),
        "native_frames": _sum_native(rows, "frames"),
    }
    summary = {
        "per_source": rows,
        "totals": totals,
        "fairness": _fairness(rates),
    }

    return summary


def host_fallback_violations(
    native_snapshots: Sequence[Optional[dict]],
    *,
    keys: Sequence[str],
) -> List[dict]:
    """Sources whose native host-copy counters are not zero.

    Args:
        native_snapshots: Native counters per source (None when unreadable).
        keys: Counters that must stay zero on the zero-copy path.

    Returns:
        One entry per offending source; empty when the path stayed on GPU.
    """
    violations = []
    for index, native in enumerate(native_snapshots):
        if native is None:
            continue
        nonzero = {key: native[key] for key in keys if native.get(key, 0) != 0}
        if nonzero:
            violations.append({"source": index, "counters": nonzero})

    return violations


def write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as indented JSON.

    Args:
        path: Output file.
        payload: JSON-serialisable data.
    """
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def _distribution(values: Sequence[float]) -> Optional[dict]:
    if not values:
        return None

    array = np.asarray(values, dtype=np.float64)
    distribution = {
        "count": int(array.size),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }
    for quantile in QUANTILES:
        distribution[f"p{quantile}"] = float(np.percentile(array, quantile))

    return distribution


def _fairness(rates: Sequence[float]) -> dict:
    if not rates or sum(rates) == 0:
        return {"jain_index": None, "min_over_max": None}

    # Jain's index: 1.0 when every source gets the same rate, 1/n when one
    # source gets everything.
    jain = sum(rates) ** 2 / (len(rates) * sum(rate * rate for rate in rates))
    fairness = {
        "jain_index": jain,
        "min_over_max": min(rates) / max(rates),
        "min_fps": min(rates),
        "median_fps": float(np.median(rates)),
        "max_fps": max(rates),
    }

    return fairness


def _delta(before: dict, after: dict) -> dict:
    # *_max_ns counters are running maxima, not totals: keep the end value.
    delta = {
        key: after[key] if key.endswith("_max_ns") else after[key] - before[key]
        for key in after
    }

    return delta


def _sum_native(rows: Sequence[dict], key: str) -> Optional[int]:
    if any(row["native_bridge"] is None for row in rows):
        return None

    total = sum(row["native_bridge"][key] for row in rows)

    return total
