"""Multi-stream NVDEC detection benchmark on Jetson Thor.

```
N RTSP sources ─ NVDEC ─ latest slot per source ─┐
                                                 ├─ round-robin admission ─ backend
                       (same for every mode) ────┘   serial: process()  (1 in flight)
                                                     v2_pipeline: submit() (≤ N futures)
```

One process, one loaded model, one mode. Sources keep decoding during the
whole run; frames the backend cannot take are replaced and counted. The runner
adds no device synchronisation: a frame is ``ready`` when ``process()`` returns
or its Future completes, and the backend guarantees its GPU work is done then.
``--decoder-only`` runs the same sources and admission without a backend to
measure source capacity on its own.

Outputs in ``--output-dir``: ``summary.json`` and ``frames.csv`` (one row per
completed frame). See ``metrics.py`` for timestamp definitions.
"""

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Optional

import click
from backends import MODES

DEFAULT_URL_TEMPLATE = "rtsp://192.168.10.55:8554/hd/cam_thor_{index}"
DEFAULT_MODEL_ID = "yolov8n-640"
WAIT_SECONDS = 0.1
STOP_TIMEOUT_SECONDS = 10.0
DRAIN_TIMEOUT_SECONDS = 10.0


@dataclass
class _Completion:
    arrived: Any
    admit_ns: int
    ready_ns: int
    result: Any = None
    error: Optional[BaseException] = None


class Admission:
    """Hands frames to the backend and collects completions.

    Serial backends run ``process`` on the calling thread, one frame at a
    time. Pipeline backends get ``submit``; at most ``capacity`` futures are
    pending. Completions are queued under the sources' wakeup lock, which is
    notified so the main loop wakes on new frames and on completions alike.

    ``max_per_source`` records the most frames of one source in flight at
    once; the native bridge pools 8 buffers per source, and frames beyond the
    pool are allocated and freed per frame (a device-wide sync on Jetson).

    Args:
        wakeup: The source set's shared condition.
        process: Synchronous ``(frame, image_id=) -> result``, or None.
        submit: ``(frame, image_id=) -> Future``, or None.
        capacity: Maximum frames in flight.
        summarize: Small JSON summary of a result; called on the main thread.
    """

    def __init__(
        self,
        wakeup: threading.Condition,
        *,
        process: Optional[Callable] = None,
        submit: Optional[Callable] = None,
        capacity: int,
        summarize: Callable[[Any], Optional[dict]] = lambda result: None,
    ):
        self.capacity = capacity
        self.in_flight = 0
        self.max_per_source = 0
        self.failed_completions = 0
        self.wakeup = wakeup
        self._process = process
        self._submit = submit
        self.summarize = summarize
        self._completions: Deque[_Completion] = deque()
        self._per_source: Dict[int, int] = {}

    def admit(self, arrived) -> None:
        """Run or submit one frame; the caller does not hold the lock.

        Args:
            arrived: ``sources.ArrivedFrame``; kept referenced until ready, so
                the native frame buffer stays alive while the backend uses it.
        """
        image_id = f"s{arrived.source_index}-f{arrived.frame.frame_id}"
        with self.wakeup:
            self.in_flight += 1
            source_in_flight = self._per_source.get(arrived.source_index, 0) + 1
            self._per_source[arrived.source_index] = source_in_flight
            self.max_per_source = max(self.max_per_source, source_in_flight)
        admit_ns = time.monotonic_ns()

        if self._submit is None:
            completion = _Completion(arrived=arrived, admit_ns=admit_ns, ready_ns=0)
            try:
                completion.result = self._process(
                    arrived.frame.image, image_id=image_id
                )
                completion.ready_ns = time.monotonic_ns()
            except Exception as error:  # noqa: BLE001 - stops the run
                completion.error = error
            self._complete(completion)
            return

        try:
            future = self._submit(arrived.frame.image, image_id=image_id)
        except Exception as error:  # noqa: BLE001 - stops the run
            self._complete(
                _Completion(
                    arrived=arrived,
                    admit_ns=admit_ns,
                    ready_ns=time.monotonic_ns(),
                    error=error,
                )
            )
            return
        future.add_done_callback(
            lambda done: self._on_future_done(
                done,
                arrived=arrived,
                admit_ns=admit_ns,
            )
        )

    def take_completions_locked(self) -> list:
        """Remove queued completions; the caller holds the wakeup lock."""
        completions = list(self._completions)
        self._completions.clear()
        self.in_flight -= len(completions)
        for completion in completions:
            self._per_source[completion.arrived.source_index] -= 1

        return completions

    def _on_future_done(self, future, *, arrived, admit_ns: int) -> None:
        # Stamped in the completing thread, so the main loop's wake-up delay
        # is not part of the measured latency.
        ready_ns = time.monotonic_ns()
        completion = _Completion(arrived=arrived, admit_ns=admit_ns, ready_ns=ready_ns)
        try:
            completion.result = future.result()
        except BaseException as error:  # noqa: BLE001 - stops the run
            completion.error = error
        self._complete(completion)

    def _complete(self, completion: _Completion) -> None:
        with self.wakeup:
            self._completions.append(completion)
            if completion.error is not None:
                self.failed_completions += 1
            self.wakeup.notify_all()


@click.command()
@click.option(
    "--mode",
    type=click.Choice(MODES),
    default=None,
    help="Backend mode. Required unless --decoder-only.",
)
@click.option(
    "--sources",
    "source_count",
    type=click.IntRange(
        min=1,
    ),
    default=8,
    show_default=True,
    help="Number of RTSP sources.",
)
@click.option(
    "--url-template",
    default=DEFAULT_URL_TEMPLATE,
    show_default=True,
    help="Source URL; {index} is replaced by 0..N-1.",
)
@click.option(
    "--duration-seconds",
    type=click.FloatRange(
        min=1.0,
    ),
    default=60.0,
    show_default=True,
    help="Measurement window length.",
)
@click.option(
    "--warmup-seconds",
    type=click.FloatRange(
        min=0.0,
    ),
    default=15.0,
    show_default=True,
    help="Live traffic before the window; not measured.",
)
@click.option(
    "--max-in-flight",
    type=click.IntRange(
        min=1,
    ),
    default=4,
    show_default=True,
    help="Pending futures in v2_pipeline. Serial modes always use 1.",
)
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    required=True,
    help="Directory for summary.json and frames.csv.",
)
@click.option(
    "--decoder-only",
    is_flag=True,
    default=False,
    help="No model: admit and release frames to measure source capacity.",
)
@click.option(
    "--model-id",
    default=DEFAULT_MODEL_ID,
    show_default=True,
    help="Platform model ID; TensorRT is required, with no backend fallback.",
)
@click.option(
    "--confidence",
    type=click.FloatRange(
        min=0.0,
        max=1.0,
    ),
    default=0.4,
    show_default=True,
    help="Detection confidence threshold, identical in every mode.",
)
@click.option(
    "--device",
    default="cuda:0",
    show_default=True,
    help="CUDA device for decoding and the model.",
)
@click.option(
    "--progress-seconds",
    type=click.FloatRange(
        min=0.5,
    ),
    default=5.0,
    show_default=True,
    help="Interval of progress lines.",
)
def main(
    mode: Optional[str],
    source_count: int,
    url_template: str,
    duration_seconds: float,
    warmup_seconds: float,
    max_in_flight: int,
    output_dir: Path,
    decoder_only: bool,
    model_id: str,
    confidence: float,
    device: str,
    progress_seconds: float,
) -> None:
    """Run N NVDEC sources through one detection backend and write results."""
    if mode is None and not decoder_only:
        raise click.UsageError("--mode is required unless --decoder-only is set")
    if mode is not None and decoder_only:
        raise click.UsageError("--decoder-only runs no backend; omit --mode")

    output_dir.mkdir(parents=True, exist_ok=True)
    capacity = max_in_flight if mode == "v2_pipeline" else 1
    config = {
        "mode": mode,
        "decoder_only": decoder_only,
        "sources": source_count,
        "url_template": url_template,
        "duration_seconds": duration_seconds,
        "warmup_seconds": warmup_seconds,
        "max_in_flight_requested": max_in_flight,
        "max_in_flight_effective": capacity,
        "model_id": None if decoder_only else model_id,
        "confidence": None if decoder_only else confidence,
        "device": device,
    }

    exit_code = run(
        config,
        output_dir=output_dir,
        progress_seconds=progress_seconds,
    )

    sys.exit(exit_code)


def run(config: dict, *, output_dir: Path, progress_seconds: float) -> int:
    """Start sources and backend, run the loop, clean up and write the summary.

    Args:
        config: Run configuration from the command line.
        output_dir: Directory for outputs.
        progress_seconds: Interval of progress lines.

    Returns:
        Process exit code: 0 for a full, clean run, 1 otherwise.
    """
    backend = None
    if not config["decoder_only"]:
        # Mode configuration must precede every workflow / streamvision import.
        import backends

        backends.configure_mode(config["mode"], device=config["device"])

    import metrics
    import torch
    from sources import HOST_FALLBACK_STATS_KEYS, JetsonSourceSet, native_bridge_facts

    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is not available; this benchmark needs NVDEC")

    urls = [
        config["url_template"].format(index=index) for index in range(config["sources"])
    ]
    source_set = JetsonSourceSet(urls, device=config["device"])
    summary: dict = {
        "config": config,
        "environment": _environment_facts(torch, device=config["device"]),
        "native_bridge": native_bridge_facts(),
        "timestamp_semantics": metrics.__doc__,
    }
    stop_reason = "not_started"
    loop_result: dict = {}
    stop_report = None
    try:
        if not config["decoder_only"]:
            print(f"loading backend {config['mode']} ...", flush=True)
            backend = backends.build_backend(
                config["mode"],
                config["model_id"],
                config["confidence"],
                device=config["device"],
                max_in_flight=config["max_in_flight_effective"],
            )
            summary["backend"] = backend.facts
            summary["offline_warmup_seconds"] = backend.warm_up()

        print(f"starting {len(urls)} sources ...", flush=True)
        source_set.start()

        admission = _build_admission(
            backend,
            wakeup=source_set.wakeup,
            mode=config["mode"],
            capacity=config["max_in_flight_effective"],
        )
        loop_result = _run_loop(
            source_set,
            admission=admission,
            config=config,
            output_dir=output_dir,
            progress_seconds=progress_seconds,
        )
        stop_reason = loop_result["stop_reason"]
        summary["sources"] = [source.facts() for source in source_set.sources]
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    except Exception as error:  # noqa: BLE001 - recorded, then cleanup runs
        stop_reason = f"error: {error!r}"
    finally:
        # Sources first: their stop is bounded, and native frame buffers held
        # by an unfinished backend call stay valid (the bridge pool outlives
        # the pipeline).
        print(f"stopping ({stop_reason}) ...", flush=True)
        stop_report = source_set.stop(timeout_seconds=STOP_TIMEOUT_SECONDS)
        backend_close_error = None
        if backend is not None:
            try:
                backend.close()
            except Exception as error:  # noqa: BLE001 - recorded in summary
                backend_close_error = repr(error)

    summary["stop_reason"] = stop_reason
    summary["stop"] = stop_report
    summary["backend_close_error"] = backend_close_error
    summary.update(loop_result.get("summary", {}))
    fallback_violations = metrics.host_fallback_violations(
        stop_report["final_native"],
        keys=HOST_FALLBACK_STATS_KEYS,
    )
    summary["host_fallback_violations"] = fallback_violations

    clean = (
        stop_reason == "duration"
        and backend_close_error is None
        and not loop_result.get("summary", {}).get("undrained_in_flight")
        and not loop_result.get("summary", {}).get("failed_completions")
        and not fallback_violations
        and not stop_report["threads_alive"]
        and not stop_report["unfinished_stops"]
    )
    summary["clean_run"] = clean
    metrics.write_json(output_dir / "summary.json", summary)
    _print_final(summary, output_dir=output_dir)

    exit_code = 0 if clean else 1

    return exit_code


def _build_admission(
    backend,
    *,
    wakeup: threading.Condition,
    mode: Optional[str],
    capacity: int,
) -> Admission:
    if backend is None:
        admission = Admission(
            wakeup,
            process=lambda frame, image_id: None,
            capacity=capacity,
        )
        return admission

    if mode != "v2_pipeline":
        admission = Admission(
            wakeup,
            process=backend.process,
            capacity=capacity,
            summarize=backend.summarize,
        )
        return admission

    admission = Admission(
        wakeup,
        submit=backend.submit,
        capacity=capacity,
        summarize=backend.summarize,
    )

    return admission


def _run_loop(
    source_set,
    *,
    admission: Admission,
    config: dict,
    output_dir: Path,
    progress_seconds: float,
) -> dict:
    import metrics

    start_ns = time.monotonic_ns()
    window_start_ns = start_ns + int(config["warmup_seconds"] * 1e9)
    window_end_ns = window_start_ns + int(config["duration_seconds"] * 1e9)
    recorder = metrics.RunRecorder(
        output_dir,
        source_count=len(source_set.sources),
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
    )
    progress = metrics.ProgressPrinter(
        interval_seconds=progress_seconds,
        source_count=len(source_set.sources),
    )
    snapshots = {}
    cpu_seconds = {}
    stop_reason = None
    backend_error = None
    try:
        while stop_reason is None:
            now_ns = time.monotonic_ns()
            if "start" not in snapshots and now_ns >= window_start_ns:
                snapshots["start"] = source_set.snapshot()
                cpu_seconds["start"] = metrics.process_cpu_seconds()
            if now_ns >= window_end_ns:
                stop_reason = "duration"
                break

            arrived = None
            with source_set.wakeup:
                completions = admission.take_completions_locked()
                source_end = source_set.first_end_locked()
                if not completions and source_end is None:
                    if admission.in_flight < admission.capacity:
                        arrived = source_set.take_next_locked()
                    if arrived is None:
                        remaining_s = (window_end_ns - now_ns) / 1e9
                        source_set.wakeup.wait(timeout=min(WAIT_SECONDS, remaining_s))

            backend_error = _record(
                completions, recorder, progress, summarize=admission.summarize
            )
            if backend_error is not None:
                stop_reason = f"backend_error: {backend_error!r}"
            elif source_end is not None:
                stop_reason = f"{source_end.reason}: {source_end.detail}"
            elif arrived is not None:
                admission.admit(arrived)

            progress.maybe_print(
                time.monotonic_ns(),
                phase="warmup" if now_ns < window_start_ns else "measured",
                in_flight=admission.in_flight,
                capacity=admission.capacity,
            )
    except KeyboardInterrupt:
        stop_reason = "interrupted"

    # Window end: stop admitting, read counters, then drain what is in flight.
    end_ns = time.monotonic_ns()
    snapshots["end"] = source_set.snapshot()
    cpu_seconds["end"] = metrics.process_cpu_seconds()
    for source in source_set.sources:
        source.request_stop()
    undrained, drain_error = _drain(admission, recorder, progress)
    if drain_error is not None and backend_error is None:
        stop_reason = f"backend_error: {drain_error!r}"
    recorder.close()

    summary = {
        "phase_counts": recorder.phase_counts,
        "undrained_in_flight": undrained,
        "failed_completions": admission.failed_completions,
        "max_in_flight_per_source": admission.max_per_source,
    }
    if "start" in snapshots:
        measured_seconds = (min(end_ns, window_end_ns) - window_start_ns) / 1e9
        summary["window_seconds"] = measured_seconds
        summary["throughput"] = metrics.summarize_sources(
            start=snapshots["start"],
            end=snapshots["end"],
            measured_counts=recorder.measured_counts(),
            declared_fps=[source.declared_fps() for source in source_set.sources],
            duration_seconds=measured_seconds,
        )
        summary["latency_ms"] = recorder.latency_summary()
        summary["process_cpu"] = {
            "cpu_seconds": cpu_seconds["end"] - cpu_seconds["start"],
            "cores_busy": (cpu_seconds["end"] - cpu_seconds["start"])
            / measured_seconds,
        }
    loop_result = {
        "stop_reason": stop_reason,
        "summary": summary,
    }

    return loop_result


def _record(completions, recorder, progress, *, summarize) -> Optional[BaseException]:
    import metrics

    first_error = None
    for completion in completions:
        if completion.error is not None:
            first_error = first_error or completion.error
            continue
        frame = completion.arrived.frame
        recorder.record(
            metrics.FrameTiming(
                source_index=completion.arrived.source_index,
                frame_id=frame.frame_id,
                reader_wallclock_s=frame.frame_timestamp.timestamp(),
                arrival_ns=completion.arrived.arrival_ns,
                admit_ns=completion.admit_ns,
                ready_ns=completion.ready_ns,
                result=summarize(completion.result),
            )
        )
        progress.completed(completion.arrived.source_index)

    return first_error


def _drain(
    admission: Admission, recorder, progress
) -> tuple[int, Optional[BaseException]]:
    # Frames admitted before the stop finish and are recorded as tail; their
    # native buffers stay referenced until then.
    deadline = time.monotonic() + DRAIN_TIMEOUT_SECONDS
    first_error = None
    while True:
        with admission.wakeup:
            completions = admission.take_completions_locked()
            if not completions and admission.in_flight > 0:
                admission.wakeup.wait(timeout=WAIT_SECONDS)
        error = _record(completions, recorder, progress, summarize=admission.summarize)
        if first_error is None:
            first_error = error
        if admission.in_flight == 0 or time.monotonic() >= deadline:
            break

    undrained = admission.in_flight

    return undrained, first_error


def _environment_facts(torch, *, device: str) -> dict:
    import platform

    facts = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(torch.device(device)),
        "machine": platform.machine(),
        "monotonic_clock": "time.monotonic_ns",
    }

    return facts


def _print_final(summary: dict, *, output_dir: Path) -> None:
    throughput = summary.get("throughput")
    print(f"stop reason: {summary['stop_reason']} | clean run: {summary['clean_run']}")
    if throughput is not None and throughput["totals"]["completed"] > 0:
        totals = throughput["totals"]
        fairness = throughput["fairness"]
        age = summary["latency_ms"]["aggregate"]["age_ms"]
        print(
            f"completed {totals['completed_fps']:.1f} fps over "
            f"{summary['window_seconds']:.1f} s | per source "
            f"{fairness.get('min_fps', 0):.1f}..{fairness.get('max_fps', 0):.1f} "
            f"(jain {fairness['jain_index']:.3f}) | age p50/p95 "
            + (f"{age['p50']:.1f}/{age['p95']:.1f} ms" if age else "n/a")
        )
        print(
            f"drops: runner slot {totals['replaced_in_slot']} | "
            f"VideoSource {totals['video_source_dropped']} | "
            f"native bridge {totals['native_dropped_by_consumer']}"
        )
    if summary["host_fallback_violations"]:
        print(f"HOST FALLBACK DETECTED: {summary['host_fallback_violations']}")
    if summary["stop"]["threads_alive"]:
        print(f"THREADS STILL ALIVE: {summary['stop']['threads_alive']}")
    print(f"wrote {output_dir / 'summary.json'} and {output_dir / 'frames.csv'}")


if __name__ == "__main__":
    main()
