"""Live V1 with physical batching: a practical V1 baseline next to batch 1.

```
N RTSP sources ─ NVDEC ─ latest slot per source ─ round-robin admission
  └─ Admission.submit (≤ B in flight) ─ _Batcher: up to B frames, ≤ collect timeout
       └─ engine.run(image=[B WorkflowImageData]) ─ CUDA event wait ─ B futures done
```

Sources, admission, ``_run_loop``, metrics and outputs (``summary.json``,
``frames.csv``) are those of ``run_benchmark.py``. The model and V1 engine
come from ``backends`` (stock blocks, strict TRT). Only the call differs: one
``engine.run`` receives a list of up to ``--batch-size`` images. At most B
frames are admitted at once, so a second batch is never queued behind a
running one. A frame is ready when the event recorded after its batch's
``engine.run`` (all annotations included) has completed; v1_numpy converts
each frame to host BGR inside the batch, so the conversion is timed service.

Adaptation, labelled: ``run`` below is a short copy of ``run_benchmark.run``,
because that one builds the batch-1 admission itself. The frozen runner is
imported, not changed. Extra output: ``batches.csv``, one row per engine call.

    python run_batched.py --mode v1_tensor --batch-size 8 --sources 16 \\
        --output-dir /tmp/thor/batched/v1_tensor_b8_s16
"""

import csv
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Deque, List, Optional, Tuple

import click
from run_benchmark import (
    DEFAULT_MODEL_ID,
    DEFAULT_URL_TEMPLATE,
    STOP_TIMEOUT_SECONDS,
    Admission,
    _environment_facts,
    _print_final,
    _run_loop,
)

BATCH_SIZE_KEY = "batch_size"
WARMUP_ROUNDS = 3
WARMUP_SIZE_HW = (1080, 1920)


class _Batcher:
    """One worker thread that runs submitted frames in batches of at most B.

    The worker takes the first pending frame, waits up to the collect timeout
    for more (it stops waiting at B), and runs them as one batch. Each future
    is completed only after ``run_batch`` returned, i.e. after the batch is
    ready; a failure fails every frame of that batch.

    Args:
        run_batch: ``(frames, image_ids) -> results`` in input order.
        batch_size: Maximum frames per batch, and maximum frames pending.
        collect_timeout_s: Longest wait for a partial batch to fill.
    """

    def __init__(
        self,
        run_batch: Callable[[List[Any], List[str]], List[Any]],
        *,
        batch_size: int,
        collect_timeout_s: float,
    ):
        self.batch_size = batch_size
        self.batches: List[Tuple[int, int, int]] = []  # size, started_ns, ready_ns
        self.failed_batches = 0
        self._run_batch = run_batch
        self._collect_timeout_s = collect_timeout_s
        self._condition = threading.Condition()
        self._pending: Deque[Tuple[Any, str, Future]] = deque()
        self._closed = False
        self._thread = threading.Thread(target=self._work, name="v1-batch", daemon=True)
        self._thread.start()

    def submit(self, frame: Any, *, image_id: str) -> Future:
        """Queue one frame; the future is done when its batch is ready."""
        future: Future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("batcher is closed")
            if len(self._pending) >= self.batch_size:
                raise RuntimeError(f"more than {self.batch_size} frames pending")
            self._pending.append((frame, image_id, future))
            self._condition.notify_all()

        return future

    def close(self, *, timeout_s: float) -> bool:
        """Run what is still pending, stop the worker; True if it stopped."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout_s)
        stopped = not self._thread.is_alive()

        return stopped

    def _work(self) -> None:
        while True:
            batch = self._collect()
            if not batch:
                return

            self._execute(batch)

    def _collect(self) -> List[Tuple[Any, str, Future]]:
        with self._condition:
            while not self._pending and not self._closed:
                self._condition.wait()
            deadline = time.monotonic() + self._collect_timeout_s
            while 0 < len(self._pending) < self.batch_size and not self._closed:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    break
                self._condition.wait(timeout=remaining_s)
            count = min(self.batch_size, len(self._pending))
            batch = [self._pending.popleft() for _ in range(count)]

        return batch

    def _execute(self, batch: List[Tuple[Any, str, Future]]) -> None:
        frames, image_ids, futures = (list(items) for items in zip(*batch))
        started_ns = time.monotonic_ns()
        try:
            results = self._run_batch(frames, image_ids)
            if len(results) != len(futures):
                raise RuntimeError(f"{len(results)} results for {len(futures)} frames")
        except BaseException as error:  # noqa: BLE001 - every member fails
            self.failed_batches += 1
            for future in futures:
                future.set_exception(error)
            return

        self.batches.append((len(batch), started_ns, time.monotonic_ns()))
        for future, result in zip(futures, results):
            future.set_result(result)


def v1_batch_runner(backend: Any) -> Callable[[List[Any], List[str]], List[dict]]:
    """``run_batch`` for a ``backends.V1Backend``: one engine.run per batch.

    Mirrors ``V1Backend.process`` for a list of images (explicit experimental
    reuse of its ``_engine``, ``_confidence`` and ``backends._host_bgr``).

    Args:
        backend: The v1_numpy or v1_tensor backend.

    Returns:
        A function that runs frames as one batch and returns ready results,
        each with ``host_conversion_ms`` and ``batch_size`` added.
    """
    import backends
    import torch
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    def run_batch(frames: List[Any], image_ids: List[str]) -> List[dict]:
        images = []
        conversions_ms: List[Optional[float]] = []
        for frame, image_id in zip(frames, image_ids):
            parent = ImageParentMetadata(parent_id=image_id)
            if backend.mode == "v1_numpy":
                started = time.perf_counter()
                pixels = backends._host_bgr(frame)
                conversions_ms.append((time.perf_counter() - started) * 1000.0)
                images.append(
                    WorkflowImageData(parent_metadata=parent, numpy_image=pixels)
                )
            else:
                conversions_ms.append(None)
                images.append(
                    WorkflowImageData(parent_metadata=parent, tensor_image=frame)
                )

        results = backend._engine.run(
            runtime_parameters={"image": images, "confidence": backend._confidence}
        )
        if backend.mode == "v1_tensor" and backend.device.type == "cuda":
            # V1 painters queue on this thread's current stream; wait for them.
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(backend.device))
            done.synchronize()
        for result, conversion_ms in zip(results, conversions_ms):
            result[backends.HOST_CONVERSION_MS] = conversion_ms
            result[BATCH_SIZE_KEY] = len(images)

        return results

    return run_batch


@click.command()
@click.option(
    "--mode",
    type=click.Choice(
        ["v1_numpy", "v1_tensor"],
    ),
    required=True,
    help="Stock V1 engine representation.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(
        min=1,
    ),
    default=8,
    show_default=True,
    help="Most frames per engine.run; also the most frames admitted at once.",
)
@click.option(
    "--sources",
    "source_count",
    type=click.IntRange(
        min=1,
    ),
    default=16,
    show_default=True,
)
@click.option(
    "--url-template",
    default=DEFAULT_URL_TEMPLATE,
    show_default=True,
)
@click.option(
    "--duration-seconds",
    type=click.FloatRange(
        min=1.0,
    ),
    default=30.0,
    show_default=True,
)
@click.option(
    "--warmup-seconds",
    type=click.FloatRange(
        min=0.0,
    ),
    default=10.0,
    show_default=True,
)
@click.option(
    "--collect-timeout-ms",
    type=click.FloatRange(
        min=0.0,
        max=50.0,
    ),
    default=3.0,
    show_default=True,
    help="Longest wait for a partial batch to fill.",
)
@click.option(
    "--output-dir",
    type=click.Path(
        file_okay=False,
        path_type=Path,
    ),
    required=True,
)
@click.option(
    "--record-forward-batches",
    is_flag=True,
    default=False,
    help="Diagnostic: count batch sizes reaching TRT forward (wraps it).",
)
@click.option(
    "--model-id",
    default=DEFAULT_MODEL_ID,
    show_default=True,
)
@click.option(
    "--confidence",
    type=click.FloatRange(
        min=0.0,
        max=1.0,
    ),
    default=0.4,
    show_default=True,
)
@click.option(
    "--device",
    default="cuda:0",
    show_default=True,
)
@click.option(
    "--progress-seconds",
    type=click.FloatRange(
        min=0.5,
    ),
    default=5.0,
    show_default=True,
)
def main(
    mode: str,
    batch_size: int,
    source_count: int,
    url_template: str,
    duration_seconds: float,
    warmup_seconds: float,
    collect_timeout_ms: float,
    output_dir: Path,
    record_forward_batches: bool,
    model_id: str,
    confidence: float,
    device: str,
    progress_seconds: float,
) -> None:
    """Run N NVDEC sources through the stock V1 engine in physical batches."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "runner": "run_batched.py",
        "mode": mode,
        "decoder_only": False,
        "sources": source_count,
        "url_template": url_template,
        "duration_seconds": duration_seconds,
        "warmup_seconds": warmup_seconds,
        "max_in_flight_requested": batch_size,
        "max_in_flight_effective": batch_size,
        "batch_size": batch_size,
        "collect_timeout_ms": collect_timeout_ms,
        "record_forward_batches": record_forward_batches,
        "model_id": model_id,
        "confidence": confidence,
        "device": device,
    }

    exit_code = run(config, output_dir=output_dir, progress_seconds=progress_seconds)

    sys.exit(exit_code)


def run(config: dict, *, output_dir: Path, progress_seconds: float) -> int:
    """``run_benchmark.run`` with a batching admission; same summary keys.

    Args:
        config: Run configuration from the command line.
        output_dir: Directory for outputs.
        progress_seconds: Interval of progress lines.

    Returns:
        Process exit code: 0 for a full, clean run, 1 otherwise.
    """
    import backends

    backends.configure_mode(config["mode"], device=config["device"])

    import metrics
    import torch
    from sources import HOST_FALLBACK_STATS_KEYS, JetsonSourceSet, native_bridge_facts

    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is not available; this benchmark needs NVDEC")

    urls = [config["url_template"].format(index=i) for i in range(config["sources"])]
    source_set = JetsonSourceSet(urls, device=config["device"])
    summary: dict = {
        "config": config,
        "environment": _environment_facts(torch, device=config["device"]),
        "native_bridge": native_bridge_facts(),
        "timestamp_semantics": metrics.__doc__,
    }
    batcher = None
    forward_sizes = None
    stop_reason = "not_started"
    loop_result: dict = {}
    loop_started_ns = None
    try:
        print(f"loading backend {config['mode']} ...", flush=True)
        # build_backend's two steps, kept apart so the model can be observed.
        model = backends.load_trt_model(config["model_id"], device=config["device"])
        backend = backends._build_for_model(
            config["mode"],
            model=model,
            model_id=config["model_id"],
            confidence=config["confidence"],
            max_in_flight=1,
        )
        batcher = _Batcher(
            v1_batch_runner(backend),
            batch_size=config["batch_size"],
            collect_timeout_s=config["collect_timeout_ms"] / 1000.0,
        )
        summary["backend"] = {
            **backend.facts,
            "allow_untrusted_packages": True,
            "batching": (
                "one engine.run per batch of up to batch_size WorkflowImageData "
                "on one worker thread; ready = event after engine.run of the batch. "
                "Per-frame service includes batch collection wait and batch execution."
            ),
        }
        summary["offline_warmup"] = _warm_up(batcher, device=backend.device)
        if config["record_forward_batches"]:
            forward_sizes = _observe_forward_batches(model)

        print(f"starting {len(urls)} sources ...", flush=True)
        source_set.start()
        admission = Admission(
            source_set.wakeup,
            submit=batcher.submit,
            capacity=config["batch_size"],
            summarize=lambda result: {
                **backend.summarize(result),
                BATCH_SIZE_KEY: result[BATCH_SIZE_KEY],
            },
        )
        loop_started_ns = time.monotonic_ns()
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
        print(f"stopping ({stop_reason}) ...", flush=True)
        stop_report = source_set.stop(timeout_seconds=STOP_TIMEOUT_SECONDS)
        backend_close_error = None
        if batcher is not None and not batcher.close(timeout_s=STOP_TIMEOUT_SECONDS):
            backend_close_error = "batch worker still running after close"

    summary["stop_reason"] = stop_reason
    summary["stop"] = stop_report
    summary["backend_close_error"] = backend_close_error
    summary.update(loop_result.get("summary", {}))
    if batcher is not None:
        summary["batching"] = _batch_summary(
            batcher, config=config, loop_started_ns=loop_started_ns
        )
        summary["batching"]["forward_batch_sizes_live"] = (
            dict(sorted(forward_sizes.items())) if forward_sizes is not None else None
        )
        _write_batches(batcher, path=output_dir / "batches.csv")
    fallback_violations = metrics.host_fallback_violations(
        stop_report["final_native"], keys=HOST_FALLBACK_STATS_KEYS
    )
    summary["host_fallback_violations"] = fallback_violations

    clean = (
        stop_reason == "duration"
        and backend_close_error is None
        and not summary.get("undrained_in_flight")
        and not summary.get("failed_completions")
        and not fallback_violations
        and not stop_report["threads_alive"]
        and not stop_report["unfinished_stops"]
    )
    summary["clean_run"] = clean
    metrics.write_json(output_dir / "summary.json", summary)
    _print_final(summary, output_dir=output_dir)
    print(f"batching summary: {summary.get('batching', {})}")

    exit_code = 0 if clean else 1

    return exit_code


def _warm_up(batcher: _Batcher, *, device: Any) -> dict:
    # Through the worker thread, so the model's per-thread streams warm up
    # where live batches run: every batch size once, then full batches.
    import torch

    blank = torch.zeros((3, *WARMUP_SIZE_HW), dtype=torch.uint8, device=device)
    torch.cuda.current_stream(device).synchronize()
    started = time.perf_counter()
    sizes = [*range(1, batcher.batch_size + 1), *[batcher.batch_size] * WARMUP_ROUNDS]
    for round_index, size in enumerate(sizes):
        futures = [
            batcher.submit(blank, image_id=f"warmup-{round_index}-{member}")
            for member in range(size)
        ]
        for future in futures:
            future.result()
    warmup = {
        "seconds": time.perf_counter() - started,
        "batch_sizes": [size for size, _, _ in batcher.batches],
    }
    batcher.batches.clear()

    return warmup


def _observe_forward_batches(model: Any) -> Counter:
    # Diagnostic instance-attribute wrap, as in batch_diagnostic.py.
    sizes: Counter = Counter()
    original = model.forward

    def observed(pre_processed: Any, *args: Any, **kwargs: Any) -> Any:
        sizes[int(pre_processed.shape[0])] += 1
        return original(pre_processed, *args, **kwargs)

    model.forward = observed

    return sizes


def _batch_summary(batcher: _Batcher, *, config: dict, loop_started_ns) -> dict:
    import metrics

    batches = list(batcher.batches)
    in_window = []
    if loop_started_ns is not None:
        # _run_loop starts its window clock a few microseconds after this stamp.
        start_ns = loop_started_ns + int(config["warmup_seconds"] * 1e9)
        end_ns = start_ns + int(config["duration_seconds"] * 1e9)
        in_window = [b for b in batches if b[1] >= start_ns and b[2] <= end_ns]
    service_ms_by_size = {}
    for size in sorted({b[0] for b in in_window}):
        durations = [(b[2] - b[1]) / 1e6 for b in in_window if b[0] == size]
        service_ms_by_size[size] = metrics._distribution(durations)
    summary = {
        "batch_service_semantics": (
            "engine_run_ms measures the full batch callback: image wrapping, "
            "host conversion when used, engine.run, and GPU readiness wait."
        ),
        "batches_all": len(batches),
        "frames_all": sum(b[0] for b in batches),
        "failed_batches": batcher.failed_batches,
        "batch_size_histogram_all": dict(
            sorted(Counter(b[0] for b in batches).items())
        ),
        "batch_size_histogram_window": dict(
            sorted(Counter(b[0] for b in in_window).items())
        ),
        "engine_run_ms_by_batch_size_window": service_ms_by_size,
    }

    return summary


def _write_batches(batcher: _Batcher, *, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["batch_size", "started_ns", "ready_ns", "engine_run_ms"])
        for size, started_ns, ready_ns in batcher.batches:
            writer.writerow([size, started_ns, ready_ns, (ready_ns - started_ns) / 1e6])


if __name__ == "__main__":
    main()
