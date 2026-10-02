"""Live physical batches: V1 tensor comparator, V2 serial and V2 pipeline.

```
N RTSP sources ─ NVDEC ─ latest slot per source ─ round-robin admission (≤ capacity)
  └─ collector (09 _Batcher._collect): first frame, then up to B frames or the
     collect timeout, whichever comes first
       v1_tensor, v2_serial   09 _Batcher: one batch at a time       capacity B
       v2_pipeline            _PipelinedBatcher: ≤ D batches running  capacity B x D
  └─ a frame is ready when its batch's rows are ready (all painters included)
```

Sources, admission, ``_run_loop``, drain, metrics, ``summary.json`` and
``frames.csv`` are ``09 run_benchmark``'s; collection, warmup, forward-size
recording, ``batches.csv`` and the batching summary are ``09 run_batched``'s.
Every mode therefore shares the collect-timeout semantics of the 09 V1 batch
runs. The work per batch comes from ``batched_backend``. The pipelined
collector waits for a free batch slot before it collects, so frames that arrive
meanwhile join the next batch instead of waiting in a frozen one.

    python run_batched.py --mode v2_pipeline --batch-size 8 --pipeline-depth 2 \\
        --sources 16 --output-dir /tmp/thor/batched/v2_pipeline_b8_d2_s16
"""

import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, List

import click
import thor_imports  # installs the 08 and 09 search paths

# isort: split

import batched_backend
from run_benchmark import (
    DEFAULT_MODEL_ID,
    DEFAULT_URL_TEMPLATE,
    STOP_TIMEOUT_SECONDS,
    Admission,
    _environment_facts,
    _print_final,
    _run_loop,
)

batched_09 = thor_imports.load_09_module("run_batched")
_Batcher = batched_09._Batcher


class _PipelinedBatcher(_Batcher):
    """09's ``_Batcher`` with up to ``depth`` batches executing at once.

    The worker takes a free batch slot, collects exactly as 09 does, and
    submits the batch without waiting for it. The slot returns when the batch
    is ready; then its frames' futures complete. ``batches`` rows are
    ``(size, submitted_ns, ready_ns)``; they overlap in time.

    Args:
        submit_batch: ``(frames, image_ids) -> Future`` of rows in input order.
        batch_size: Maximum frames per batch.
        depth: Maximum batches executing at once.
        collect_timeout_s: Longest wait for a partial batch to fill.
    """

    def __init__(
        self,
        submit_batch: Callable[[List[Any], List[str]], Future],
        *,
        batch_size: int,
        depth: int,
        collect_timeout_s: float,
    ):
        # Set before the base class starts the worker, which uses them.
        self.depth = depth
        self._slots = threading.BoundedSemaphore(depth)
        super().__init__(
            submit_batch, batch_size=batch_size, collect_timeout_s=collect_timeout_s
        )

    def submit(self, frame: Any, *, image_id: str) -> Future:
        """Queue one frame; up to ``batch_size x depth`` may be pending."""
        future: Future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("batcher is closed")
            if len(self._pending) >= self.batch_size * self.depth:
                raise RuntimeError(f"more than {self.batch_size * self.depth} pending")
            self._pending.append((frame, image_id, future))
            self._condition.notify_all()

        return future

    def _collect(self) -> list:
        self._slots.acquire()
        batch = super()._collect()
        if not batch:
            self._slots.release()

        return batch

    def _execute(self, batch: list) -> None:
        frames, image_ids, futures = (list(items) for items in zip(*batch))
        submitted_ns = time.monotonic_ns()
        try:
            rows = self._run_batch(frames, image_ids)
        except BaseException as error:  # noqa: BLE001 - every member fails
            rows = Future()
            rows.set_exception(error)

        rows.add_done_callback(
            lambda done: self._finish(done, futures=futures, submitted_ns=submitted_ns)
        )

    def _finish(
        self, rows: Future, *, futures: List[Future], submitted_ns: int
    ) -> None:
        # Runs where the batch completed: a pipeline worker, or here on failure.
        ready_ns = time.monotonic_ns()
        self._slots.release()
        try:
            results = rows.result()
            if len(results) != len(futures):
                raise RuntimeError(f"{len(results)} results for {len(futures)} frames")
        except BaseException as failure:  # noqa: BLE001 - every member fails
            self.failed_batches += 1
            for future in futures:
                future.set_exception(failure)
            return

        self.batches.append((len(futures), submitted_ns, ready_ns))
        for future, result in zip(futures, results):
            future.set_result(result)


@click.command()
@click.option(
    "--mode",
    type=click.Choice(batched_backend.MODES),
    required=True,
    help="v1_tensor: 09 V1 comparator; v2_serial / v2_pipeline: batched V2.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(
        min=1,
    ),
    default=8,
    show_default=True,
    help="Most frames per batch.",
)
@click.option(
    "--pipeline-depth",
    type=click.IntRange(
        min=1,
    ),
    default=2,
    show_default=True,
    help="v2_pipeline: most batches executing at once. Other modes use 1.",
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
    help="Longest wait for a partial batch to fill (09 default).",
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
    pipeline_depth: int,
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
    """Run N NVDEC sources through one batched backend and write results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    depth = pipeline_depth if mode == "v2_pipeline" else 1
    config = {
        "runner": "10-thor-batching/run_batched.py",
        "mode": mode,
        "decoder_only": False,
        "sources": source_count,
        "url_template": url_template,
        "duration_seconds": duration_seconds,
        "warmup_seconds": warmup_seconds,
        "batch_size": batch_size,
        "pipeline_depth_requested": pipeline_depth,
        "pipeline_depth_effective": depth,
        # 09 summary keys, in FRAMES: the Admission capacity, batch_size x depth.
        "max_in_flight_requested": batch_size * pipeline_depth,
        "max_in_flight_effective": batch_size * depth,
        "max_in_flight_unit": "frames",
        "collect_timeout_ms": collect_timeout_ms,
        "record_forward_batches": record_forward_batches,
        "model_id": model_id,
        "confidence": confidence,
        "device": device,
    }

    exit_code = run(config, output_dir=output_dir, progress_seconds=progress_seconds)

    sys.exit(exit_code)


def run(config: dict, *, output_dir: Path, progress_seconds: float) -> int:
    """09 ``run_batched.run`` with a batched backend of any mode; same outputs.

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
    backend = batcher = forward_sizes = None
    stop_reason = "not_started"
    loop_result: dict = {}
    loop_started_ns = None
    try:
        print(f"loading backend {config['mode']} ...", flush=True)
        model = backends.load_trt_model(config["model_id"], device=config["device"])
        backend = batched_backend.build_for_model(
            config["mode"],
            model=model,
            model_id=config["model_id"],
            confidence=config["confidence"],
            pipeline_depth=config["pipeline_depth_effective"],
        )
        batcher = _build_batcher(backend, config=config)
        summary["backend"] = {**backend.facts, "allow_untrusted_packages": True}
        summary["offline_warmup"] = batched_09._warm_up(batcher, device=backend.device)
        if config["record_forward_batches"]:
            forward_sizes = batched_09._observe_forward_batches(model)

        print(f"starting {len(urls)} sources ...", flush=True)
        source_set.start()
        admission = Admission(
            source_set.wakeup,
            submit=batcher.submit,
            capacity=config["max_in_flight_effective"],
            summarize=backend.summarize,
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
        backend_close_error = _close(backend, batcher)

    summary["stop_reason"] = stop_reason
    summary["stop"] = stop_report
    summary["backend_close_error"] = backend_close_error
    summary.update(loop_result.get("summary", {}))
    if batcher is not None:
        summary["batching"] = {
            **batched_09._batch_summary(
                batcher, config=config, loop_started_ns=loop_started_ns
            ),
            "batch_service_semantics": _service_semantics(config["mode"]),
            "forward_batch_sizes_live": (
                dict(sorted(forward_sizes.items()))
                if forward_sizes is not None
                else None
            ),
        }
        batched_09._write_batches(batcher, path=output_dir / "batches.csv")
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


def _build_batcher(backend: Any, *, config: dict) -> _Batcher:
    collect_timeout_s = config["collect_timeout_ms"] / 1000.0
    if config["mode"] == "v2_pipeline":
        batcher = _PipelinedBatcher(
            lambda frames, ids: backend.submit_batch(frames, image_ids=ids),
            batch_size=config["batch_size"],
            depth=config["pipeline_depth_effective"],
            collect_timeout_s=collect_timeout_s,
        )
        return batcher

    batcher = _Batcher(
        lambda frames, ids: backend.process_batch(frames, image_ids=ids),
        batch_size=config["batch_size"],
        collect_timeout_s=collect_timeout_s,
    )

    return batcher


def _close(backend: Any, batcher: Any) -> Any:
    # Worker first (it runs what is still pending), then the V2 pipeline.
    if batcher is not None and not batcher.close(timeout_s=STOP_TIMEOUT_SECONDS):
        return "batch worker still running after close"
    if backend is None:
        return None

    try:
        backend.close()
    except Exception as error:  # noqa: BLE001 - recorded in summary
        return repr(error)

    return None


def _service_semantics(mode: str) -> str:
    if mode == "v2_pipeline":
        semantics = (
            "engine_run_ms: submit_batch to rows ready, per batch. Up to "
            "pipeline_depth batches run at once, so durations overlap; do not sum."
        )
        return semantics

    semantics = (
        "engine_run_ms: one batch on the worker thread: image wrapping, "
        "engine.run / session.run of the whole batch and the readiness wait."
    )

    return semantics


if __name__ == "__main__":
    main()
