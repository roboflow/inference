"""How much does one physical batch of N frames help the stock V1 tensor workflow?

Diagnostic only. The benchmark (``run_benchmark.py``) runs every mode at model
batch 1; this script does not feed it. It measures workload capacity on held
frames: no decoding, no sources, no admission.

```
fixture.pt (check_parity.py) -> N GPU frames, held
  batch N:  engine.run(image=[N WorkflowImageData]) -> one TRT forward of N
  batch 1:  N x engine.run(image=WorkflowImageData)  -> N TRT forwards of 1
  each iteration ends when an event recorded after the last run has completed
```

Same process, same loaded model, same V1 engine, stock blocks and confidence
as ``v1_tensor`` in ``backends.py``. Before timing, one call per variant
records the batch sizes the TRT ``forward`` actually receives, and the
batch-N outputs are compared with the batch-1 outputs frame by frame.
``--model-only`` adds the bare model call (pre-process, forward, NMS) at both
batch sizes, to separate model from workflow overhead.

    python batch_diagnostic.py --fixture /tmp/thor/parity/fixture.pt --output /tmp/thor/batch8.json
"""

import json
import statistics
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List

import click
from run_benchmark import DEFAULT_MODEL_ID

MODE = "v1_tensor"


@click.command()
@click.option(
    "--fixture",
    type=click.Path(
        exists=True,
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="fixture.pt written by check_parity.py.",
)
@click.option(
    "--output",
    type=click.Path(
        dir_okay=False,
        path_type=Path,
    ),
    required=True,
    help="JSON report path.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(
        min=2,
    ),
    default=8,
    show_default=True,
    help="Frames per physical batch; fixture frames repeat when it has fewer.",
)
@click.option(
    "--warmup",
    type=click.IntRange(
        min=1,
    ),
    default=10,
    show_default=True,
    help="Untimed iterations per variant.",
)
@click.option(
    "--iterations",
    type=click.IntRange(
        min=1,
    ),
    default=50,
    show_default=True,
    help="Timed iterations per variant; one iteration processes batch-size frames.",
)
@click.option(
    "--model-only/--no-model-only",
    default=False,
    show_default=True,
    help="Also time the bare model call at batch 1 and batch N.",
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
def main(
    fixture: Path,
    output: Path,
    batch_size: int,
    warmup: int,
    iterations: int,
    model_only: bool,
    model_id: str,
    confidence: float,
    device: str,
) -> None:
    """Time V1 tensor workflow batch N against N runs of batch 1 on held frames."""
    import backends

    backends.configure_mode(MODE, device=device)

    import torch
    from check_parity import load_fixture
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    stored = load_fixture(fixture)["frames"]
    frames = [stored[index % len(stored)].to(device) for index in range(batch_size)]
    # The model and painters read frames on their own streams; the uploads must
    # be complete first. Waits on the upload stream only, not the device.
    torch.cuda.current_stream(device).synchronize()
    model = backends.load_trt_model(model_id, device=device)
    # Experimental reuse of the backend internals: the v1_tensor backend
    # around a model this script can also call and observe directly.
    backend = backends._build_for_model(
        MODE, model=model, model_id=model_id, confidence=confidence, max_in_flight=1
    )
    engine = backend._engine
    stream = torch.cuda.current_stream(backend.device)

    def images(count: int) -> List[Any]:
        wrapped = [
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id=f"frame-{index}"),
                tensor_image=frame,
            )
            for index, frame in enumerate(frames[:count])
        ]
        return wrapped

    def run_workflow(image: Any) -> List[dict]:
        results = engine.run(
            runtime_parameters={"image": image, "confidence": confidence}
        )
        return results

    def workflow_batch() -> List[dict]:
        results = run_workflow(images(batch_size))
        return results

    def workflow_singles() -> List[dict]:
        results = [run_workflow(image)[0] for image in images(batch_size)]
        return results

    model_kwargs = {
        "confidence": confidence,
        "input_color_format": "rgb",
        **backends.post_process_parameters(),
    }

    def model_batch() -> List[Any]:
        detections = model(frames, **model_kwargs)
        return detections

    def model_singles() -> List[Any]:
        detections = [model([frame], **model_kwargs)[0] for frame in frames]
        return detections

    variants = {
        "workflow_batch1": workflow_singles,
        f"workflow_batch{batch_size}": workflow_batch,
    }
    if model_only:
        variants["model_batch1"] = model_singles
        variants[f"model_batch{batch_size}"] = model_batch

    forward_batches = {
        name: record_forward_batches(model, call=call, stream=stream)
        for name, call in variants.items()
    }
    parity = compare_batched(workflow_singles(), workflow_batch(), backend=backend)
    timings = {
        name: measure(
            call, stream=stream, warmup=warmup, iterations=iterations, frames=batch_size
        )
        for name, call in variants.items()
    }
    report = {
        "scope": (
            "Workload capacity on held GPU frames: V1 engine + stock tensor blocks "
            "(detector -> boxes -> labels). Excludes decoding, sources and "
            "admission. Diagnostic; the primary benchmark runs model batch 1."
        ),
        "readiness": (
            "an iteration ends when a CUDA event recorded on the current stream "
            "after its last call has completed; no device-wide synchronize"
        ),
        "iteration": f"{batch_size} frames: one batch-{batch_size} call or {batch_size} batch-1 calls",
        "fixture": str(fixture),
        "fixture_frames": len(stored),
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "confidence": confidence,
        "backend": backend.facts,
        "forward_batch_sizes_per_iteration": forward_batches,
        "batch_vs_single_parity": parity,
        "timings": timings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str))
    for name, timing in timings.items():
        click.echo(
            f"{name}: {timing['frames_per_second']:.1f} frames/s, "
            f"iteration p50 {timing['iteration_ms_p50']:.2f} ms, "
            f"forward batches {forward_batches[name]}"
        )
    click.echo(f"batch vs single: {parity['status']}; wrote {output}")


def record_forward_batches(model: Any, *, call: Callable, stream: Any) -> List[int]:
    """Batch sizes the TRT ``forward`` receives during one ``call``."""
    sizes: List[int] = []
    original = model.forward

    def observed(pre_processed: Any, *args: Any, **kwargs: Any) -> Any:
        sizes.append(int(pre_processed.shape[0]))
        return original(pre_processed, *args, **kwargs)

    model.forward = observed
    try:
        call()
        _wait(stream)
    finally:
        del model.forward

    return sizes


def measure(
    call: Callable, *, stream: Any, warmup: int, iterations: int, frames: int
) -> Dict[str, float]:
    """Warm up, then time ``iterations`` ready iterations of ``call``."""
    for _ in range(warmup):
        call()
        _wait(stream)

    durations_ms = []
    started = perf_counter()
    for _ in range(iterations):
        iteration_started = perf_counter()
        call()
        _wait(stream)
        durations_ms.append((perf_counter() - iteration_started) * 1000.0)
    elapsed = perf_counter() - started

    ordered = sorted(durations_ms)
    timing = {
        "frames_per_second": frames * iterations / elapsed,
        "iteration_ms_mean": statistics.fmean(durations_ms),
        "iteration_ms_p50": ordered[len(ordered) // 2],
        "iteration_ms_p90": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "iteration_ms_max": ordered[-1],
        "frame_ms_mean": statistics.fmean(durations_ms) / frames,
    }

    return timing


def compare_batched(
    singles: List[dict], batched: List[dict], *, backend: Any
) -> Dict[str, Any]:
    """Per frame: are batch-N predictions and images equal to batch-1 ones?

    TensorRT may pick other kernels per batch size, so small numeric
    differences are possible; they are reported, not hidden.
    """
    frames = []
    for single, batch in zip(singles, batched):
        expected = backend.predictions(single)
        actual = backend.predictions(batch)
        same_count = len(expected) == len(actual)
        frame = {
            "count_batch1": len(expected),
            "count_batchN": len(actual),
            "classes_equal": same_count
            and bool((expected.class_id == actual.class_id).all()),
            "max_box_diff_px": (
                float((expected.xyxy - actual.xyxy).abs().max())
                if same_count and len(expected)
                else None
            ),
            "max_confidence_diff": (
                float((expected.confidence - actual.confidence).abs().max())
                if same_count and len(expected)
                else None
            ),
            "images_identical": bool(
                (backend.annotated(single) == backend.annotated(batch)).all()
            ),
        }
        frames.append(frame)

    identical = all(
        frame["classes_equal"]
        and frame["images_identical"]
        and not frame["max_box_diff_px"]
        and not frame["max_confidence_diff"]
        for frame in frames
    )
    parity = {"status": "identical" if identical else "differs", "frames": frames}

    return parity


def _wait(stream: Any) -> None:
    import torch

    done = torch.cuda.Event()
    done.record(stream)
    done.synchronize()


if __name__ == "__main__":
    main()
