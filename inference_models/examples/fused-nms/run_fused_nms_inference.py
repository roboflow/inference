import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import click
import cv2
import supervision as sv

from inference_models import AutoModel


def _detections_to_json_dict(detections: sv.Detections) -> dict[str, Any]:
    return {
        "xyxy": detections.xyxy.tolist(),
        "mask": detections.mask.tolist() if detections.mask is not None else None,
        "confidence": (
            detections.confidence.tolist()
            if detections.confidence is not None
            else None
        ),
        "class_id": (
            detections.class_id.tolist() if detections.class_id is not None else None
        ),
    }


def _latency_report_dict(
    *,
    image_path: Path,
    model_path: Path,
    warmup_runs: int,
    latencies_ms: list[float],
) -> dict[str, Any]:
    n = len(latencies_ms)
    return {
        "image_path": str(image_path.resolve()),
        "model_path": str(model_path.resolve()),
        "warmup_runs": warmup_runs,
        "timed_runs": n,
        "unit": "ms",
        "latencies_ms": latencies_ms,
        "mean_ms": statistics.mean(latencies_ms) if n else None,
        "median_ms": statistics.median(latencies_ms) if n else None,
        "min_ms": min(latencies_ms) if n else None,
        "max_ms": max(latencies_ms) if n else None,
        "std_ms": statistics.stdev(latencies_ms) if n > 1 else None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


@click.command()
@click.option(
    "--run-name",
    type=str,
    required=True,
    help="Name of the run for reporting. Will be used as a subdirectory in the target directory.",
)
@click.option(
    "--image-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the input image.",
)
@click.option(
    "--model-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=True, readable=True),
    required=True,
    help="Path to the model directory.",
)
@click.option(
    "--target-dir",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Directory for latency.json and prediction.json (created if missing).",
)
@click.option(
    "--confidence",
    type=float,
    help="Confidence threshold used by post-processing.",
)
@click.option(
    "--iou-threshold",
    type=float,
    help="IOU threshold used by post-processing.",
)
@click.option(
    "--max-detections",
    type=int,
    help="Maximum number of detections used by post-processing.",
)
@click.option(
    "-n",
    "--benchmark-iters",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help=(
        "Number of timed inference runs for benchmarking (mean/median/std in ms). "
        "0 runs inference once without benchmark stats."
    ),
)
@click.option(
    "--warmup",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help="Untimed warmup runs before timed iterations.",
)
def main(
    run_name: str,
    image_path: Path,
    model_path: Path,
    target_dir: Path,
    confidence: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    benchmark_iters: int = 0,
    warmup: int = 5,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise click.ClickException(f"Could not load image from: {image_path}")

    nms_params = {
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
    }

    nms_params = {name: value for name, value in nms_params.items() if value is not None}
    if nms_params:
        click.echo(f"User provided NMS parameters: {nms_params}")

    click.echo(f"Loading model: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        onnx_execution_providers=["CPUExecutionProvider"],
        device="cpu",
    )

    click.echo(f"Fused NMS available: {model._inference_config.post_processing.fused}")

    forward_pass = model._inference_config.forward_pass
    if forward_pass.static_batch_size is None:
        max_dyn = forward_pass.max_dynamic_batch_size
        if max_dyn is not None:
            click.echo(
                "Batching: dynamic mode (no static batch size); "
                f"maximum batch size is {max_dyn}."
            )
        else:
            click.echo(
                "Batching: dynamic mode (no static batch size); "
                "max_dynamic_batch_size is not set in the model config."
            )

    latencies_ms: list[float] = []

    if warmup > 0:
        click.echo(f"Warmup: {warmup} untimed runs...")
        for _ in range(warmup):
            model(image, **nms_params)

    click.echo(f"Benchmarking: {benchmark_iters} timed runs...")

    predictions = None
    for _ in range(benchmark_iters):
        t0 = time.perf_counter()
        predictions = model(image, **nms_params)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    assert predictions is not None

    click.echo("Writing reports ...")

    detections = predictions[0].to_supervision()
    pred_payload: dict[str, Any] = {
        "image_path": str(image_path.resolve()),
        "detections": _detections_to_json_dict(detections),
    }

    target_dir.mkdir(parents=True, exist_ok=True)
    latency_path = target_dir / run_name / "latency.json"
    prediction_path = target_dir / run_name / "prediction.json"
    nms_params_path = target_dir / run_name / "nms_params.json"
    inference_config_path = target_dir / run_name / "inference_config.json"

    _write_json(
        latency_path,
        _latency_report_dict(
            image_path=image_path,
            model_path=model_path,
            warmup_runs=warmup,
            latencies_ms=latencies_ms,
        ),
    )
    _write_json(prediction_path, pred_payload)
    _write_json(nms_params_path, nms_params)
    _write_json(inference_config_path, model._inference_config.model_dump_json())

    click.echo("Done!")

if __name__ == "__main__":
    main()
