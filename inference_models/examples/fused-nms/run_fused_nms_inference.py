import statistics
import time
from pathlib import Path
from typing import Optional

import click
import cv2

from inference_models import AutoModel


@click.command()
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
    type=click.IntRange(min=0),
    default=0,
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
    help="Untimed warmup runs before timed iterations (only used when -n > 0).",
)
def main(
    image_path: Path,
    model_path: Path,
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

    if benchmark_iters > 0:
        if warmup > 0:
            click.echo(f"Warmup: {warmup} untimed run(s)...")
            for _ in range(warmup):
                model(image, **nms_params)

        click.echo(f"Benchmark: {benchmark_iters} timed run(s)...")

        latencies_s: list[float] = []
        predictions = None
        for _ in range(benchmark_iters):
            t0 = time.perf_counter()
            predictions = model(image, **nms_params)
            latencies_s.append(time.perf_counter() - t0)

        latencies_ms = [t * 1000.0 for t in latencies_s]
        mean_ms = statistics.mean(latencies_ms)
        median_ms = statistics.median(latencies_ms)

        if len(latencies_ms) > 1:
            stdev_str = f"{statistics.stdev(latencies_ms):.4f}"
        else:
            stdev_str = "n/a (use -n 2 or more for std)"
            
        click.echo(
            f"Inference latency (ms): mean={mean_ms:.4f}, median={median_ms:.4f}, "
            f"std={stdev_str}"
        )
    else:
        click.echo("Running inference...")
        predictions = model(image, **nms_params)

    assert predictions is not None
    detections = predictions[0].to_supervision()

    click.echo(f"Detected {len(detections)} objects")
    for idx, (xyxy, class_id, conf) in enumerate(
        zip(detections.xyxy, detections.class_id, detections.confidence), start=1
    ):
        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
        click.echo(
            f"[{idx}] class_id={int(class_id)} confidence={float(conf):.4f} "
            f"bbox=({x1}, {y1}, {x2}, {y2})"
        )


if __name__ == "__main__":
    main()
