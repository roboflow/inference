import json
import time
from pathlib import Path
from typing import Any, Optional

import click
import cv2
import numpy as np

from inference_models import AutoModel


def _onnx_ep_preset_to_providers_and_device(
    preset: str,
) -> tuple[list[str], str]:
    """Map CLI preset to ONNX Runtime provider chain and PyTorch device string."""
    if preset == "cpu":
        return (["CPUExecutionProvider"], "cpu")
    if preset == "cuda":
        return (["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda")
    if preset == "tensorrt":
        return (
            [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "cuda",
        )
    raise click.ClickException(f"Unknown onnx-execution-providers preset: {preset!r}")


def _latency_report_dict(
    *,
    model_path: Path,
    warmup_runs: int,
    latencies_ms: list[float],
    onnx_execution_providers_preset: str,
    onnx_execution_providers: list[str],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "model_path": str(model_path.resolve()),
        "onnx_execution_providers_preset": onnx_execution_providers_preset,
        "onnx_execution_providers": onnx_execution_providers,
        "device": device,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "timed_runs": len(latencies_ms),
        "mean_ms": np.mean(latencies_ms),
        "p_50_ms": np.percentile(latencies_ms, 50),
        "p_95_ms": np.percentile(latencies_ms, 95),
        "p_99_ms": np.percentile(latencies_ms, 99),
        "mean_per_image_ms": np.mean(latencies_ms) / batch_size,
        "throughput_fps": (batch_size * len(latencies_ms)) / (np.sum(latencies_ms) / 1000),
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
    "--image-dir",
    type=click.Path(path_type=Path, exists=True, dir_okay=True, readable=True),
    required=True,
    help="Path to the input image directory.",
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
    help="Directory for latency.json (created if missing).",
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
@click.option(
    "--onnx-execution-providers",
    "onnx_ep_preset",
    type=click.Choice(["cpu", "cuda", "tensorrt"], case_sensitive=False),
    default="cpu",
    show_default=True,
    help=(
        "ONNX Runtime execution provider chain: "
        "cpu (CPUExecutionProvider); "
        "cuda (CUDAExecutionProvider then CPUExecutionProvider); "
        "tensorrt (TensorrtExecutionProvider, CUDA, then CPU fallbacks)."
    ),
)
@click.option(
    "--batch-images",
    type=click.Flag(default=False),
    help="Batch the input images and run a single batched forward pass.",
)
def main(
    run_name: str,
    image_dir: Path,
    model_path: Path,
    target_dir: Path,
    confidence: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    benchmark_iters: int = 0,
    warmup: int = 5,
    onnx_ep_preset: str = "cpu",
    batch_images: bool = False,
) -> None:
    image_paths = list(image_dir.glob("*.jpg"))

    images = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise click.ClickException(f"Could not load image from: {image_path}")
        images.append(image)

    if batch_images:
        batch_input = images[:4]
    else:
        batch_input = images[0]

    nms_params = {
        "confidence": confidence,
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
    }

    nms_params = {name: value for name, value in nms_params.items() if value is not None}
    if nms_params:
        click.echo(f"User provided NMS parameters: {nms_params}")

    onnx_ep_preset = onnx_ep_preset.lower()
    onnx_providers, device_str = _onnx_ep_preset_to_providers_and_device(onnx_ep_preset)
    click.echo(
        f"Loading model: {model_path} "
        f"(onnx_execution_providers={onnx_providers!r}, device={device_str!r})"
    )
    model = AutoModel.from_pretrained(
        model_path,
        onnx_execution_providers=list(onnx_providers),
        device=device_str,
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

    click.echo(f"Warmup: {warmup} untimed runs..." if warmup > 0 else "No warmup runs.")

    for _ in range(warmup):
        predictions = model(batch_input, **nms_params)
        _ = predictions[0].to_supervision()

    click.echo(f"Benchmarking: {benchmark_iters} timed runs...")

    latencies_ms: list[float] = []
    for _ in range(benchmark_iters):
        t0 = time.perf_counter()
        predictions = model(batch_input, **nms_params)
        _ = predictions[0].to_supervision()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    click.echo("Writing reports ...")

    target_dir.mkdir(parents=True, exist_ok=True)
    latency_path = target_dir / run_name / "latency.json"
    nms_params_path = target_dir / run_name / "nms_params.json"
    inference_config_path = target_dir / run_name / "inference_config.json"

    _write_json(
        latency_path,
        _latency_report_dict(
            model_path=model_path,
            warmup_runs=warmup,
            latencies_ms=latencies_ms,
            onnx_execution_providers_preset=onnx_ep_preset,
            onnx_execution_providers=list(onnx_providers),
            device=device_str,
            batch_size=len(batch_input) if isinstance(batch_input, list) else 1,
        ),
    )
    _write_json(nms_params_path, nms_params)
    _write_json(inference_config_path, model._inference_config.model_dump_json())

    click.echo("Done!")

if __name__ == "__main__":
    main()
