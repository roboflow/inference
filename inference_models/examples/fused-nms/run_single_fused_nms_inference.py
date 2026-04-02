from pathlib import Path
from typing import Optional

import click
import cv2

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
def main(
    image_path: Path,
    model_path: Path,
    confidence: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    onnx_ep_preset: str = "cpu",
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

    click.echo("Running inference...")
    predictions = model(image, **nms_params)
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
