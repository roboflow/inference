from pathlib import Path

import click
import cv2
from typing import Optional
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
def main(
    image_path: Path,
    model_path: Path,
    confidence: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise click.ClickException(f"Could not load image from: {image_path}")

    click.echo(f"Loading model: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        confidence=confidence,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
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
    predictions = model(image, confidence=confidence)
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
