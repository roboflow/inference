import os
import json
import logging
from pathlib import Path
from typing import Optional

import click
import cv2
from inference_models import AutoModel
import supervision as sv

from inference.core.utils.image_utils import load_image_rgb
from inference.models.yolo_world.yolo_world import YOLOWorld


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_image_path(sequence_name: str, frame_number: int) -> Path:
    """Get the image path from a frame annotation."""
    with open(DATA_DIR / "frame_annotations", "r") as f:
        frame_annotations = json.load(f)

    frame_sequence = [
        frame_annotation for frame_annotation in frame_annotations
        if frame_annotation["sequence_name"] == sequence_name
    ]

    frame_annotation = [
        frame_annotation for frame_annotation in frame_sequence
        if frame_annotation["frame_number"] == frame_number
    ][0]

    return DATA_DIR / frame_annotation["image"]["path"]
    

@click.command()
@click.option("--sequence-name", type=str, default="246_26304_51384")
@click.option("--frame-number", type=int, default=1)
@click.option("--output-path", type=click.Path(path_type=Path), default=OUTPUT_DIR / "mask.png")
def main(sequence_name: str, frame_number: int, output_path: Optional[Path] = None) -> None:
    model = YOLOWorld(model_id="yolo_world/s")
    image_path = get_image_path(sequence_name, frame_number)

    classes = ["teddybear"]
    results = model.infer(str(image_path), text=classes, confidence=0.03)

    detections = sv.Detections.from_inference(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [classes[class_id] for class_id in detections.class_id]

    annotated_image = bounding_box_annotator.annotate(
        scene=cv2.imread(str(image_path)), detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    image = load_image_rgb(str(image_path))

    sam_model = AutoModel.from_pretrained("sam2/hiera_b_plus", api_key=os.getenv("ROBOFLOW_API_KEY"))

    results_prompt = sam_model.segment_images(
        images=image,
        boxes=detections.xyxy,
        input_color_format="rgb",
    )

    masks_prompt = results_prompt[0].masks
    scores_prompt = results_prompt[0].scores

    best_mask_idx = scores_prompt.argmax()
    best_mask_prompt = masks_prompt[best_mask_idx].numpy()

    sv.plot_images_grid(
        [cv2.cvtColor(image, cv2.COLOR_RGB2BGR), annotated_image, best_mask_prompt * 255],
        grid_size=(1, 3),
        titles=["Image", "Annotated Image", "Mask Prompt"],
    )

if __name__ == "__main__":
    main()
