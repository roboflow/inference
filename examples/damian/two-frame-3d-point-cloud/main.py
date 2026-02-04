import os
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import click
import cv2
import numpy as np
import plotly.graph_objects as go
from inference_models import AutoModel
import supervision as sv

from inference.core.utils.image_utils import load_image_rgb
from inference.models.yolo_world.yolo_world import YOLOWorld


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def get_frame_annotation(sequence_name: str, frame_number: int) -> dict:
    """Get the frame annotation for the given sequence and frame."""
    with open(DATA_DIR / "frame_annotations", "r") as f:
        frame_annotations = json.load(f)

    frame_sequence = [
        fa for fa in frame_annotations
        if fa["sequence_name"] == sequence_name
    ]
    frame_annotation = [
        fa for fa in frame_sequence
        if fa["frame_number"] == frame_number
    ][0]
    return frame_annotation


def get_image_path(sequence_name: str, frame_number: int) -> Path:
    """Get the image path from a frame annotation."""
    frame_annotation = get_frame_annotation(sequence_name, frame_number)
    return DATA_DIR / frame_annotation["image"]["path"]


def camera_intrinsics_from_annotation(frame_annotation: dict) -> Tuple[float, float, float, float]:
    """Compute (fx, fy, cx, cy) from frame viewpoint and image size."""
    p = frame_annotation["viewpoint"]["principal_point"]
    f = frame_annotation["viewpoint"]["focal_length"]
    h, w = frame_annotation["image"]["size"]
    s = (min(h, w) - 1) / 2
    fx = f[0] * (w - 1) / 2
    fy = f[1] * (h - 1) / 2
    cx = -p[0] * s + (w - 1) / 2
    cy = -p[1] * s + (h - 1) / 2
    return fx, fy, cx, cy


def backproject_mask_to_camera_xyz(
    mask: np.ndarray,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    subsample: int = 1,
) -> np.ndarray:
    """
    Back-project pixels where mask is True to 3D camera coordinates (X, Y, Z).

    Camera convention: X right, Y down, Z forward. Depth is along Z.
    Pixels with non-positive depth are excluded. For monocular depth (e.g. Depth
    Anything), scale is arbitrary; geometry is consistent but not metric.
    """
    if mask.shape != depth_map.shape:
        depth_map = cv2.resize(
            depth_map,
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    mask_flat = mask.astype(bool).ravel()
    h, w = mask.shape
    v = np.arange(h, dtype=np.float64)
    u = np.arange(w, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    u_flat = uu.ravel()[mask_flat]
    v_flat = vv.ravel()[mask_flat]
    z_flat = np.asarray(depth_map, dtype=np.float64).ravel()[mask_flat]
    if subsample > 1:
        u_flat = u_flat[::subsample]
        v_flat = v_flat[::subsample]
        z_flat = z_flat[::subsample]
    valid = z_flat > 0
    u_flat = u_flat[valid]
    v_flat = v_flat[valid]
    z_flat = z_flat[valid]
    x = (u_flat - cx) * z_flat / fx
    y = (v_flat - cy) * z_flat / fy
    return np.column_stack([x, y, z_flat])
    

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

    depth_anything_model = AutoModel.from_pretrained("depth-anything-v3/small", api_key=os.getenv("ROBOFLOW_API_KEY"))

    depth_results = depth_anything_model.infer(
        images=image,
    )

    depth_map = depth_results[0].numpy()

    frame_annotation = get_frame_annotation(sequence_name, frame_number)
    fx, fy, cx, cy = camera_intrinsics_from_annotation(frame_annotation)

    points_3d = backproject_mask_to_camera_xyz(
        best_mask_prompt,
        depth_map,
        fx, fy, cx, cy,
        subsample=4,
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1],
                z=points_3d[:, 2],
                mode="markers",
                marker=dict(size=1.5, color=points_3d[:, 2], colorscale="Viridis", opacity=0.8),
                name="teddybear",
            )
        ],
        layout=go.Layout(
            title="Teddy bear 3D point cloud (camera coordinates)",
            scene=dict(
                xaxis_title="X (right)",
                yaxis_title="Y (down)",
                zaxis_title="Z (forward)",
                aspectmode="data",
            ),
        ),
    )
    html_path = OUTPUT_DIR / "teddybear_pointcloud.html"
    fig.write_html(str(html_path))
    logging.info("Saved %d 3D points to %s", len(points_3d), html_path)
    fig.show()

    sv.plot_images_grid(
        [cv2.cvtColor(image, cv2.COLOR_RGB2BGR), annotated_image, best_mask_prompt * 255, depth_map],
        grid_size=(2, 2),
        titles=["Image", "Annotated Image", "Mask Prompt", "Depth Map"],
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
