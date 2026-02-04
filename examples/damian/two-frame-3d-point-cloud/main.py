import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import click
import cv2
import numpy as np
import plotly.graph_objects as go
import torch
import supervision as sv

from inference.core.utils.image_utils import load_image_rgb
from inference.core.workflows.core_steps.classical_cv.feature_comparison.v1 import FeatureComparisonBlockV1
from inference.core.workflows.core_steps.classical_cv.sift.v1 import apply_sift
from inference.core.workflows.prototypes.block import BlockResult
from inference.models.yolo_world.yolo_world import YOLOWorld, ObjectDetectionInferenceResponse
from inference_models import AutoModel
from inference_models.models.sam2.sam2_torch import SAM2Prediction


# from inference.core.workflows.core_steps.transformations.essential_matrix.v1 import EssentialMatrixBlockV1


API_KEY = os.getenv("ROBOFLOW_API_KEY")

DATA_DIR_NAME = "data"
DATA_DIR = Path(__file__).resolve().parent.parent / DATA_DIR_NAME

FRAME_ANNOTATIONS_FILENAME = "frame_annotations.json"
FRAME_ANNOTATIONS_PATH = DATA_DIR / FRAME_ANNOTATIONS_FILENAME

OUTPUT_DIR_NAME = "output"
OUTPUT_DIR = Path(__file__).resolve().parent / OUTPUT_DIR_NAME
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass
class FrameAnnotation:
    frame_id: int
    viewpoint: dict
    image_data: dict

    @property
    def image_path(self) -> str:
        return str(Path(DATA_DIR) / self.image_data["path"])

    @property
    def camera_intrinsics(self) -> Tuple[float, float, float, float]:
        return camera_intrinsics_from_annotation(self)


def get_frame_sequence(frame_sequence_id: str) -> list[dict]:
    """Get the frame sequence for the given frame sequence ID.

    Args:
        frame_sequence_id: The ID of the frame sequence.

    Returns:
        A list of frame annotations.
    """
    with open(DATA_DIR / FRAME_ANNOTATIONS_FILENAME, "r") as f:
        frame_annotations = json.load(f)

    frame_sequence = [
        fa for fa in frame_annotations
        if fa["sequence_name"] == frame_sequence_id
    ]

    return frame_sequence


def get_frame_annotations_from_sequence(frame_sequence_id: str, frame_ids: list[int]) -> list[FrameAnnotation]:
    """Get the frame annotations from a frame sequence for the given frame IDs.
    
    Args:
        frame_sequence_id: The frame sequence ID.
        frame_ids: The frame IDs to get the annotations for.

    Returns:
        A list of frame annotations.
    """
    return [
        FrameAnnotation(
            frame_id=fa["frame_number"],
            viewpoint=fa["viewpoint"],
            image_data=fa["image"],
        )
        for fa in get_frame_sequence(frame_sequence_id=frame_sequence_id)
        if fa["frame_number"] in frame_ids
    ]


def camera_intrinsics_from_annotation(frame_annotation: FrameAnnotation) -> Tuple[float, float, float, float]:
    """Compute (fx, fy, cx, cy) from frame viewpoint and image size."""
    p = frame_annotation.viewpoint["principal_point"]
    f = frame_annotation.viewpoint["focal_length"]

    h, w = frame_annotation.image_data["size"]
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
@click.option(
    "--frame-sequence-id",
    type=str,
    default="246_26304_51384",
    help="Frame sequence ID to process.",
)
@click.option(
    "--frame-ids",
    type=str,
    default="1,13",
    help="Comma-separated list of frame IDs to process.",
    callback=lambda ctx, param, value: [int(x) for x in value.split(",")],
)
@click.option(
    "--class-name",
    type=str,
    default="teddybear",
    help="Class to process.",
)
@click.option(
    "--yolo-world-model-id",
    type=str,
    default="yolo_world/s",
    help="YOLO World model ID to use for object detection.",
)
@click.option(
    "--sam-model-id",
    type=str,
    default="sam2/hiera_b_plus",
    help="SAM model ID to use for segmentation.",
)
@click.option(
    "--depth-anything-model-id",
    type=str,
    default="depth-anything-v3/small",
    help="Depth Anything model ID to use for depth estimation.",
)
@click.option(
    "--detection-confidence",
    type=float,
    default=0.03,
    help="Confidence threshold for object detection.",
)
@click.option(
    "--point-cloud-subsample",
    type=int,
    default=4,
    help="Subsample factor for depth map.",
)
def main(
    frame_sequence_id: str,
    frame_ids: list[int],
    class_name: str,
    yolo_world_model_id: str,
    sam_model_id: str,
    depth_anything_model_id: str,
    detection_confidence: float,
    point_cloud_subsample: int,
) -> None:
    object_detection_model = YOLOWorld(
        model_id=yolo_world_model_id
    )
    sam_model = AutoModel.from_pretrained(
        model_id_or_path=sam_model_id,
        api_key=API_KEY,
    )
    depth_anything_model = AutoModel.from_pretrained(
        model_id_or_path=depth_anything_model_id,
        api_key=API_KEY,
    )

    frame_annotations: list[FrameAnnotation] = get_frame_annotations_from_sequence(
        frame_sequence_id=frame_sequence_id,
        frame_ids=frame_ids,
    )

    object_detection_results_batch: list[ObjectDetectionInferenceResponse] = [
        object_detection_model.infer(
            image=frame_annotation.image_path,
            text=[class_name],
            confidence=detection_confidence,
        )
        for frame_annotation in frame_annotations
    ]

    detections_batch: list[sv.Detections] = [
        sv.Detections.from_inference(result)
        for result in object_detection_results_batch
    ]

    segmentation_results_batch: list[list[SAM2Prediction]] = [
        sam_model.segment_images(
            images=load_image_rgb(frame_annotation.image_path),
            boxes=detections.xyxy,
            input_color_format="rgb",
        )
        for frame_annotation, detections in zip(frame_annotations, detections_batch)
    ]

    final_masks: list[np.ndarray] = []
    for segmentation_result in segmentation_results_batch:
        masks = segmentation_result[0].masks
        scores = segmentation_result[0].scores

        final_mask_index = scores.argmax()
        final_mask = masks[final_mask_index].detach().cpu().numpy()
        final_masks.append(final_mask)

    depth_maps_batch: list[torch.Tensor] = depth_anything_model.infer(
        images=[load_image_rgb(frame_annotation.image_path) for frame_annotation in frame_annotations],
    )
    depth_maps: list[np.ndarray] = [map.numpy() for map in depth_maps_batch]

    camera_intrinsics: list[Tuple[float, float, float, float]] = [frame_annotation.camera_intrinsics for frame_annotation in frame_annotations]

    points_3d: list[np.ndarray] = [
        backproject_mask_to_camera_xyz(
            mask=mask,
            depth_map=depth_map,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            subsample=point_cloud_subsample,
        )
        for mask, depth_map, (fx, fy, cx, cy) in zip(final_masks, depth_maps, camera_intrinsics)
    ]

    for point_cloud, frame_annotation in zip(points_3d, frame_annotations):
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    z=point_cloud[:, 2],
                    mode="markers",
                    marker=dict(size=1.5, color=point_cloud[:, 2], colorscale="Viridis", opacity=0.8),
                    name=f"{class_name} {frame_annotation.frame_id}",
                )
            ],
            layout=go.Layout(
                title=f"{class_name} {frame_annotation.frame_id} 3D point cloud (camera coordinates)",
                scene=dict(
                    xaxis_title="X (right)",
                    yaxis_title="Y (down)",
                    zaxis_title="Z (forward)",
                    aspectmode="data",
                ),
            ),
        )

        html_path = OUTPUT_DIR / f"{class_name}_{frame_annotation.frame_id}_pointcloud.html"
        fig.write_html(str(html_path))
        logging.info("Saved %d 3D points to %s", len(points_3d), html_path)
        fig.show()

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    visualizations_list = []

    for idx, (detections, frame_annotation) in enumerate(zip(detections_batch, frame_annotations)):
        image_with_detections = bounding_box_annotator.annotate(
            scene=cv2.imread(frame_annotation.image_path), detections=detections
        )
        image_with_detections = label_annotator.annotate(
            scene=image_with_detections, detections=detections, labels=[class_name]
        )

        image = cv2.imread(frame_annotation.image_path)
        mask = final_masks[idx] * 255
        depth_map = depth_maps[idx]

        visualizations = []
        visualizations.append(image)
        visualizations.append(image_with_detections)
        visualizations.append(mask)
        visualizations.append(depth_map)
        visualizations_list.append(visualizations)

    titles = [[f"Image {idx}", f"Detection {idx}", f"Mask {idx}", f"Depth Map {idx}"] for idx in range(len(visualizations_list))]

    sv.plot_images_grid(
        images=[visualization for visualizations in visualizations_list for visualization in visualizations],
        grid_size=(len(visualizations_list), 4),
        titles=[title for sublist in titles for title in sublist],
    )

    sift_results_batch: list[tuple[np.ndarray, list, np.ndarray]] = []

    for frame_annotation in frame_annotations:
        image = cv2.imread(frame_annotation.image_path)
        sift_results = apply_sift(image)
        sift_results_batch.append(sift_results)
        

    feature_comparison_block = FeatureComparisonBlockV1()

    r1, r2 = sift_results_batch[0], sift_results_batch[1]
    img1_h, img1_w = cv2.imread(frame_annotations[0].image_path).shape[:2]
    img2_h, img2_w = cv2.imread(frame_annotations[1].image_path).shape[:2]
    mask_1 = final_masks[0]
    mask_2 = final_masks[1]
    if mask_1.shape[:2] != (img1_h, img1_w):
        mask_1 = cv2.resize(
            mask_1.astype(np.uint8),
            (img1_w, img1_h),
            interpolation=cv2.INTER_NEAREST,
        )
    if mask_2.shape[:2] != (img2_h, img2_w):
        mask_2 = cv2.resize(
            mask_2.astype(np.uint8),
            (img2_w, img2_h),
            interpolation=cv2.INTER_NEAREST,
        )

    fc_results = feature_comparison_block.run(
        keypoints_1=r1[1],
        descriptors_1=r1[2],
        keypoints_2=r2[1],
        descriptors_2=r2[2],
        mask_1=mask_1,
        mask_2=mask_2,
    )
    logging.info(
        "Feature comparison (masked): %d good matches",
        fc_results["good_matches_count"],
    )

    # essential_matrix_block = EssentialMatrixBlockV1()
    # em_results = essential_matrix_block.run(
    #     good_matches=fc_results["good_matches"],
    #     camera_intrinsics_1=camera_intrinsics,
    #     camera_intrinsics_2=camera_intrinsics,
    # )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
