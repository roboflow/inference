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
from inference.core.workflows.core_steps.classical_cv.sift.v1 import apply_sift
from inference.models.yolo_world.yolo_world import YOLOWorld, ObjectDetectionInferenceResponse
from inference_models import AutoModel
from inference_models.models.sam2.sam2_torch import SAM2Prediction

from inference.core.workflows.core_steps.transformations.essential_matrix.v1 import EssentialMatrixBlockV1
from inference.core.workflows.core_steps.classical_cv.feature_comparison.v1 import FeatureComparisonBlockV1


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
    def camera_intrinsics(self) -> dict[str, float]:
        return camera_intrinsics_from_annotation(self)

    @property
    def camera_extrinsics(self) -> dict[str, np.ndarray]:
        R = np.asarray(self.viewpoint["R"])
        T = np.asarray(self.viewpoint["T"])

        S = np.diag([-1, -1, 1]).astype(np.float32)
        R = S @ R
        T = S @ T

        return {
            "R": R,
            "t": T,
        }


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


def camera_intrinsics_from_annotation(frame_annotation: FrameAnnotation) -> dict[str, float]:
    """Compute (fx, fy, cx, cy) from frame viewpoint and image size."""
    p = frame_annotation.viewpoint["principal_point"]
    f = frame_annotation.viewpoint["focal_length"]

    h, w = frame_annotation.image_data["size"]
    s = (min(h, w) - 1) / 2

    fx = f[0] * (w - 1) / 2
    fy = f[1] * (h - 1) / 2
    cx = -p[0] * s + (w - 1) / 2
    cy = -p[1] * s + (h - 1) / 2

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }


def _intrinsics_to_K(intrinsics: dict[str, float]) -> np.ndarray:
    """Build 3x3 calibration matrix from camera intrinsics dict."""
    return np.array(
        [
            [float(intrinsics["fx"]), 0, float(intrinsics["cx"])],
            [0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def _intrinsics_to_K_and_D(intrinsics: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Build 3x3 K and 5-element dist (k1, k2, p1, p2, k3) from dict."""
    K = np.array(
        [
            [float(intrinsics["fx"]), 0, float(intrinsics["cx"])],
            [0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0, 0, 1],
        ]
    )
    D = np.array(
        [
            float(intrinsics.get("k1", 0)),
            float(intrinsics.get("k2", 0)),
            float(intrinsics.get("p1", 0)),
            float(intrinsics.get("p2", 0)),
            float(intrinsics.get("k3", 0)),
        ],
        dtype=np.float64,
    )
    return K, D


def _extract_point_pairs(good_matches: list) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Extract (pt1, pt2) from good_matches; pt1/pt2 are (x, y)."""
    pairs = []
    for m in good_matches or []:
        if not isinstance(m, dict):
            continue
        kp = m.get("keypoint_pairs")
        if not kp or len(kp) != 2:
            continue
        pt1, pt2 = kp[0], kp[1]
        if pt1 is None or pt2 is None:
            continue
        try:
            x1, y1 = float(pt1[0]), float(pt1[1])
            x2, y2 = float(pt2[0]), float(pt2[1])
        except (TypeError, IndexError):
            continue
        pairs.append(((x1, y1), (x2, y2)))
    return pairs


def triangulate_points_opencv(
    good_matches: list,
    camera_intrinsics_1: dict[str, float],
    camera_intrinsics_2: dict[str, float],
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Triangulate matched keypoints using OpenCV's triangulatePoints.
    Returns (N, 3) array of 3D points in camera 1 coordinate frame.
    """
    pairs = _extract_point_pairs(good_matches)
    if not pairs:
        return np.zeros((0, 3), dtype=np.float64)
    pts1 = np.array([[p[0][0], p[0][1]] for p in pairs], dtype=np.float64)
    pts2 = np.array([[p[1][0], p[1][1]] for p in pairs], dtype=np.float64)
    K1 = _intrinsics_to_K(camera_intrinsics_1)
    K2 = _intrinsics_to_K(camera_intrinsics_2)
    t = np.asarray(t, dtype=np.float64).ravel()[:3]
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    P1 = np.hstack([K1, np.zeros((3, 1), dtype=np.float64)])
    P2 = K2 @ np.hstack([R, t.reshape(3, 1)])
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d[:3] / (points_4d[3] + 1e-12)).T
    return points_3d


def relative_pose_wold2cam(
    R_i: np.ndarray,
    t_i: np.ndarray,
    R_j: np.ndarray,
    t_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
      R_i, R_j: (3,3) world->cam rotations
      t_i, t_j: (3,)  world->cam translations

    Returns:
      R_rel, t_rel such that x_j = R_rel @ x_i + t_rel
    """
    R_rel = R_j @ R_i.T
    t_rel = t_j - R_rel @ t_i
    return R_rel, t_rel


def transform_cam2_to_cam1(
    points_cam2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Transform 3D points from camera 2 frame to camera 1 frame.
    Convention: x_cam2 = R @ x_cam1 + t, so x_cam1 = R.T @ (x_cam2 - t).
    points_cam2: (N, 3). Returns (N, 3) in camera 1 frame.
    """
    t = np.asarray(t, dtype=np.float64).ravel()[:3]
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    return (points_cam2 - t) @ R


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


def rectify_images(
    img1: np.ndarray,
    img2: np.ndarray,
    camera_intrinsics_1: dict[str, float],
    camera_intrinsics_2: dict[str, float],
    rotation: np.ndarray,
    translation: np.ndarray,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if img1 is None or img2 is None:
        raise ValueError("Stereo rectification requires loaded numpy images.")
    h, w = img1.shape[:2]
    size = (w, h)

    K1, D1 = _intrinsics_to_K_and_D(camera_intrinsics_1)
    K2, D2 = _intrinsics_to_K_and_D(camera_intrinsics_2)
    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T = np.asarray(translation, dtype=np.float64).ravel()[:3]

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
    )

    map1_1, map1_2 = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, size, cv2.CV_32FC1
    )
    map2_1, map2_2 = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, size, cv2.CV_32FC1
    )

    rect1 = cv2.remap(img1, map1_1, map1_2, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2_1, map2_2, cv2.INTER_LINEAR)

    return rect1, rect2


def remove_outliers(points: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Remove outliers from point cloud using statistical methods.
    Drops an observation if at least one of its coordinates is over the threshold
    (in units of per-coordinate standard deviation from the mean).
    """
    if len(points) < 3:
        return points

    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    std = np.sqrt(np.diag(cov))
    within = np.abs(points - mean) < threshold * std
    keep = np.all(within, axis=1)
    return points[keep]


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

    camera_intrinsics: list[dict[str, float]] = [frame_annotation.camera_intrinsics for frame_annotation in frame_annotations]

    points_3d: list[np.ndarray] = [
        backproject_mask_to_camera_xyz(
            mask=mask,
            depth_map=depth_map,
            fx=camera_intrinsics["fx"],
            fy=camera_intrinsics["fy"],
            cx=camera_intrinsics["cx"],
            cy=camera_intrinsics["cy"],
            subsample=point_cloud_subsample,
        )
        for mask, depth_map, camera_intrinsics in zip(final_masks, depth_maps, camera_intrinsics)
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

    titles_list = [[f"Image {idx}", f"Detection {idx}", f"Mask {idx}", f"Depth Map {idx}"] for idx in range(len(visualizations_list))]

    for visualizations, titles in zip(visualizations_list, titles_list):
        sv.plot_images_grid(
            images=visualizations,
            grid_size=(2, 2),
            titles=titles,
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
        mask_1=None,
        mask_2=None,
    )
    logging.info(
        "Feature comparison (masked): %d good matches",
        fc_results["good_matches_count"],
    )

    R_rel, t_rel = relative_pose_wold2cam(
        R_i=frame_annotations[0].camera_extrinsics["R"],
        t_i=frame_annotations[0].camera_extrinsics["t"],
        R_j=frame_annotations[1].camera_extrinsics["R"],
        t_j=frame_annotations[1].camera_extrinsics["t"],
    )

    logging.info(
        "Relative pose: %s",
        R_rel,
    )
    logging.info(
        "Relative translation: %s",
        t_rel,
    )

    essential_matrix_block = EssentialMatrixBlockV1()

    em_results = essential_matrix_block.run(
        good_matches=fc_results["good_matches"],
        camera_intrinsics_1=camera_intrinsics[0],
        camera_intrinsics_2=camera_intrinsics[1],
        pose_estimation="opencv",
    )

    logging.info(
        "Essential matrix: %s",
        em_results["essential_matrix"],
    )
    logging.info(
        "Rotation: %s",
        em_results["rotation"],
    )
    logging.info(
        "Translation: %s",
        em_results["translation"],
    )

    R_est = em_results["rotation"]
    t_est = em_results["translation"]

    # Fuse backprojected point clouds into camera 1 frame
    points_cam1 = points_3d[0]
    points_cam2_in_cam1 = transform_cam2_to_cam1(points_3d[1], R_est, t_est)
    fused_point_cloud = np.vstack([points_cam1, points_cam2_in_cam1])
    logging.info(
        "Fused point cloud: %d from image 1 + %d from image 2 = %d points (camera 1 frame)",
        len(points_cam1),
        len(points_cam2_in_cam1),
        len(fused_point_cloud),
    )
    # Log point sets side by side (first N points from each)
    _n_show = min(10, len(points_cam1), len(points_cam2_in_cam1))
    if _n_show > 0:
        logging.info(
            "Point correspondences (camera 1 frame) — image 1 vs image 2 (first %d):",
            _n_show,
        )
        for i in range(_n_show):
            p1 = points_cam1[i]
            p2 = points_cam2_in_cam1[i]
            logging.info(
                "  [%d]  (%.4f, %.4f, %.4f)  |  (%.4f, %.4f, %.4f)",
                i,
                p1[0],
                p1[1],
                p1[2],
                p2[0],
                p2[1],
                p2[2],
            )
    if len(fused_point_cloud) > 0:
        fig_fused = go.Figure(
            data=[
                go.Scatter3d(
                    x=fused_point_cloud[:, 0],
                    y=fused_point_cloud[:, 1],
                    z=fused_point_cloud[:, 2],
                    mode="markers",
                    marker=dict(
                        size=1.5,
                        color=fused_point_cloud[:, 2],
                        colorscale="Viridis",
                        opacity=0.8,
                    ),
                    name="Fused point cloud",
                )
            ],
            layout=go.Layout(
                title=f"{class_name} fused point cloud (camera 1 frame)",
                scene=dict(
                    xaxis_title="X (right)",
                    yaxis_title="Y (down)",
                    zaxis_title="Z (forward)",
                    aspectmode="data",
                ),
            ),
        )
        fused_html_path = OUTPUT_DIR / f"{class_name}_fused_pointcloud.html"
        fig_fused.write_html(str(fused_html_path))
        logging.info("Saved fused point cloud to %s", fused_html_path)
        fig_fused.show()

    # Triangulate keypoint correspondences with OpenCV
    triangulated_3d = triangulate_points_opencv(
        good_matches=fc_results["good_matches"],
        camera_intrinsics_1=camera_intrinsics[0],
        camera_intrinsics_2=camera_intrinsics[1],
        R=R_est,
        t=t_est,
    )
    object_center = np.mean(triangulated_3d, axis=0) if len(triangulated_3d) > 0 else np.zeros(3)
    cam1_position = np.array([0.0, 0.0, 0.0])
    cam2_position = -R_est.T @ np.asarray(t_est).ravel()[:3]

    # Remove outliers from triangulated_3d before visualization
    triangulated_3d = remove_outliers(triangulated_3d)

    # 3D visualization: cameras and object center (camera 1 frame)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=[cam1_position[0]],
            y=[cam1_position[1]],
            z=[cam1_position[2]],
            mode="markers+text",
            marker=dict(size=10, color="blue", symbol="diamond"),
            text=["Cam 1"],
            textposition="top center",
            name="Camera 1",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[cam2_position[0]],
            y=[cam2_position[1]],
            z=[cam2_position[2]],
            mode="markers+text",
            marker=dict(size=10, color="green", symbol="diamond"),
            text=["Cam 2"],
            textposition="top center",
            name="Camera 2",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[object_center[0]],
            y=[object_center[1]],
            z=[object_center[2]],
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="circle"),
            text=["Object center"],
            textposition="top center",
            name="Object center",
        )
    )
    if len(triangulated_3d) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=triangulated_3d[:, 0],
                y=triangulated_3d[:, 1],
                z=triangulated_3d[:, 2],
                mode="markers",
                marker=dict(size=2, color="gray", opacity=0.5),
                name="Triangulated points",
            )
        )
    fig.update_layout(
        title="Cameras and object center (camera 1 frame)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        showlegend=True,
    )
    viz_path = OUTPUT_DIR / f"{class_name}_cameras_and_object_center.html"
    fig.write_html(str(viz_path))
    logging.info("Saved cameras + object center 3D viz to %s", viz_path)
    fig.show()

    img1 = cv2.imread(frame_annotations[0].image_path)
    img2 = cv2.imread(frame_annotations[1].image_path)

    rect_results = rectify_images(
        img1=img1,
        img2=img2,
        camera_intrinsics_1=camera_intrinsics[0],
        camera_intrinsics_2=camera_intrinsics[1],
        rotation=em_results["rotation"],
        translation=em_results["translation"],
    )

    rect1, rect2 = rect_results
    sv.plot_images_grid(
        images=[img1, img2, rect1, rect2],
        grid_size=(2, 2),
        titles=["Image 1", "Image 2", "Rectified image 1", "Rectified image 2"],
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
