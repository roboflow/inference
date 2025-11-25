from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import torch

from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    ray_bundle_to_ray_points,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds

INVALID_XYZ = np.inf


def get_img_coords_grid(H, W):
    U, V = np.mgrid[0:W, 0:H]
    _uv = np.vstack((U.T.ravel(), V.T.ravel()))
    uv = np.vstack([_uv, np.ones(U.size)])
    return uv


def get_point_map(depth, intrinsics, H, W):
    # Load depth map, resize to image dims
    depth[depth == 0] = INVALID_XYZ
    depth_resized = cv2.resize(depth, (W, H))

    # Unproject pixel coordinates to 3D
    uv = get_img_coords_grid(H, W)
    inv_intrinsics = np.linalg.inv(intrinsics)
    xyz = (inv_intrinsics @ uv) * depth_resized.ravel()
    return xyz.T.reshape(H, W, 3)


def get_points_in_front_of_camera(pcd, box_width=10, box_depth=20):
    # Restrict to points in front of camera (nonnegative depth)
    bbox_fov = o3d.geometry.AxisAlignedBoundingBox(
        (-box_width, -box_width, 0), (box_width, box_width, box_depth)
    )
    return pcd.crop(bbox_fov)


def get_points_in_frustum(pcd, intrinsics, H, W, sigma=0.25):
    # Project points onto image plane
    # Limit to points within image boundaries, plus some expansion (sigma)
    uv = (intrinsics @ np.array(pcd.points).T).T
    uv = uv / uv[:, 2][..., np.newaxis]  # homogenize by dividing by depth

    # Get points whose coordinates are within image boundaries, plus some expansion
    in_x_bounds = (uv[:, 0] > (0 - sigma * W)) & (uv[:, 0] < (W + sigma * W))
    in_y_bounds = (uv[:, 1] > (0 - sigma * H)) & (uv[:, 1] < (H + sigma * H))

    in_view = in_x_bounds & in_y_bounds
    in_view_idx = np.where(in_view)[0]

    return pcd.select_by_index(in_view_idx)


def sample_pcd(pcd, n_gt_pts):
    # Sample points from GT point cloud, w/ replacement if not enough GT points
    n_pts = len(pcd.points)
    sample_replace = n_gt_pts > n_pts
    sampled_idx = np.sort(np.random.choice(n_pts, n_gt_pts, replace=sample_replace))

    # Get xyz and color of sampled points
    gt_xyz = np.array(pcd.points)[sampled_idx]
    gt_rgb = np.array(pcd.colors)[sampled_idx]

    return gt_xyz, gt_rgb


def normalize_objects_in_cam(points, cam_transform, box_size=6):
    points = cam_transform.transform_points(points)
    # normalize points to [-3, 3], 0 center
    min_box = points.min(0).values
    max_box = points.max(0).values
    max_side = (max_box - min_box).max()
    if max_side == 0:
        max_side = 1
    points -= (min_box + max_box) / 2
    points *= box_size / max_side
    return points


def get_rgbd_points(
    imh,
    imw,
    camera: CamerasBase,
    depth_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mask_thr: float = 0.5,
) -> Pointclouds:
    """
    Given a batch of images, depths, masks and cameras, generate a colored
    point cloud by unprojecting depth maps to the  and coloring with the source
    pixel colors.
    """
    depth_map = torch.nn.functional.interpolate(
        depth_map,
        size=[imh, imw],
        mode="nearest",
    )
    # convert the depth maps to point clouds using the grid ray sampler
    pts_3d = ray_bundle_to_ray_points(
        NDCMultinomialRaysampler(
            image_width=imw,
            image_height=imh,
            n_pts_per_ray=1,
            min_depth=1.0,
            max_depth=1.0,
        )(camera)._replace(lengths=depth_map[:, 0, ..., None])
    ).squeeze(3)[None]

    pts_mask = depth_map > 0.0
    if mask is not None:
        mask = torch.nn.functional.interpolate(
            mask,
            size=[imh, imw],
            mode="bilinear",
            align_corners=False,
        )
        pts_mask *= mask > mask_thr
    pts_3d[~pts_mask] = float("inf")
    return pts_3d.squeeze(0).squeeze(0), pts_mask
