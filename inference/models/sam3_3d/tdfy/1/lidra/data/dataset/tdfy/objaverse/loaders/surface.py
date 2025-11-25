import os
import trimesh
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
import pytorch3d

from pytorch3d.renderer.cameras import FoVPerspectiveCameras

from lidra.data.dataset.tdfy.point_cloud import get_rgbd_points
from lidra.data.dataset.tdfy.objaverse.utils import (
    blender2pytorch3d,
    read_depth_channel,
    get_cam_transform,
    get_relative_cam_transform,
)


def load_surface_points(dataset, uid, geometry):
    # Sample points from the mesh
    points, face_indices = trimesh.sample.sample_surface(
        geometry,
        dataset.SURFACE_SAMPLES,
    )
    # check if not enough positive points to sample
    if len(points) < dataset.n_queries:
        logger.warning(f"not enough points found {len(points)} < {dataset.n_queries}")
        return None

    # Load images and convert to MCC format:
    try:
        seq_dir = os.path.join(dataset.path, dataset.rendering_folder_name, uid)
        return __loading_mess(dataset, points, face_indices, geometry, seq_dir, uid)
    except:
        logger.opt(exception=True).warning(f"error loading the mess (uid={uid})")
        return None


def load_image(image_path, pts_mask=None):
    image = plt.imread(image_path)
    seen_rgb = torch.from_numpy(image.copy()) / 255.0
    if pts_mask is not None:
        seen_rgb = seen_rgb * pts_mask + torch.ones_like(seen_rgb) * (~pts_mask)
    seen_rgb = seen_rgb.permute(2, 0, 1).contiguous()
    seen_rgb = seen_rgb
    return seen_rgb


# TODO(Pierre) clean this
def __loading_mess(dataset, points, face_indices, geometry, seq_dir, uid):
    normals = geometry.face_normals[face_indices]
    points = torch.from_numpy(points)
    normals = torch.from_numpy(normals)
    new_points = points[:, [0, 2, 1]].clone().float()
    new_normals = normals[:, [0, 2, 1]].clone().float()

    new_points[:, 1] *= -1
    new_normals[:, 1] *= -1
    new_points *= dataset.scale

    rendering_indicis = sorted(
        [
            f.split("_rgb0001.jpg")[0]
            for f in os.listdir(seq_dir)
            if f.endswith("_rgb0001.jpg")
        ]
    )
    rendering_idx = random.choice(rendering_indicis)
    camera_pos = np.load(os.path.join(seq_dir, f"{rendering_idx}_pose.npy"))
    fov = np.load(os.path.join(seq_dir, f"{rendering_idx}_intrinsics.npy")).item()
    image_path = os.path.join(seq_dir, f"{rendering_idx}_rgb0001.jpg")
    depth = read_depth_channel(os.path.join(seq_dir, f"{rendering_idx}_depth0001.exr"))
    R, T = camera_pos[:3, :3], camera_pos[:3, -1]
    R_pytorch3d, T_pytorch3d = blender2pytorch3d(
        torch.from_numpy(R), torch.from_numpy(T) * dataset.scale
    )

    camera = FoVPerspectiveCameras(
        fov=fov, device=R_pytorch3d.device, R=R_pytorch3d[None], T=T_pytorch3d
    )
    depth_map = torch.from_numpy(depth.copy())
    mask = (depth_map < 1000).float()
    depth_map[depth_map > 1000] = 0

    seen_xyz, pts_mask = get_rgbd_points(
        112,
        112,
        camera,
        depth_map.unsqueeze(0).unsqueeze(0) * dataset.scale,
        mask.unsqueeze(0).unsqueeze(0),
    )

    pts_mask = (depth_map > 0).unsqueeze(-1)
    seen_rgb = load_image(image_path, pts_mask)

    # if self.args.obj_center_norm:
    if True:
        rel_cam_transform = get_cam_transform(R_pytorch3d[None], T_pytorch3d)[0]
        rel_cam_transform = get_relative_cam_transform(
            torch.eye(4).unsqueeze(0), rel_cam_transform.unsqueeze(0)
        )[0]
        cam_transform = pytorch3d.transforms.transform3d.Transform3d(
            matrix=rel_cam_transform
        )
        seen_xyz = cam_transform.transform_points(seen_xyz)
        new_points = cam_transform.transform_points(new_points)
        new_normals = pytorch3d.transforms.transform3d.Transform3d(
            matrix=get_cam_transform(R_pytorch3d[None], torch.zeros_like(T_pytorch3d))[
                0
            ]
        ).transform_points(new_normals)
        # normalize points to [-3, 3], 0 center
        min_box = new_points.min(0).values
        max_box = new_points.max(0).values
        max_side = (max_box - min_box).max()
        new_points -= (min_box + max_box) / 2
        new_points *= dataset.scale / max_side

        seen_xyz -= (min_box + max_box) / 2
        seen_xyz *= dataset.scale / max_side

    # if not self.args.rgbd_input:
    if True:
        # no depth condition
        seen_xyz = torch.zeros_like(seen_xyz)

    # retry if NaN (e.g. flat surface)
    if new_points.isnan().any() or seen_xyz.isnan().any():
        print(
            f"Sample has NaN in {seq_dir}"
        )  # Shouldn't trigger, but keeping as a safeguard
        return None

    return (
        (seen_xyz, seen_rgb, mask.unsqueeze(0)),  # location, color, mask
        (
            new_points,
            torch.ones_like(new_points) / 2.0,
            new_normals,
        ),  # location, color, normal
        None,
        ("Objaverse", uid, None),
    )
