import copy

import numpy as np
import torch

from lidra.data.dataset.tdfy.hypersim_dataset import get_example_std, random_crop
from lidra.data.dataset.tdfy.point_cloud import (
    get_img_coords_grid,
    get_point_map,
    get_points_in_front_of_camera,
    get_points_in_frustum,
    sample_pcd,
)


class BaseSceneDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        is_train: bool,
        preload_gt_pts=True,
        n_gt_pts=30000,
        frustum_visible=True,
    ):
        self.is_train = is_train

        # Remap idx list if necessary
        self.idx_list = self._get_idx_list()

        # Load GT (one for scene)
        self.preload_gt_pts = preload_gt_pts
        if self.preload_gt_pts:
            self.gt_pts = self.load_gt_points(voxel_size=None)
        self.n_gt_pts = 30000
        self.frustum_visible = frustum_visible

    def _get_scene_name(self):
        raise NotImplementedError

    def _get_idx_list(self):
        raise NotImplementedError

    def load_intrinsics(self, idx):
        raise NotImplementedError

    def load_extrinsics(self, idx):
        raise NotImplementedError

    def load_rgb(self, idx):
        raise NotImplementedError

    def load_depth(self, idx):
        raise NotImplementedError

    def load_gt_points(self):
        raise NotImplementedError

    def get_view_i_of_gt_pts(self, idx, n_gt_pts, H, W):
        if self.preload_gt_pts:
            gt_pts = copy.deepcopy(self.gt_pts)
        else:
            gt_pts = self.load_gt_points(voxel_size=None)

        # Load camera extrinsics (4x4)
        extrinsics_c2w = self.load_extrinsics(idx)
        extrinsics_w2c = np.linalg.inv(extrinsics_c2w)

        # Transform to this frame index's camera's perspective
        gt_pcd_i = gt_pts.transform(extrinsics_w2c)

        # Restrict to points in front of camera (nonnegative depth)
        gt_pcd_i = get_points_in_front_of_camera(gt_pcd_i)

        # Restrict to camera frustum
        if self.frustum_visible:
            intrinsics = self.load_intrinsics(idx)
            gt_pcd_i = get_points_in_frustum(gt_pcd_i, intrinsics, H, W)

        # TODO: Don't see through walls

        # Random sample n_gt_pts from GT point clouds
        gt_xyz, gt_rgb = sample_pcd(gt_pcd_i, n_gt_pts)
        gt_xyz = torch.tensor(gt_xyz, dtype=torch.float32)
        gt_rgb = torch.tensor(gt_rgb, dtype=torch.float32)

        return gt_xyz, gt_rgb

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, j):
        idx = self.idx_list[j]

        # Load RGB image
        img = self.load_rgb(idx) / 255
        H, W, _ = img.shape

        # Convert RGB + depth pixels to point map
        depth = self.load_depth(idx)
        intrinsics = self.load_intrinsics(idx)
        xyz = get_point_map(depth, intrinsics, H, W)

        # Random crop during train, square crop for eval
        seen_xyz = torch.tensor(xyz, dtype=torch.float32)
        seen_rgb = torch.tensor(img, dtype=torch.float32)
        seen_xyz, seen_rgb = random_crop(seen_xyz, seen_rgb, is_train=self.is_train)
        seen_rgb = seen_rgb.permute(2, 0, 1)

        # Load GT point cloud and transform to camera's perspective
        gt_xyz, gt_rgb = self.get_view_i_of_gt_pts(idx, self.n_gt_pts, H, W)

        # Normalize point clouds
        xyz_std = get_example_std(seen_xyz)
        seen_xyz = seen_xyz / xyz_std
        gt_xyz = gt_xyz / xyz_std
        # mesh_points = mesh_points / example_std

        seen_data = (seen_xyz, seen_rgb)
        gt_data = (gt_xyz, gt_rgb)
        mesh_points = torch.zeros((1,))  # Not used during training, so dummy

        return seen_data, gt_data, mesh_points, self._get_scene_name()
