import random

import numpy as np
import torch

from lidra.data.dataset.tdfy.base_scene_dataset import BaseSceneDataset
from lidra.data.dataset.tdfy.img_processing import (
    preprocess_img,
    pad_to_square_centered,
    crop_img_to_obj,
)
from lidra.data.dataset.tdfy.point_cloud import (
    normalize_objects_in_cam,
    get_rgbd_points,
)


class ObjInSceneSingleSceneDataset(BaseSceneDataset):
    def __init__(
        self,
        masked_img: bool,
        obj_center_norm: bool,
        add_context_to_bbox: float,
        is_train: bool,
        preload_gt_pts=True,
        n_gt_pts=30000,
        frustum_visible=True,
        padding_value=0,  # r3 set 1, cadestate set 0
        black_bg=True,  # r3 set False, cadestate set True
        normalize_img=False,
    ):

        self.masked_img = masked_img
        self.obj_center_norm = obj_center_norm
        self.add_context_to_bbox = add_context_to_bbox
        self.padding_value = padding_value
        self.black_bg = black_bg

        super().__init__(
            is_train=is_train,
            preload_gt_pts=preload_gt_pts,
            n_gt_pts=n_gt_pts,
            frustum_visible=frustum_visible,
        )

        self.color_cams = None
        self.depth_cams = None
        self._prepare_cameras()
        self.name = "BaseObjectsInScene"
        self.normalize_img = normalize_img

    # convert camera to pytorch3d cameras
    def _prepare_cameras(self):
        raise NotImplementedError

    # retrieve mask of an object given selected object info
    def _get_obj_mask(self, idx, obj_info):
        raise NotImplementedError

    def _get_depth_cam(self, idx):
        return self.depth_cams[idx]

    def _get_color_cam(self, idx):
        return self.color_cams[idx]

    def _load_obj_info(self, idx):
        raise NotImplementedError

    def __getitem__(self, j):
        idx = self.idx_list[j]
        img = self.load_rgb(idx) / 255.0
        depth = torch.from_numpy(self.load_depth(idx)).unsqueeze(0).unsqueeze(0).float()

        obj_info = self._load_obj_info(idx)
        mask, selected_obj = self._get_obj_mask(idx, obj_info)
        mask = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)

        # pad depth to square (112, 112, 3)
        seen_xyz, _ = get_rgbd_points(
            112,
            112,
            self._get_depth_cam(idx),
            pad_to_square_centered(depth, self.padding_value),
            pad_to_square_centered(mask, self.padding_value),
        )
        seen_rgb = torch.from_numpy(img)
        seen_rgb = seen_rgb.permute(2, 0, 1).float()
        full_rgb = torch.from_numpy(np.stack(obj_info[selected_obj]["colors"])).float()
        full_points = torch.from_numpy(
            np.stack(obj_info[selected_obj]["points"])
        ).float()

        # process full_points to match self.n_gt_pts
        if full_points.shape[0] != self.n_gt_pts:
            indices = random.choices(list(range(full_points.shape[0])), k=self.n_gt_pts)
            full_rgb = full_rgb[indices]
            full_points = full_points[indices]

        if self.obj_center_norm:
            cam_transform = self._get_color_cam(idx).get_world_to_view_transform()
            seen_xyz = cam_transform.transform_points(seen_xyz)
            full_points = normalize_objects_in_cam(full_points, cam_transform)

            # no depth condition
            seen_xyz = torch.zeros_like(seen_xyz)

        if self.masked_img:
            if self.black_bg:
                seen_rgb *= mask[0]
            else:  # use white background to be the same as objaverse
                seen_rgb = seen_rgb * mask[0] + torch.ones_like(seen_rgb) * (
                    1 - mask[0]
                )

        if self.add_context_to_bbox is not None:
            left, right, top, bot = crop_img_to_obj(
                mask[0, 0], self.add_context_to_bbox
            )
            if left is not None:
                seen_rgb = seen_rgb[:, top:bot, left:right]
                mask = mask[:, :, top:bot, left:right]

        # pad and resize for batch collating:
        seen_rgb, mask = preprocess_img(
            seen_rgb.unsqueeze(0),
            mask,
            img_target_shape=1024,
            mask_target_shape=1024,
            normalize=self.normalize_img,
        )
        return (
            (seen_xyz, seen_rgb[0], mask[0]),
            (full_points, full_rgb),
            None,
            f"{self.name}_{self.scene_name}_{idx}_{selected_obj}",
        )
