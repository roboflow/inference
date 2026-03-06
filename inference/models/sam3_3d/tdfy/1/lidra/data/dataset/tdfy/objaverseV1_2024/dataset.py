from collections import namedtuple
import numpy as np
import os
import utils3d
import torch
from torch.utils.data import Dataset
from typing import Union
import matplotlib.pyplot as plt
from lidra.data.dataset.tdfy.objaverse.utils import (
    read_depth_channel,
    blender2pytorch3d,
)

from lidra.data.dataset.tdfy.img_and_mask_transforms import (
    compute_mask_bbox,
    crop_and_pad,
)

ObjaverseV1_2024SampleID = namedtuple("ObjaverseV1_2024SampleID", ["uuid", "frame_id"])


class ObjaverseV1_2024Dataset(Dataset):
    """Dataset for loading old Objaverse V1 dataset from 2024"""

    def __init__(
        self,
        data_dir: str,
        sample_frame: bool = False,  # randomly sample a frame or not
        load_voxels: bool = False,  # load voxels with key "voxels"
        tight_obj_boundary: bool = True,
        pad_obj_crop_ratio: float = 1.0,
        pad_ratio: float = 0.1,
    ):
        self.base_dir = data_dir
        self.all_uuids = os.listdir(os.path.join(self.base_dir, "renders_cond"))
        self.sample_frame = sample_frame
        self.load_voxels = load_voxels
        self.tight_obj_boundary = tight_obj_boundary
        self.pad_obj_crop_ratio = pad_obj_crop_ratio
        self.pad_ratio = pad_ratio

    def __len__(self) -> int:
        return len(self.all_uuids)

    def _load_img(self, image_dir, frame_id):
        """Read image, [H, W, 3]"""
        img_path = os.path.join(image_dir, frame_id + "_rgb0001.jpg")
        image = plt.imread(img_path)
        image = image / 255
        image = image.astype(np.float32)
        return torch.from_numpy(image)

    def _load_mask(self, image_dir, frame_id):
        """Mask read from depth image, [H, W]"""
        depth_path = os.path.join(image_dir, frame_id + "_depth0001.exr")
        depth = read_depth_channel(depth_path)
        depth_map = torch.from_numpy(depth.copy())
        # get mask by using finite depth
        mask = (depth_map < 1000).float().numpy()
        return torch.from_numpy(mask)

    def _load_voxels(self, uuid):
        voxel_path = os.path.join(self.base_dir, "voxels", f"{uuid}.ply")
        return torch.tensor(utils3d.io.read_ply(voxel_path)[0])

    def _load_pose(self, uuid, frame_id):
        pose_path = os.path.join(
            self.base_dir, "renders_cond", uuid, f"{frame_id}_pose.npy"
        )
        camera_pos = np.load(pose_path)
        R, T = camera_pos[:3, :3], camera_pos[:3, -1]
        R_pytorch3d, T_pytorch3d = blender2pytorch3d(
            torch.from_numpy(R), torch.from_numpy(T)
        )
        return R_pytorch3d, T_pytorch3d

    def __getitem__(self, idx: Union[int, ObjaverseV1_2024SampleID]) -> dict:
        if isinstance(idx, tuple):
            uuid = idx.uuid
            frame_id = idx.frame_id
            image_dir = os.path.join(self.base_dir, "renders_cond", uuid)
        else:
            uuid = self.all_uuids[idx]
            image_dir = os.path.join(self.base_dir, "renders_cond", uuid)
            if self.sample_frame:
                available_frame_ids = [
                    f[: -len("_rgb0001.jpg")]
                    for f in os.listdir(image_dir)
                    if f.endswith("_rgb0001.jpg")
                    and os.path.isfile(os.path.join(image_dir, f))
                ]
                frame_id = np.random.choice(available_frame_ids)
            else:
                frame_id = "0000"
        sample_uuid = ObjaverseV1_2024SampleID(uuid, frame_id)
        image = self._load_img(image_dir, frame_id)
        mask = self._load_mask(image_dir, frame_id)

        # remove background; actually might not matter here for ObjaverseV1 old
        image = image * mask[..., None]
        image = image.permute(2, 0, 1).contiguous()[:3]
        if self.tight_obj_boundary:
            bbox = compute_mask_bbox(mask, self.pad_obj_crop_ratio)
            image = crop_and_pad(image, bbox)
            mask = crop_and_pad(mask[None], bbox)[0]
        _, H, W = image.shape
        max_dim = max(H, W)  # Get the larger dimension
        extend_size = int(max_dim * self.pad_ratio)  # 10% extension on each side
        image = torch.nn.functional.pad(
            image,
            (extend_size, extend_size, extend_size, extend_size),
            mode="constant",
            value=0,
        )
        mask = torch.nn.functional.pad(
            mask[None],
            (extend_size, extend_size, extend_size, extend_size),
            mode="constant",
            value=0,
        )

        R_gt, T_gt = self._load_pose(uuid, frame_id)

        out = {
            "image": image,
            "mask": mask,
            "R_gt": R_gt,
            "T_gt": T_gt,
        }

        # We won't be using this for now; but adding it when we revise the code
        if self.load_voxels:
            out["voxels"] = self._load_voxels(uuid)

        return sample_uuid, out
