from collections import namedtuple
import numpy as np
import os
import json
import utils3d
import torch
from torch.utils.data import Dataset
from typing import Union
import matplotlib.pyplot as plt

from lidra.data.dataset.tdfy.img_and_mask_transforms import (
    load_rgb,
    split_rgba,
    crop_around_mask_with_padding,
)

R3SampleID = namedtuple("R3SampleID", ["uuid"])


class R3Dataset(Dataset):
    """Dataset for loading old Objaverse V1 dataset from 2024"""

    def __init__(
        self,
        data_dir: str,
        load_voxels: bool = False,  # load voxels with key "voxels"
        masked_image=None,
    ):
        self.base_dir = data_dir
        self.all_uuids = os.listdir(os.path.join(self.base_dir, "renders_cond"))
        self.load_voxels = load_voxels

    def __len__(self) -> int:
        return len(self.all_uuids)

    def _load_img_mask(self, image_dir):
        # todo, use trellis or rgba image reading
        img_path = os.path.join(image_dir, "rgba_001.png")
        rgba = load_rgb(img_path)
        return split_rgba(rgba)

    def _load_voxels(self, uuid):
        voxel_path = os.path.join(self.base_dir, "voxels", f"{uuid}.ply")
        return torch.tensor(utils3d.io.read_ply(voxel_path)[0])

    # TODO: @weiyaowang refactor with trellis dataset
    def _load_pose(self, uuid):
        pose_path = os.path.join(self.base_dir, "renders_cond", uuid, "transforms.json")
        if not os.path.exists(pose_path):
            R = torch.eye(3, dtype=torch.float32)
            T = torch.zeros([1, 3], dtype=torch.float32)
        else:
            with open(pose_path, "r") as file:
                transforms_data = json.load(file)
            frames_data = transforms_data["frames"][0]
            R = torch.tensor(frames_data["RR"], dtype=torch.float32, device="cpu")
            T = torch.tensor(
                frames_data["TT"], dtype=torch.float32, device="cpu"
            ).unsqueeze(0)
        return R, T

    def __getitem__(self, idx: Union[int, R3SampleID]) -> dict:
        if isinstance(idx, tuple):
            uuid = idx.uuid
            image_dir = os.path.join(self.base_dir, "renders_cond", uuid)
        else:
            uuid = self.all_uuids[idx]
            image_dir = os.path.join(self.base_dir, "renders_cond", uuid)
        sample_uuid = R3SampleID(uuid)
        image, mask = self._load_img_mask(image_dir)

        R_gt, T_gt = self._load_pose(uuid)

        out = {
            "image": image,
            "mask": mask[None],
            "R_gt": R_gt,
            "T_gt": T_gt,
        }

        # We won't be using this for now; but adding it when we revise the code
        if self.load_voxels:
            out["voxels"] = self._load_voxels(uuid)

        return sample_uuid, out
