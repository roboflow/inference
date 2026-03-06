import json
import os
from collections import namedtuple
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms import Transform3d
from scipy.spatial.transform import Rotation as R

from lidra.data.dataset.tdfy.aria_digital_twin.dataset import OPENCV_TO_P3D_TRANSFORM3D
from lidra.data.dataset.tdfy.trellis.pose_loader import (
    convert_to_decoupled_instance_pose,
)


HOT3DDatasetSampleID = namedtuple(
    "HOT3DDatasetSampleID", ["seq_id", "frame_id", "instance_id"]
)

ROTATE_90_CW = np.array(
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
)


class HOT3DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        # vis_sampled_only: bool = True,
        # handheld_only: bool = False,
    ):
        self.path = path
        self.eval_dir = os.path.join(self.path, "eval_tdfy")
        self.data_dir = os.path.join(self.eval_dir, "data")
        self.latent_dir = os.path.join(self.path, "trellis_preprocess")

        # TODO: We can include masks for other objects visible in scene
        # self.vis_sampled_only = vis_sampled_only

        # TODO: We can sample objects based on translational movement (likely handheld)
        # self.handheld_only = handheld_only

        # Load metadata
        sampling_metadata_path = os.path.join(self.eval_dir, "sampled_obj_frames.csv")
        self.sampling_metadata = pd.read_csv(sampling_metadata_path)

        latent_metadata_path = os.path.join(self.latent_dir, "metadata.csv")
        self.latent_metadata = pd.read_csv(latent_metadata_path)

    def _get_uid(self, inst):
        file_identifier = f"obj_{int(inst):06d}"
        sha256 = self.latent_metadata.loc[
            self.latent_metadata["file_identifier"] == file_identifier, "sha256"
        ].values

        assert (
            len(sha256) == 1
        ), f"Did not get single unique sha256 from {file_identifier}"
        return sha256[0]

    def _load_img(self, seq, frame):
        img_path = os.path.join(self.data_dir, seq, frame, "image.jpg")
        img = cv2.imread(img_path)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    def _load_mask(self, seq, frame, inst):
        mask_path = os.path.join(self.data_dir, seq, frame, f"mask_{int(inst):02d}.jpg")
        mask = cv2.imread(mask_path)

        mask = torch.from_numpy(mask)[:, :, 0].float().unsqueeze(0) / 255.0
        return mask

    def _load_camera(self, height, width):
        intrinsics_path = os.path.join(self.data_dir, "intrinsics.txt")
        K = np.loadtxt(intrinsics_path)

        camera = PerspectiveCameras(
            focal_length=((K[0, 0], K[1, 1]),),
            principal_point=((K[0, 2], K[1, 2]),),
            in_ndc=False,
            image_size=((height, width),),
        )

        return camera

    def _get_pointmap(self, rgb_image, seq, frame, camera):
        # TODO: load something from vrs semi-dense point cloud
        return NotImplementedError

    def _transform_dict_to_RT(self, transform_dict):
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = R.from_quat(
            transform_dict["quaternion_wxyz"], scalar_first=True
        ).as_matrix()
        RT[:3, 3] = transform_dict["translation_xyz"]
        return RT

    def _load_and_compose_transform_dicts(self, seq, frame, inst_name):
        clip_frame_dir = os.path.join(self.data_dir, seq, frame)

        T_world_from_camera_file = os.path.join(
            clip_frame_dir, "T_world_from_camera.json"
        )
        with open(T_world_from_camera_file, "r") as f:
            T_w_from_c = json.load(f)
        RT_w_from_c = self._transform_dict_to_RT(T_w_from_c)

        T_world_from_object_file = os.path.join(
            clip_frame_dir, f"T_world_from_object_{int(inst_name):02d}.json"
        )
        with open(T_world_from_object_file, "r") as f:
            T_w_from_o = json.load(f)
        RT_w_from_o = self._transform_dict_to_RT(T_w_from_o)

        # Get to camera from object, rotate 90 degrees because of Aria cam
        RT_c_from_o = ROTATE_90_CW @ np.linalg.inv(RT_w_from_c) @ RT_w_from_o
        return RT_c_from_o

    def _load_pose(self, seq, frame, inst_name):
        T_cam_obj = self._load_and_compose_transform_dicts(seq, frame, inst_name)
        transform3d_cam_obj = Transform3d(matrix=torch.tensor(T_cam_obj).T)

        # Convert camera conventions from OpenCV to Pytorch3D
        transform3d_cam_obj = transform3d_cam_obj.compose(OPENCV_TO_P3D_TRANSFORM3D)

        return convert_to_decoupled_instance_pose(transform3d_cam_obj)

    def __len__(self) -> int:
        return len(self.sampling_metadata)

    def __getitem__(self, idx: Union[int, HOT3DDatasetSampleID]) -> dict:
        if isinstance(idx, tuple):
            if not len(idx) == 3:
                raise ValueError(f"Expected 3-tuple, got {len(idx)}")
            sample_uuid = HOT3DDatasetSampleID(*idx)

            seq = idx.seq_id
            frame = idx.frame_id
            inst = idx.instance_id
        else:
            row = self.sampling_metadata.iloc[idx]
            seq = str(int(row["clip_id"]))
            frame = str(int(row["frame_id"]))
            inst = str(int(row["obj_id"]))

            sample_uuid = HOT3DDatasetSampleID(
                seq_id=seq, frame_id=frame, instance_id=inst
            )

        # Load image and mask
        img = self._load_img(seq, frame)
        mask = self._load_mask(seq, frame, inst)

        # Load pose
        height, width = img.shape[1:3]
        camera = self._load_camera(height, width)
        instance_pose = self._load_pose(seq, frame, inst)

        # Point map (TODO: Need to add this, perhaps from vrs)
        # pointmap_dict = self._get_pointmap(img, seq, frame, camera)
        pointmap_dict = {}

        return sample_uuid, {
            "image": img,
            "mask": mask,
            "camera": camera,
            "instance_pose": instance_pose,
            "pointmap_dict": pointmap_dict,
        }
