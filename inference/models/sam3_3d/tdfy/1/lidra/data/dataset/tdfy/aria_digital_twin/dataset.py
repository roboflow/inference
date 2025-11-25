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

from lidra.data.dataset.tdfy.artists_3d.dataset import get_subdirs
from lidra.data.dataset.tdfy.point_cloud import get_rgbd_points
from lidra.data.dataset.tdfy.trellis.dataset import PerSubsetDataset
from lidra.data.dataset.tdfy.trellis.pose_loader import (
    convert_to_decoupled_instance_pose,
)


ADTDatasetSampleID = namedtuple(
    "ADTDatasetSampleID", ["seq_id", "frame_id", "instance_id"]
)

OPENCV_TO_P3D = np.array(
    [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)
OPENCV_TO_P3D_TRANSFORM3D = Transform3d().rotate(torch.tensor(OPENCV_TO_P3D))


class ADTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        use_synth_img: bool = False,
        reload_cache: bool = False,
    ):
        self.path = path
        self.data_dir = os.path.join(self.path, "adt_tdfy_evaluation")
        self.use_synth_img = use_synth_img
        self.reload_cache = reload_cache

        # Load metadata
        self.metadata = pd.read_csv(os.path.join(self.path, "metadata.csv"))
        with open(os.path.join(path, "instance2mesh_mapping.json"), "r") as f:
            self.inst_2_mesh = json.load(f)
        self.inst_name_2_uid = self.metadata.set_index(
            self.metadata["file_identifier"].apply(lambda x: x.split("/")[-1])
        )["sha256"].to_dict()

        # Load/build seq-frame-instance map and indexing
        self.idx_map = self._get_idx_2_seq_frame_inst()

    def _get_idx_2_seq_frame_inst(self):
        idx_path = os.path.join(self.data_dir, "idx_2_seq_frame_inst_filter.json")

        if self.reload_cache or not (
            os.path.exists(idx_path) and os.path.exists(idx_path)
        ):
            idx_map = {}
            count = 0
            seq_dirs = sorted(get_subdirs(self.data_dir))

            for seq_dir in seq_dirs:
                abs_seq_dir = os.path.join(self.data_dir, seq_dir)
                frame_dirs = sorted(get_subdirs(abs_seq_dir))

                for frame_dir in frame_dirs:
                    abs_frame_dir = os.path.join(abs_seq_dir, frame_dir)
                    inst_dirs = sorted(get_subdirs(abs_frame_dir))

                    for inst_dir in inst_dirs:
                        idx_map[count] = (seq_dir, frame_dir, inst_dir)
                        count += 1

            # Cache idx_map
            with open(idx_path, "w") as f:
                json.dump(idx_map, f)

        else:
            with open(idx_path, "r") as f:
                _idx_map = json.load(f)
            idx_map = {int(k): v for k, v in _idx_map.items()}

        return idx_map

    def _get_uid(self, inst_name):
        # Map to preprocessed mesh, in case of duplicates
        dedup_inst_name = self.inst_2_mesh[inst_name]
        # Get sha256 hash of mesh
        uid = self.inst_name_2_uid[dedup_inst_name]
        return uid

    def _load_img(self, seq, frame):
        frame_dir = os.path.join(self.data_dir, seq, frame)

        if self.use_synth_img:
            img_path = os.path.join(frame_dir, "synth_img.png")
        else:
            img_path = os.path.join(frame_dir, "aria_img.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    def _load_mask(self, seq, frame, inst):
        mask_path = os.path.join(self.data_dir, seq, frame, inst, "mask.png")
        mask = cv2.imread(mask_path)

        mask = torch.from_numpy(mask)[:, :, 0].float().unsqueeze(0) / 255.0
        return mask

    def _load_camera(self, seq, height, width):
        intrinsics_path = os.path.join(self.data_dir, seq, "intrinsics.txt")
        K = np.loadtxt(intrinsics_path)

        camera = PerspectiveCameras(
            focal_length=((K[0, 0], K[1, 1]),),
            principal_point=((K[0, 2], K[1, 2]),),
            in_ndc=False,
            image_size=((height, width),),
        )

        return camera

    def _get_depthmap(self, seq, frame):
        depth_path = os.path.join(self.data_dir, seq, frame, "depth.png")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth / 1000  # Convert mm to m
        return torch.tensor(depth).unsqueeze(0)  # Add channel dim back

    def _get_pointmap(self, rgb_image, seq, frame, camera):
        pts_3d, _ = get_rgbd_points(
            camera.image_size[0, 0].int(),
            camera.image_size[0, 1].int(),
            camera,
            self._get_depthmap(seq, frame).unsqueeze(0),  # Add batch dim
        )
        return {
            "pointmap": pts_3d,
            "pointmap_colors": rgb_image,
        }

    def _load_pose(self, seq, frame, inst_name):
        pose_path = os.path.join(
            self.data_dir, seq, frame, inst_name, "transform_cam_obj.txt"
        )
        T_cam_obj = np.loadtxt(pose_path, dtype=np.float32)
        transform3d_cam_obj = Transform3d(matrix=torch.tensor(T_cam_obj).T)

        # Convert camera conventions from OpenCV to Pytorch3D
        transform3d_cam_obj = transform3d_cam_obj.compose(OPENCV_TO_P3D_TRANSFORM3D)

        return convert_to_decoupled_instance_pose(transform3d_cam_obj)

    def __len__(self) -> int:
        return len(self.idx_map)

    def __getitem__(self, idx: Union[int, ADTDatasetSampleID]) -> dict:
        if isinstance(idx, tuple):
            if not len(idx) == 3:
                raise ValueError(f"Expected 3-tuple, got {len(idx)}")
            sample_uuid = ADTDatasetSampleID(*idx)

            seq = idx.seq_id
            frame = idx.frame_id
            inst = idx.instance_id
        else:
            seq, frame, inst = self.idx_map[idx]

            sample_uuid = ADTDatasetSampleID(
                seq_id=seq, frame_id=frame, instance_id=inst
            )

        # Load image and mask
        img = self._load_img(seq, frame)
        mask = self._load_mask(seq, frame, inst)

        # Load pose
        height, width = img.shape[1:3]
        camera = self._load_camera(seq, height, width)
        instance_pose = self._load_pose(seq, frame, inst)

        # Point map
        pointmap_dict = self._get_pointmap(img, seq, frame, camera)

        return sample_uuid, {
            "image": img,
            "mask": mask,
            "camera": camera,
            "instance_pose": instance_pose,
            "pointmap_dict": pointmap_dict,
        }
