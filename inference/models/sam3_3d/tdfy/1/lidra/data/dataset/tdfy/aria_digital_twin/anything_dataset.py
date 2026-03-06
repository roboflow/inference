from collections import namedtuple
from typing import List, Dict, Optional, Union

from pytorch3d.transforms import quaternion_to_matrix
import torch
from torch.utils.data import Dataset

from lidra.data.dataset.tdfy.aria_digital_twin.dataset import (
    ADTDataset,
    ADTDatasetSampleID,
)
from lidra.data.dataset.tdfy.img_and_mask_transforms import BoundingBoxError
from lidra.data.dataset.tdfy.transforms_3d import (
    compose_transform,
    decompose_transform,
)
from lidra.data.dataset.tdfy.trellis.pose_loader import (
    convert_to_decoupled_instance_pose,
)
from loguru import logger


ADTAnythingSampleID = namedtuple(
    "ADTAnythingSampleID", ["seq_id", "frame_id", "instance_id"]
)

SWAP_YZ_AXIS = torch.tensor(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=torch.float32
)


class AnythingDataset(Dataset):
    def __init__(
        self,
        dataset: ADTDataset,
        latent_loader_dataset: Optional["TrellisPerSubsetDataset"] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.latent_loader_dataset = latent_loader_dataset

    def __len__(self):
        return len(self.dataset)

    def _load_mesh_to_latent_transform(self, inst_name: str):
        uid = self.dataset._get_uid(inst_name)
        return self.latent_loader_dataset._load_mesh_to_latent_transform(uid)

    def _apply_mesh_to_latent_transform(
        self, instance_pose, transform3d_mesh_to_latent
    ):
        # Instance pose in ADT's metric scale
        transform3d_cam_obj = compose_transform(
            instance_pose["instance_scale_l2c"],
            quaternion_to_matrix(instance_pose["instance_quaternion_l2c"]),
            instance_pose["instance_position_l2c"],
        )

        # Transform to go from ADT metric to Trellis preprocess
        decomposed_transform3d_mesh_to_latent = decompose_transform(
            transform3d_mesh_to_latent
        )
        transform3d_mesh_to_latent = compose_transform(
            decomposed_transform3d_mesh_to_latent.scale,
            SWAP_YZ_AXIS,  # ADT has additional YZ axis swap
            decomposed_transform3d_mesh_to_latent.translation,
        )
        transform3d_latent_to_mesh = transform3d_mesh_to_latent.inverse()

        # Transform instance pose to Trellis preprocess scale and decompose
        transform3D_inst_pose = transform3d_latent_to_mesh.compose(transform3d_cam_obj)
        return convert_to_decoupled_instance_pose(transform3D_inst_pose)

    def _load_instance_latent(self, inst_name: str):
        if self.latent_loader_dataset is None:
            return None

        # Get latent
        uid = self.dataset._get_uid(inst_name)
        shape_latent = self.latent_loader_dataset._load_latent(uid)
        return shape_latent

    def _preprocess_image_and_mask(self, rgb_image, mask_image):
        try:
            rgb_image, mask_image = (
                self.latent_loader_dataset._preprocess_image_and_mask_inference(
                    rgb_image, mask_image
                )
            )
        except BoundingBoxError as e:
            logger.warning(f"Error: {e}, no changes to rgb_image and mask_image")

        return rgb_image, mask_image

    def __getitem__(self, idx: Union[int, ADTAnythingSampleID]):
        # Upgrade tuple to ADTAnythingSampleID
        if isinstance(idx, tuple):
            if not len(idx) == 3:
                raise ValueError(f"Expected 3-tuple, got {len(idx)}")
            requested_uuid = ADTAnythingSampleID(*idx)
            raw_uuid = ADTDatasetSampleID(**requested_uuid._asdict())
        else:
            requested_uuid = None
            raw_uuid = idx

        # Draw sample from ADTDataset
        raw_uuid, raw_sample = self.dataset[raw_uuid]
        sample_uuid = ADTAnythingSampleID(
            seq_id=raw_uuid.seq_id,
            frame_id=raw_uuid.frame_id,
            instance_id=raw_uuid.instance_id,
        )
        assert (
            sample_uuid == requested_uuid or requested_uuid is None
        ), f"Sample UUID mismatch: {sample_uuid} != {requested_uuid}"

        # Apply preprocess transform to ADT metric instance pose
        transform3d_mesh_to_latent = self._load_mesh_to_latent_transform(
            raw_uuid.instance_id
        )
        instance_pose = self._apply_mesh_to_latent_transform(
            raw_sample["instance_pose"], transform3d_mesh_to_latent
        )

        # Load instance latent
        instance_latent = self._load_instance_latent(sample_uuid.instance_id)

        # Load pointmap
        rgb_image = raw_sample["image"]
        mask_image = raw_sample["mask"]
        raw_pointmap = raw_sample["pointmap_dict"]["pointmap"].float()
        pointmap_dict = self.latent_loader_dataset._prepare_pointmap(
            raw_pointmap,
            return_pointmap=self.latent_loader_dataset.return_pointmap,
            mask=mask_image,
        )
        # Convert pointmap from (H, W, 3) to (3, H, W) for preprocessor
        raw_pointmap_for_preprocessor = (
            raw_pointmap.permute(2, 0, 1) if raw_pointmap is not None else None
        )
        image_dict = (
            self.latent_loader_dataset.preprocessor._process_image_mask_pointmap_mess(
                rgb_image,
                mask_image,
                raw_pointmap_for_preprocessor,
            )
        )
        return sample_uuid, {
            # **raw_sample,
            **image_dict,
            **instance_latent,
            **instance_pose,
            **pointmap_dict,
        }
