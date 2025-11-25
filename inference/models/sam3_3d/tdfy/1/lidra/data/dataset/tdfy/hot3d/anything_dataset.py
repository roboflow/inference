from collections import namedtuple
from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset

from lidra.data.dataset.tdfy.hot3d.dataset import (
    HOT3DDataset,
    HOT3DDatasetSampleID,
)
from lidra.data.dataset.tdfy.img_and_mask_transforms import BoundingBoxError
from loguru import logger


HOT3DAnythingSampleID = namedtuple(
    "HOT3DAnythingSampleID", ["seq_id", "frame_id", "instance_id"]
)


class AnythingDataset(Dataset):
    def __init__(
        self,
        dataset: HOT3DDataset,
        latent_loader_dataset: Optional["TrellisPerSubsetDataset"] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.latent_loader_dataset = latent_loader_dataset

    def __len__(self):
        return len(self.dataset)

    def _load_instance_latent(self, inst):
        if self.latent_loader_dataset is None:
            return None

        # Get sha256 hash of mesh
        uid = self.dataset._get_uid(inst)

        # Get latent
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

    def _parse_idx(self, idx: Union[int, HOT3DAnythingSampleID]):
        # Upgrade tuple to HOT3DAnythingSampleID
        if isinstance(idx, tuple):
            if not len(idx) == 3:
                raise ValueError(f"Expected 3-tuple, got {len(idx)}")
            requested_uuid = HOT3DAnythingSampleID(*idx)
            raw_uuid = HOT3DDatasetSampleID(**requested_uuid._asdict())
        else:
            requested_uuid = None
            raw_uuid = idx
        return requested_uuid, raw_uuid

    def __getitem__(self, idx: Union[int, HOT3DAnythingSampleID]):
        requested_uuid, raw_uuid = self._parse_idx(idx)

        # Draw sample from HOT3DDataset
        raw_uuid, raw_sample = self.dataset[raw_uuid]
        sample_uuid = HOT3DAnythingSampleID(
            seq_id=raw_uuid.seq_id,
            frame_id=raw_uuid.frame_id,
            instance_id=raw_uuid.instance_id,
        )
        assert (
            sample_uuid == requested_uuid or requested_uuid is None
        ), f"Sample UUID mismatch: {sample_uuid} != {requested_uuid}"

        # Apply transforms
        rgb_image = raw_sample["image"]
        mask_image = raw_sample["mask"]
        processed_image, processed_mask = self._preprocess_image_and_mask(
            rgb_image, mask_image
        )

        processed_image = self.latent_loader_dataset.img_transform(processed_image)
        processed_mask = self.latent_loader_dataset.mask_transform(processed_mask)
        rgb_image = self.latent_loader_dataset.img_transform(rgb_image)
        mask_image = self.latent_loader_dataset.mask_transform(mask_image)

        # Load instance latent
        instance_latent = self._load_instance_latent(sample_uuid.instance_id)

        # Override with dummy pointmap for now, until dataset implements it (then delete)
        raw_sample["pointmap_dict"] = (
            self.latent_loader_dataset._dummy_pointmap_moments()
        )

        return sample_uuid, {
            "image": processed_image,
            "mask": processed_mask,
            "rgb_image": rgb_image,
            "rgb_image_mask": mask_image,
            **instance_latent,
            **raw_sample["instance_pose"],
            **raw_sample["pointmap_dict"],
        }
