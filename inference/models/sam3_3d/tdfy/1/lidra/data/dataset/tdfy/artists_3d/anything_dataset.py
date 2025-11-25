from collections import namedtuple
from typing import List, Dict, Optional, Union
import torch
from torch.utils.data import Dataset

from lidra.data.dataset.tdfy.artists_3d.dataset import (
    Artist3DDataset,
    Artist3DDatasetSampleID,
)
from lidra.data.dataset.tdfy.img_and_mask_transforms import BoundingBoxError
from loguru import logger


Artist3DAnythingSampleID = namedtuple(
    "Artist3DAnythingSampleID", ["artist", "img_obj", "version"]
)


class AnythingDataset(Dataset):
    def __init__(
        self,
        dataset: Artist3DDataset,
        latent_loader_dataset: Optional["TrellisPerSubsetDataset"] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.latent_loader_dataset = latent_loader_dataset

    def __len__(self):
        return len(self.dataset)

    def _load_instance_latent(self, artist: str, img_obj: str, version: int):
        if self.latent_loader_dataset is None:
            return None

        # Get sha256 hash of mesh
        uid = self.dataset._get_uid(artist, img_obj, version)

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

    def _parse_idx(self, idx: Union[int, Artist3DAnythingSampleID]):
        # Upgrade tuple to Artist3DAnythingSampleID
        if isinstance(idx, tuple):
            if not len(idx) == 3:
                raise ValueError(f"Expected 3-tuple, got {len(idx)}")
            requested_uuid = Artist3DAnythingSampleID(*idx)
            raw_uuid = Artist3DDatasetSampleID(**requested_uuid._asdict())
        else:
            requested_uuid = None
            raw_uuid = idx
        return requested_uuid, raw_uuid

    def _load_pointmap(self):
        return self.latent_loader_dataset._dummy_pointmap_moments()

    def __getitem__(self, idx: Union[int, Artist3DAnythingSampleID]):
        requested_uuid, raw_uuid = self._parse_idx(idx)

        # Draw sample from Artist3DDataset
        raw_uuid, raw_sample = self.dataset[raw_uuid]
        sample_uuid = Artist3DAnythingSampleID(
            artist=raw_uuid.artist,
            img_obj=raw_uuid.img_obj,
            version=raw_uuid.version,
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
        instance_latent = self._load_instance_latent(
            sample_uuid.artist, sample_uuid.img_obj, sample_uuid.version
        )
        # no pose, set to a default value of identity rotation and zero translation
        instance_pose = self.latent_loader_dataset.pose_loader(None)
        pointmap_dict = self._load_pointmap()
        return sample_uuid, {
            "image": processed_image,
            "mask": processed_mask,
            "rgb_image": rgb_image,
            "rgb_image_mask": mask_image,
            **pointmap_dict,
            **instance_latent,
            **instance_pose,
        }
