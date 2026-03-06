import os
from typing import Union
import torch
from torch.utils.data import Dataset

from lidra.data.dataset.tdfy.objaverseV1_2024.dataset import (
    ObjaverseV1_2024Dataset,
    ObjaverseV1_2024SampleID,
)
from lidra.data.dataset.tdfy.trellis.dataset_deprecated import (
    PerSubsetDataset as TrellisPerSubsetDataset,
)

from ..img_and_mask_transforms import crop_around_mask_with_padding


class AnythingDataset(Dataset):
    def __init__(
        self,
        dataset: ObjaverseV1_2024Dataset,
        latent_loader_dataset: TrellisPerSubsetDataset,
    ):
        super().__init__()
        self.dataset = dataset
        self.latent_loader_dataset = latent_loader_dataset

    def __len__(self):
        # read from metadata loaded in the latent loader dataset
        return len(self.latent_loader_dataset)

    def _load_latent(self, uuid: str):
        shape_latent = self.latent_loader_dataset._load_latent(uuid)
        return shape_latent

    def _load_pose(self, uuid: str, frame_id: str):
        pose_path = os.path.join(
            self.latent_loader_dataset.path,
            "renders_cond",
            uuid,
            f"{frame_id}_pose.npy",
        )
        return self.latent_loader_dataset.pose_loader(pose_path)

    def _preprocess_image_and_mask(self, rgb_image, mask_image):
        rgb_image, mask_image = (
            self.latent_loader_dataset._preprocess_image_and_mask_inference(
                rgb_image, mask_image
            )
        )

        return rgb_image, mask_image

    def __getitem__(self, idx: Union[int, ObjaverseV1_2024SampleID]):
        if type(idx) == int:
            uuid = self.latent_loader_dataset.uids[idx]
            # use fixed frame_id for eval
            frame_id = "0000"
            idx = ObjaverseV1_2024SampleID(uuid, frame_id)
        if isinstance(idx, tuple):
            # convert to ID format
            idx = ObjaverseV1_2024SampleID(*idx)
        raw_uuid, raw_sample = self.dataset[idx]

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
        pointmap_dict = self.latent_loader_dataset._dummy_pointmap_moments()

        instance_latent = self._load_latent(raw_uuid.uuid)
        instance_pose = self._load_pose(raw_uuid.uuid, raw_uuid.frame_id)
        raw_sample["image"] = processed_image
        raw_sample["mask"] = processed_mask
        raw_sample["rgb_image"] = rgb_image
        raw_sample["rgb_image_mask"] = mask_image
        pointmap_dict["pointmap_scale"] = pointmap_dict["pointmap_scale"][0]
        pointmap_dict["pointmap_shift"] = pointmap_dict["pointmap_shift"][0]
        return raw_uuid, {
            **pointmap_dict,
            **raw_sample,
            **instance_latent,
            **instance_pose,
        }
