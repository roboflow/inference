from typing import Union
import os
import torch
from torch.utils.data import Dataset

from lidra.data.dataset.tdfy.r3.dataset import R3Dataset, R3SampleID

from lidra.data.dataset.tdfy.trellis.dataset import (
    PerSubsetDataset as TrellisPerSubsetDataset,
)


# We can refactor this into a general Trellis-Anything dataset later
class AnythingDataset(Dataset):
    def __init__(
        self,
        dataset: R3Dataset,
        latent_loader_dataset: TrellisPerSubsetDataset,
    ):
        super().__init__()
        self.dataset = dataset
        self.latent_loader_dataset = latent_loader_dataset

    def get_row(self, sha256):
        metadata = self.latent_loader_dataset.metadata
        return metadata[metadata["sha256"] == sha256].iloc[0]

    def _load_latent(self, uuid: str):
        shape_latent = self.latent_loader_dataset._load_latent(uuid)
        return shape_latent

    def _load_pose(self, uuid: str):
        return self.latent_loader_dataset._load_pose(uuid, "rgba_001.png")

    def _load_pointmap(self, uuid: str, rgb_image: torch.Tensor):
        image_fname = self.get_row(uuid)["image_basename"]
        raw_pointmap = self.latent_loader_dataset._load_pointmap(
            uuid, rgb_image, image_fname
        )
        return raw_pointmap

    def __len__(self):
        # read from metadata loaded in the latent loader dataset
        return len(self.latent_loader_dataset)

    def __getitem__(self, idx: Union[int, R3SampleID]):
        if type(idx) == int:
            uuid = self.latent_loader_dataset.uids[idx]
            idx = R3SampleID(uuid)
        if isinstance(idx, tuple):
            # convert to ID format
            idx = R3SampleID(*idx)
        raw_uuid, raw_sample = self.dataset[idx]
        raw_rgb_image = raw_sample["image"]

        # # Apply transforms
        rgb_image = raw_sample["image"]
        mask_image = raw_sample["mask"]

        instance_latent = self._load_latent(raw_uuid.uuid)
        instance_pose = self._load_pose(raw_uuid.uuid)
        raw_pointmap = self._load_pointmap(raw_uuid.uuid, raw_rgb_image)
        image_dict = (
            self.latent_loader_dataset.preprocessor._process_image_mask_pointmap_mess(
                rgb_image, mask_image, raw_pointmap
            )
        )
        return raw_uuid, {
            **raw_sample,
            **instance_latent,
            **instance_pose,
            **image_dict,
        }
