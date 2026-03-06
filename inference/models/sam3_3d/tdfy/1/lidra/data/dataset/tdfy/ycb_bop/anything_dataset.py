import json
import os

import torch
from typing import Union
from torch.utils.data import Dataset

from lidra.data.dataset.tdfy.trellis.dataset import (
    PerSubsetDataset as TrellisPerSubsetDataset,
    PerSubsetSampleID,
    load_rgb,
)
from loguru import logger


# a customized wrapper around TrellisPerSubsetDataset for YCB
class AnythingDataset(Dataset):
    def __init__(
        self,
        eval_id_json: str,
        latent_loader_dataset: TrellisPerSubsetDataset,
        return_pointmap: bool = True,
    ):
        super().__init__()
        self.latent_loader_dataset = latent_loader_dataset
        self.eval_ids = json.load(open(eval_id_json))
        self.return_pointmap = return_pointmap

    def __len__(self):
        return len(self.eval_ids)

    def _load_pointmap(self, idx, rgb_image):
        fname = idx.image_fname
        pts_3d = torch.load(
            os.path.join(
                self.latent_loader_dataset.path, "pointmaps", fname[:-3] + "pt"
            )
        )

        raw_pointmap = pts_3d.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        if not self.return_pointmap:
            raw_pointmap = None
        return raw_pointmap

    def _load_rgb_image(self, sample_uuid):
        uid = sample_uuid.sha256
        image_fname = sample_uuid.image_fname

        img_path = os.path.join(
            self.latent_loader_dataset._get_cond_image_dir(uid), image_fname
        )
        rgba_image = load_rgb(img_path)
        rgb_image = rgba_image[:3]
        rgb_image_mask = self.latent_loader_dataset._read_mask(rgba_image)
        return rgb_image, rgb_image_mask

    def __getitem__(self, idx: Union[int, PerSubsetSampleID]):
        if type(idx) == int:
            sha256, fname = self.eval_ids[idx]
            idx = PerSubsetSampleID(sha256, fname)
        if isinstance(idx, tuple):
            # convert to ID format
            idx = PerSubsetSampleID(*idx)
        raw_items = self.latent_loader_dataset[idx][1]
        rgb_image = raw_items["rgb_image"]
        rgb_image_mask = raw_items.get("rgb_image_mask", raw_items.get("mask"))

        # Load pointmap and get both dict and raw pointmap
        raw_pointmap = self._load_pointmap(idx, rgb_image)

        # Use preprocessor to process image, mask, and pointmap together
        # This creates both 'pointmap' and 'rgb_pointmap' fields
        image_dict = (
            self.latent_loader_dataset.preprocessor._process_image_mask_pointmap_mess(
                rgb_image, rgb_image_mask, raw_pointmap
            )
        )

        # Update raw_items with processed data
        raw_items.update(image_dict)

        # the scale in Trellis loading stored the original scaling; we inverse here
        raw_items["instance_scale_l2c"] = 1.0 / raw_items["instance_scale_l2c"]
        return idx, raw_items
