import json
import os
from collections import namedtuple
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch


Artist3DDatasetSampleID = namedtuple(
    "Artist3DDatasetSampleID", ["artist", "img_obj", "version"]
)


def get_subdirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


class Artist3DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        metadata_fname: str,
    ):
        self.path = path
        self.data_dir = os.path.join(self.path, "../evaluation")

        # Load metadata
        self.metadata = pd.read_csv(os.path.join(self.path, metadata_fname))
        self.metadata = self._filter_metadata()

    def _filter_metadata(
        self,
        cols_to_check=[
            "feature_dinov2_vitl14_reg",
            "ss_latent_ss_enc_conv3d_16l8_fp16",
            "latent_dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16",
        ],
    ):
        # filter out samples where processing has failed at some stage
        return self.metadata[~self.metadata[cols_to_check].eq(False).any(axis=1)]

    def _get_uid(self, artist, img_obj, version):
        #
        file_identifier = f"{artist}/{img_obj}/v{version}"
        sha256 = self.metadata.loc[
            self.metadata["file_identifier"] == file_identifier, "sha256"
        ].values

        assert (
            len(sha256) == 1
        ), f"Did not get single unique sha256 from {file_identifier}; found {len(sha256)}"
        return sha256[0]

    def _idx_2_artist_img_obj_version(self, idx):
        file_identifier = self.metadata.iloc[idx]["file_identifier"]
        artist, img_obj, version = file_identifier.split("/")
        version = int(version.split("v")[1:])

        return artist, img_obj, version

    def _load_img(self, artist, img_obj):
        img_obj_dir = os.path.join(self.data_dir, artist, img_obj)
        img_path = os.path.join(img_obj_dir, "image.jpg")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    def _load_mask(self, artist, img_obj):
        img_obj_dir = os.path.join(self.data_dir, artist, img_obj)
        mask_path = os.path.join(img_obj_dir, "mask.jpg")
        mask = cv2.imread(mask_path)

        mask = torch.from_numpy(mask)[:, :, 0].float().unsqueeze(0) / 255.0
        return mask

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: Union[int, Artist3DDatasetSampleID]) -> dict:
        if isinstance(idx, tuple):
            if not len(idx) == 3:
                raise ValueError(f"Expected 3-tuple, got {len(idx)}")
            sample_uuid = Artist3DDatasetSampleID(*idx)

            artist = idx.artist
            img_obj = idx.img_obj
            version = idx.version
        else:
            artist, img_obj, version = self._idx_2_artist_img_obj_version(idx)
            sample_uuid = Artist3DDatasetSampleID(
                artist=artist, img_obj=img_obj, version=version
            )

        # Load image and mask
        img = self._load_img(artist, img_obj)
        mask = self._load_mask(artist, img_obj)

        return sample_uuid, {
            "image": img,
            "mask": mask,
        }
