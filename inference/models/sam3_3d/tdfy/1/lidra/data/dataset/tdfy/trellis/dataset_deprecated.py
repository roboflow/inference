import numpy as np
import pandas as pd
import random
import torch
import os
import math
import warnings
from loguru import logger
from pytorch3d.transforms import quaternion_to_matrix
from ..metadata_filter import custom_metadata_filter
import torchvision.transforms as tv_transforms
from ..img_and_mask_transforms import (
    get_mask,
    rembg,
    crop_around_mask_with_padding,
    load_rgb,
)


from ..img_processing import (
    pad_to_square_centered,
    crop_img_to_obj,
    random_pad,
    get_img_color_augmentation,
)
from lidra.utils.decorators.counter import garbage_collect
from lidra.data.dataset.tdfy.transforms_3d import get_rotation_about_x_axis

from .latent_loader import (
    load_sparse_feature_latents,
    load_structure_latents,
)

from .pose_loader import R3, load_trellis_pose, empty_pose


class PerSubsetDataset(torch.utils.data.Dataset):
    VALID_SPLITS = {"train", "val"}
    mesh_to_latent_quat = get_rotation_about_x_axis(-math.pi / 2)

    def __init__(
        self,
        path,
        split,
        latent_type="structure",
        max_num_voxels=25000,
        resnet_normalization=False,
        img_size=518,
        pad=True,
        mask_source="ALPHA_CHANNEL",
        tight_obj_boundary=False,
        random_pad=0.0,
        use_color_aug=False,
        metadata_fname="metadata.csv",
        metadata_filter=custom_metadata_filter(None),
        remove_img_bg=True,
        box_size_factor=1.0,
        padding_factor=0.1,
    ):
        warnings.warn(
            "The old PerSubsetDataset/ TrellisDataset is deprecated! Please upgrade",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.path = path
        # TODO Fujen: make sure this is the right way to read subset_name
        self.subset_name = path.split("/")[-1]
        self.split = split
        self.latent_type = latent_type

        if latent_type == "structure":
            self.latent_dir = os.path.join(
                self.path, "ss_latents/ss_enc_conv3d_16l8_fp16"
            )
        elif latent_type == "features":
            self.latent_dir = os.path.join(
                self.path, "latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16"
            )
        else:
            assert (
                latent_type == "features"
            ), f"{latent_type} not supported for latent type"

        assert (
            split in PerSubsetDataset.VALID_SPLITS
        ), f"split should be in {PerSubsetDataset.VALID_SPLITS}"

        self.metadata = pd.read_csv(os.path.join(self.path, metadata_fname))
        self.metadata = metadata_filter(self.metadata)

        # avoid OOM
        self.max_num_voxels = max_num_voxels
        self._load_uids()

        # Set transforms
        self.img_size = img_size
        self.resnet_normalization = resnet_normalization
        self.pad = pad

        self.mask_source = mask_source
        self.remove_img_bg = remove_img_bg
        assert mask_source in ["ALPHA_CHANNEL", "DEPTH"], f"{mask_source} not supported"
        self.tight_obj_boundary = tight_obj_boundary
        self.random_pad = random_pad
        self.use_color_aug = use_color_aug
        self.box_size_factor = box_size_factor
        self.padding_factor = padding_factor
        self._set_transforms()

    def _set_transforms(self):
        # # TODO: refactor this to use lidra.data.dataset.tdfy.img_and_mask_transforms.RGBAImageProcessor

        # Image transforms
        self.mask_transform = None
        if self.pad == True:
            self.img_transform = tv_transforms.Compose(
                [
                    pad_to_square_centered,
                    tv_transforms.Resize(self.img_size),
                ]
            )
            self.mask_transform = tv_transforms.Compose(
                [
                    pad_to_square_centered,
                    tv_transforms.Resize(
                        self.img_size,
                        interpolation=tv_transforms.InterpolationMode.NEAREST,
                    ),
                ]
            )
        elif self.pad == "ResizeShortestSide":
            self.img_transform = tv_transforms.Compose(
                [
                    tv_transforms.Resize(self.img_size),
                ]
            )
            self.mask_transform = tv_transforms.Compose(
                [
                    tv_transforms.Resize(
                        self.img_size,
                        interpolation=tv_transforms.InterpolationMode.NEAREST,
                    ),
                ]
            )
        else:
            self.img_transform = tv_transforms.Compose(
                [
                    tv_transforms.Resize(self.img_size),
                    tv_transforms.CenterCrop(self.img_size),
                ]
            )
            self.mask_transform = self.img_transform

        if self.use_color_aug:
            self.img_transform = tv_transforms.Compose(
                [
                    self.img_transform,
                    get_img_color_augmentation(),
                ]
            )

        if self.resnet_normalization:
            self.img_transform = tv_transforms.Compose(
                [
                    self.img_transform,
                    tv_transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )

        if self.tight_obj_boundary and self.mask_transform is None:
            raise ValueError(
                "Mask transform is not set. Please set return_mask to False or use pad='ResizeShortestSide'."
            )

    def _load_uids(self):
        self.uids = self.metadata["sha256"].values.tolist()
        # TODO: refactor to add splits here

        if self.latent_type == "structure":
            self.uids = list(
                self.metadata[
                    (self.metadata["num_voxels"] > 0) & (self.metadata["cond_rendered"])
                ]["sha256"]
            )
        else:
            self.uids = list(
                self.metadata[
                    (self.metadata["num_voxels"] < self.max_num_voxels)
                    & (self.metadata["num_voxels"] > 0)
                    & (self.metadata["cond_rendered"])
                ]["sha256"]
            )

    def __len__(self) -> int:
        return len(self.uids)

    @staticmethod
    def get_sha256(metadata: pd.DataFrame, file_identifier: str):
        return metadata[metadata["file_identifier"] == file_identifier][
            "sha256"
        ].values[0]

    def _load_latent(self, sha256: str, latent_type: str):
        latent_path = os.path.join(self.latent_dir, sha256 + ".npz")
        item = np.load(latent_path)
        if latent_type == "structure" or latent_type == "ss":
            item = load_structure_latents(item)
        elif latent_type == "features":
            item = load_sparse_feature_latents(item)
        else:
            raise ValueError(f"Latent type {latent_type} not supported")
        return item

    def _ensure_mask_binary(self, mask, threshold=0):
        return (mask > threshold).float()

    def _read_mask(self, rgba_image):
        mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        return self._ensure_mask_binary(mask)

    def _get_cond_image_dir(self, uid):
        return os.path.join(self.path, "renders_cond", uid)

    def _load_available_images(self, uid):
        # load image
        try:
            image_dir = self._get_cond_image_dir(uid)
            available_views = [
                f
                for f in os.listdir(image_dir)
                if f.endswith(".png") and os.path.isfile(os.path.join(image_dir, f))
            ]
            # TODO: weiyaowang clean this up once we standardize our data format
            if len(available_views) == 0:
                available_views = [
                    f
                    for f in os.listdir(image_dir)
                    if f.endswith("_rgb0001.jpg")
                    and os.path.isfile(os.path.join(image_dir, f))
                ]
            assert len(available_views) > 0

            return available_views
        except:
            logger.opt(exception=True).error(
                f"error while loading image file : {image_dir}"
            )
            return None

    def _load_available_cameras(self, uid):
        pose_dir = os.path.join(self.path, "renders_cond", uid)
        transforms_path = os.path.join(pose_dir, "transforms.json")
        if self.subset_name.startswith("r3") or self.subset_name.startswith("rp"):
            # Loading R3 pose not supported in deprecated dataset
            available_cameras = empty_pose(transforms_path)
        else:
            available_cameras = load_trellis_pose(transforms_path)

        return available_cameras

    def _sample_view(self, available_views, available_cameras):
        sampled_idx = random.choice(tuple(range(len(available_views))))
        sampled_view = available_views[sampled_idx]
        sampled_camera = available_cameras[sampled_view]

        return sampled_view, sampled_camera

    # TODO: we should merge the two functions. This one is used for training
    # We should verify by retraining a model with the refactored dataset
    def _preprocess_image_and_mask(self, rgb_image, mask, remove_img_bg=True):
        if self.tight_obj_boundary:
            left, right, top, bot = crop_img_to_obj(mask[0], 0)
            if left is not None:
                rgb_image = rgb_image[:, top:bot, left:right]
                mask = mask[:, top:bot, left:right]

        if self.random_pad > 0:
            rgb_image, mask = random_pad(rgb_image, mask, max_ratio=self.random_pad)

        if remove_img_bg and mask is not None:
            rgb_image, _ = rembg(rgb_image, mask)

        return rgb_image, mask

    # TODO: we should merge the two functions. This one is used for eval
    def _preprocess_image_and_mask_inference(self, rgb_image, mask_image):
        if self.tight_obj_boundary:
            if self.remove_img_bg:
                rgb_image, _ = rembg(rgb_image, mask_image)
            rgb_image, mask_image = crop_around_mask_with_padding(
                rgb_image,
                mask_image.squeeze(0),
                box_size_factor=self.box_size_factor,
                padding_factor=self.padding_factor,
            )
            mask_image = mask_image[None]

        return rgb_image, mask_image

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None:
            input = transform(input)

        return input

    @garbage_collect()
    def __getitem__(self, index):
        item = {}
        uid = self.uids[index]
        item.update(self._load_latent(uid, self.latent_type))

        # read available views if there are multiple images under render_cond
        available_views = self._load_available_images(uid)
        available_cameras = self._load_available_cameras(uid)

        # TODO: need to have a dataset that have every image, and then sample from it
        sampled_view_id, sampled_camera = self._sample_view(
            available_views, available_cameras
        )
        img_path = os.path.join(self._get_cond_image_dir(uid), sampled_view_id)
        rgba_image = load_rgb(img_path)
        rgb_image = rgba_image[:3]
        rgb_image_mask = self._read_mask(rgba_image)
        processed_rgb_image, processed_mask = self._preprocess_image_and_mask(
            rgb_image, rgb_image_mask, self.remove_img_bg
        )

        # transform tensor to model input
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, self.img_transform
        )
        processed_mask = self._apply_transform(processed_mask, self.mask_transform)

        rgb_image = self._apply_transform(rgb_image, self.img_transform)
        rgb_image_mask = self._apply_transform(rgb_image_mask, self.mask_transform)

        item["mask"] = processed_mask
        item["image"] = processed_rgb_image
        item["rgb_image"] = rgb_image
        item["rgb_image_mask"] = rgb_image_mask
        # # NOTE: No support for pose in deprecated dataset
        # item["camera_R"] = quaternion_to_matrix(
        #     sampled_camera["instance_quaternion_l2c"].squeeze(0)
        # )
        # item["camera_T"] = sampled_camera["instance_position_l2c"].squeeze(0)

        return img_path, item
