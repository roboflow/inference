from collections import namedtuple
import json
import numpy as np
import math
import pandas as pd
import random
import warnings
import torch
import os
from PIL import Image
from loguru import logger
from dataclasses import dataclass
from typing import Callable, Optional, Union
import warnings

from ..metadata_filter import custom_metadata_filter
from ..img_and_mask_transforms import (
    get_mask,
    load_rgb,
    normalize_pointmap_ssi,
    SSINormalizedPointmap,
    SSIPointmapNormalizer,
)

from lidra.utils.decorators.counter import garbage_collect
from lidra.data.dataset.tdfy.pose_target import ScaleShiftInvariant
from lidra.data.dataset.tdfy.transforms_3d import (
    compose_transform,
    get_rotation_about_x_axis,
)

from .dataset_deprecated import PerSubsetDataset as PerSubsetDatasetDeprecated

from .latent_loader import load_structure_latents
from .pose_loader import load_trellis_pose
from .mesh_loader import load_trellis_mesh


# Load R3
@dataclass
class PreProcessor:
    """
    Preprocessor configuration for image, mask, and pointmap transforms.

    Transform application order:
    1. Pointmap normalization (if normalize_pointmap=True)
    2. Joint transforms (img_mask_pointmap_joint_transform or img_mask_joint_transform)
    3. Individual transforms (img_transform, mask_transform, pointmap_transform)

    For backward compatibility, img_mask_joint_transform is preserved. When both
    img_mask_pointmap_joint_transform and img_mask_joint_transform are present,
    img_mask_pointmap_joint_transform takes priority.
    """

    img_transform: Callable = (None,)
    mask_transform: Callable = (None,)
    img_mask_joint_transform: list[Callable] = (None,)
    rgb_img_mask_joint_transform: list[Callable] = (None,)

    # New fields for pointmap support
    pointmap_transform: Callable = (None,)
    img_mask_pointmap_joint_transform: list[Callable] = (None,)

    # Pointmap normalization option
    normalize_pointmap: bool = False
    pointmap_normalizer: Optional[Callable] = None
    rgb_pointmap_normalizer: Optional[Callable] = None

    def __post_init__(self):
        if self.pointmap_normalizer is None:
            self.pointmap_normalizer = SSIPointmapNormalizer()
            if self.normalize_pointmap == False:
                warnings.warn(
                    "normalize_pointmap is also set to False, which means we will return the moments but not normalize the pointmap. This supports old unnormalized pointmap models, but this is dangerous behavior.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if self.rgb_pointmap_normalizer is None:
            logger.warning("No rgb pointmap normalizer provided, using scale + shift ")
            self.rgb_pointmap_normalizer = self.pointmap_normalizer

    def _normalize_pointmap(
        self,
        pointmap: torch.Tensor,
        mask: torch.Tensor,
        pointmap_normalizer: Callable,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ):
        if pointmap is None:
            return pointmap, None, None

        if self.normalize_pointmap == False:
            # old behavior: Pose is normalized to the pointmap center, but pointmap is not
            _, pointmap_scale, pointmap_shift = pointmap_normalizer.normalize(
                pointmap, mask
            )
            return pointmap, pointmap_scale, pointmap_shift

        if scale is not None or shift is not None:
            return pointmap_normalizer.normalize(pointmap, mask, scale, shift)

        return pointmap_normalizer.normalize(pointmap, mask)

    def _process_image_mask_pointmap_mess(
        self, rgb_image, rgb_image_mask, pointmap=None
    ):
        """Extended version that handles pointmaps"""

        # Apply pointmap normalization if enabled
        pointmap_for_crop, pointmap_scale, pointmap_shift = self._normalize_pointmap(
            pointmap, rgb_image_mask, self.pointmap_normalizer
        )

        # Apply transforms to the original full rgb image and mask.
        rgb_image, rgb_image_mask = self._preprocess_rgb_image_mask(
            rgb_image, rgb_image_mask
        )

        # These two are typically used for getting cropped images of the object
        #   : first apply joint transforms
        processed_rgb_image, processed_mask, processed_pointmap = (
            self._preprocess_image_mask_pointmap(
                rgb_image, rgb_image_mask, pointmap_for_crop
            )
        )
        #   : then apply individual transforms on top of the joint transforms
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, self.img_transform
        )
        processed_mask = self._apply_transform(processed_mask, self.mask_transform)
        if processed_pointmap is not None:
            processed_pointmap = self._apply_transform(
                processed_pointmap, self.pointmap_transform
            )

        # This version is typically the full version of the image
        #   : apply individual transforms only
        rgb_image = self._apply_transform(rgb_image, self.img_transform)
        rgb_image_mask = self._apply_transform(rgb_image_mask, self.mask_transform)

        rgb_pointmap, rgb_pointmap_scale, rgb_pointmap_shift = self._normalize_pointmap(
            pointmap,
            rgb_image_mask,
            self.rgb_pointmap_normalizer,
            pointmap_scale,
            pointmap_shift,
        )

        if rgb_pointmap is not None:
            rgb_pointmap = self._apply_transform(rgb_pointmap, self.pointmap_transform)

        result = {
            "mask": processed_mask,
            "image": processed_rgb_image,
            "rgb_image": rgb_image,
            "rgb_image_mask": rgb_image_mask,
        }

        # Add pointmap results if available
        if processed_pointmap is not None:
            result.update(
                {
                    "pointmap": processed_pointmap,
                    "rgb_pointmap": rgb_pointmap,
                }
            )

        # Add normalization parameters if normalization was applied
        if pointmap_scale is not None and pointmap_shift is not None:
            result.update(
                {
                    "pointmap_scale": pointmap_scale,
                    "pointmap_shift": pointmap_shift,
                    "rgb_pointmap_scale": rgb_pointmap_scale,
                    "rgb_pointmap_shift": rgb_pointmap_shift,
                }
            )

        return result

    def _process_image_and_mask_mess(self, rgb_image, rgb_image_mask):
        """Original method - calls extended version without pointmap"""
        return self._process_image_mask_pointmap_mess(rgb_image, rgb_image_mask, None)

    def _preprocess_rgb_image_mask(
        self, rgb_image: torch.Tensor, rgb_image_mask: torch.Tensor
    ):
        """Apply joint transforms to rgb_image and rgb_image_mask."""
        if (
            self.rgb_img_mask_joint_transform != (None,)
            and self.rgb_img_mask_joint_transform is not None
        ):
            for trans in self.rgb_img_mask_joint_transform:
                rgb_image, rgb_image_mask = trans(rgb_image, rgb_image_mask)
        return rgb_image, rgb_image_mask

    def _preprocess_image_mask_pointmap(self, rgb_image, mask_image, pointmap=None):
        """Apply joint transforms with priority: triple transforms > dual transforms."""
        # Priority: img_mask_pointmap_joint_transform when pointmap is provided
        if (
            self.img_mask_pointmap_joint_transform != (None,)
            and self.img_mask_pointmap_joint_transform is not None
            and pointmap is not None
        ):
            for trans in self.img_mask_pointmap_joint_transform:
                rgb_image, mask_image, pointmap = trans(
                    rgb_image, mask_image, pointmap=pointmap
                )
            return rgb_image, mask_image, pointmap

        # Fallback: img_mask_joint_transform (existing behavior)
        elif (
            self.img_mask_joint_transform != (None,)
            and self.img_mask_joint_transform is not None
        ):
            for trans in self.img_mask_joint_transform:
                rgb_image, mask_image = trans(rgb_image, mask_image)
            return rgb_image, mask_image, pointmap

        return rgb_image, mask_image, pointmap

    def _preprocess_image_and_mask(self, rgb_image, mask_image):
        """Backward compatibility wrapper - only applies dual transforms"""
        rgb_image, mask_image, _ = self._preprocess_image_mask_pointmap(
            rgb_image, mask_image, None
        )
        return rgb_image, mask_image

    # keep here for backward compatibility
    def _preprocess_image_and_mask_inference(self, rgb_image, mask_image):
        warnings.warn(
            "The _preprocess_image_and_mask_inference is deprecated! Please use _preprocess_image_and_mask",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self._preprocess_image_and_mask(rgb_image, mask_image)

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None and transform is not None and transform != (None,):
            input = transform(input)

        return input


PerSubsetSampleID = namedtuple("PerSubsetSampleID", ["sha256", "image_fname"])


class PerSubsetDataset(torch.utils.data.Dataset):
    VALID_SPLITS = {"train", "val"}

    mesh_to_latent_quat = get_rotation_about_x_axis(-math.pi / 2)

    def __init__(
        self,
        path: str,
        split: str,
        preprocessor: PreProcessor,
        metadata_fname: str = "metadata.csv",
        metadata_filter: Callable[
            [pd.DataFrame], pd.DataFrame
        ] = custom_metadata_filter(None),
        pose_loader: Callable = load_trellis_pose,
        # ss_latents/ss_enc_conv3d_16l8_fp16 or
        # latents/dinov2_vitl14_reg_slat_enc_swin8_B_64l8_fp16
        latent_dir: str = "ss_latents/ss_enc_conv3d_16l8_fp16",
        latent_loader: Callable = load_structure_latents,
        mesh_loader: Optional[Callable] = None,
        pointmap_loader: Optional[Callable] = None,
        voxel_loader: Optional[Callable] = None,
        # Pointmaps and meshes can be variable size;
        # TODO: To collate them we should pad with nans
        # In the meantime, we exclude them from the dict
        return_pointmap: bool = False,
        return_mesh: bool = False,
        load_pointmap_render_paste: bool = False,
    ):
        self.path = path
        self.split = split
        self.latent_dir = os.path.join(self.path, latent_dir)

        assert (
            split in PerSubsetDataset.VALID_SPLITS
        ), f"split should be in {PerSubsetDataset.VALID_SPLITS}"

        self.metadata_fname = metadata_fname
        self.metadata_filter = metadata_filter
        self.metadata = pd.read_csv(os.path.join(self.path, metadata_fname))
        self.metadata = metadata_filter(self.metadata)
        self.uids = self.metadata["sha256"].values.tolist()

        # set pose loader
        self.pose_loader = pose_loader
        if self.pose_loader is None:
            logger.warning("No pose_loader provided. Poses will not be loaded!")

        # set latent_loader
        self.latent_loader = latent_loader

        # set mesh_loader
        self.mesh_loader = mesh_loader
        self.pointmap_loader = pointmap_loader
        self.load_pointmap_render_paste = load_pointmap_render_paste

        self.return_pointmap = return_pointmap
        self.return_mesh = return_mesh
        self.voxel_loader = voxel_loader

        # set transforms
        self.preprocessor = preprocessor
        self.img_transform = preprocessor.img_transform
        self.mask_transform = preprocessor.mask_transform
        self.img_mask_joint_transform = preprocessor.img_mask_joint_transform

    def __len__(self) -> int:
        return len(self.uids)

    @staticmethod
    def get_sha256(metadata: pd.DataFrame, file_identifier: str):
        return metadata[metadata["file_identifier"] == file_identifier][
            "sha256"
        ].values[0]

    @staticmethod
    def image_fpath_to_sample_id(fpath: str) -> PerSubsetSampleID:
        sha256, img_id = fpath.split("/")[-2:]
        return PerSubsetSampleID(sha256=sha256, image_fname=img_id)

    def _load_latent(self, sha256: str):
        latent_path = os.path.join(self.latent_dir, sha256 + ".npz")
        item = np.load(latent_path)
        item = self.latent_loader(item)
        return item

    def _load_mesh(self, sha256: str):
        if self.mesh_loader is None or not self.return_mesh:
            return {}
        mesh_path = os.path.join(self.path, "renders", sha256, "mesh.ply")
        trellis_mesh = self.mesh_loader(mesh_path)
        return trellis_mesh

    def _load_pointmap(self, sha256: str, rgb_image: torch.Tensor, image_fname: str):
        if self.pointmap_loader is None:
            return None
            return self._dummy_pointmap_moments()

        # Subset-specific loading
        # TODO: remove this paragraph
        row = self.metadata[self.metadata["sha256"] == sha256]

        if "image_basename" in self.metadata.columns:  # For R3
            image_basename = str(row.image_basename.item())
        else:  # For elephant in the room

            image_metadata_path = os.path.join(
                self._get_cond_image_dir(sha256), "image_metadata.csv"
            )
            image_metadata_df = pd.read_csv(image_metadata_path)
            image_basename_no_ext = os.path.splitext(image_fname)[0]
            image_basename = str(
                image_metadata_df[
                    image_metadata_df["local_basename"] == image_basename_no_ext
                ].image_basename.item()
            )

        if self.load_pointmap_render_paste:
            file_identifier = row.file_identifier.item()
            pointmap = self.pointmap_loader(
                image_basename, sha256=sha256, file_identifier=file_identifier
            )
        else:
            # Load pointmap (will auto-detect moge vs moge_corrected format)
            pointmap = self.pointmap_loader(
                image_basename, sha256=sha256, image_fname=image_fname
            )
        if not torch.isfinite(pointmap).any():
            raise ValueError("Pointmap detected to be all nans")
        pointmap = pointmap.permute(2, 0, 1)
        return pointmap

    def _dummy_pointmap_moments(self):
        return {
            "pointmap_scale": torch.tensor(1.0).expand(3),
            "pointmap_shift": torch.zeros(1, 3),
        }

    @staticmethod
    def _get_pointmap_scale_and_shift(pointmap: torch.Tensor, mask: torch.Tensor):
        assert pointmap.shape[-1] == 3, "Pointmap must be 3D"

        return ScaleShiftInvariant.get_scale_and_shift(pointmap)

    @staticmethod
    def _prepare_pointmap(
        pointmap: torch.Tensor, return_pointmap: bool, mask: torch.Tensor
    ):
        pointmap_scale, pointmap_shift = PerSubsetDataset._get_pointmap_scale_and_shift(
            pointmap, mask
        )
        if torch.isnan(pointmap_scale).any() or torch.isnan(pointmap_shift).any():
            logger.warning(
                "NaN values detected in pointmap_scale or pointmap_shift (pointmap must be all nans)"
            )
            return None
        return_dict = {
            "pointmap_scale": pointmap_scale,
            "pointmap_shift": pointmap_shift,
        }
        if return_pointmap:
            pointmap = pointmap.permute(2, 0, 1)
            return_dict["pointmap"] = pointmap
        return return_dict

    @staticmethod
    def _get_pointmap_colors(raw_image: torch.Tensor, pointmap: torch.Tensor):
        # Get the colors of the pointmap
        warnings.warn(
            "The _get_pointmap_colors method is deprecated and will be removed in a future version. "
            "Pointmaps should be pixel-aligned with RGB images.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        if raw_image.shape[0] == 3:
            raw_image = raw_image.permute(1, 2, 0)
        colors_tensor = Image.fromarray(
            (raw_image[..., :3] * 255).numpy().astype(np.uint8)
        ).resize((pointmap.shape[1], pointmap.shape[0]))
        colors_tensor = torch.from_numpy(np.array(colors_tensor)).float() / 255.0
        return colors_tensor

    def _load_available_poses(self, uid, view_id=None):
        pose_dir = os.path.join(self.path, "renders_cond", uid)
        transforms_path = os.path.join(pose_dir, "transforms.json")
        available_poses = self.pose_loader(transforms_path, view_id)
        return available_poses

    def _load_pose(self, sha256: str, view_id: str):
        if self.pose_loader is None:
            return {}

        available_poses = self._load_available_poses(sha256, view_id)
        sampled_pose = available_poses[view_id]
        self._validate_pose(sha256, view_id, sampled_pose)
        return sampled_pose

    def _validate_pose(self, sha256: str, view_id: str, sampled_pose: dict):
        instance_position_l2c = sampled_pose.get("instance_position_l2c", None)
        if instance_position_l2c is not None:
            if instance_position_l2c.norm(dim=-1) < 1e-6:
                raise ValueError(
                    f"Object position is centered at camera for sha256: {sha256} and view_id: {view_id}"
                )

    def _load_mesh_to_latent_transform(self, sha256: str):
        render_dir = os.path.join(self.path, "renders", sha256)
        transforms_json = os.path.join(render_dir, "transforms.json")

        with open(transforms_json, "r") as f:
            mesh_to_latent_transform = json.load(f)

        return compose_transform(
            mesh_to_latent_transform["scale"] * torch.ones([1, 3]),
            torch.eye(3),
            torch.tensor(mesh_to_latent_transform["offset"]).unsqueeze(0),
        )

    def _load_voxel(self, sha256: str):
        if self.voxel_loader is None:
            return {}
        voxel_path = os.path.join(self.path, "voxels", f"{sha256}.ply")
        trellis_voxel = self.voxel_loader(voxel_path)
        return trellis_voxel

    def _ensure_mask_binary(self, mask, threshold=0):
        return (mask > threshold).float()

    def _read_mask(self, rgba_image):
        mask = get_mask(rgba_image, None, "ALPHA_CHANNEL")

        return self._ensure_mask_binary(mask)

    def _get_cond_image_dir(self, uid):
        return os.path.join(self.path, "renders_cond", uid)

    def _load_available_images(self, uid):
        try:
            image_dir = self._get_cond_image_dir(uid)
            # Use scandir instead of listdir so we don't need to read each file twice
            available_views = [
                entry.name
                for entry in os.scandir(image_dir)
                if entry.is_file(follow_symlinks=True) and entry.name.endswith(".png")
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

    def _sample_view(self, sha256: str):
        available_views = self._load_available_images(sha256)
        sampled_idx = random.randint(0, len(available_views) - 1)
        sampled_view = available_views[sampled_idx]
        return sampled_view

    def _get_sample_uuid(self, index: Union[int, str, PerSubsetSampleID]):
        if isinstance(index, tuple):
            return PerSubsetSampleID(*index)
        elif isinstance(index, str):
            return self.image_fpath_to_sample_id(index)
        elif isinstance(index, int):
            return PerSubsetSampleID(self.uids[index], None)
        raise ValueError(f"Invalid index type: {type(index)}")

    @garbage_collect()
    def __getitem__(self, index):
        try:
            sample_uuid = self._get_sample_uuid(index)

            # read available views if there are multiple images under render_cond
            if sample_uuid.image_fname is None:
                sha256 = sample_uuid.sha256

                # Check if renders_cond directory exists before attempting to sample
                image_dir = self._get_cond_image_dir(sha256)
                if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
                    logger.warning(f"Image directory not found: {image_dir}")
                    return None

                image_fname = self._sample_view(sha256)
                sample_uuid = PerSubsetSampleID(sha256=sha256, image_fname=image_fname)

            return self.compute_item(sample_uuid)
        except:
            logger.opt(exception=True).error(f"error for idx {index} in {sample_uuid}")
            return None

    def _process_image_and_mask_mess(self, rgb_image, rgb_image_mask):
        warnings.warn(
            "The _process_image_and_mask_mess is deprecated! Please use preprocessor._process_image_mask_pointmap_mess",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, None
        )

    def _preprocess_image_and_mask_inference(self, rgb_image, rgb_image_mask):
        warnings.warn(
            "The _preprocess_image_and_mask_inference is deprecated! Please use preprocessor._preprocess_image_and_mask_inference",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.preprocessor._preprocess_image_and_mask_inference(
            rgb_image, rgb_image_mask
        )

    def compute_item(self, sample_uuid: PerSubsetSampleID):
        uid = sample_uuid.sha256
        image_fname = sample_uuid.image_fname

        img_path = os.path.join(self._get_cond_image_dir(uid), image_fname)
        rgba_image = load_rgb(img_path)
        rgb_image = rgba_image[:3]
        rgb_image_mask = self._read_mask(rgba_image)

        # This must use the raw image!!! Before cropping and padding
        raw_pointmap = self._load_pointmap(uid, rgb_image, image_fname)

        # # Not all processors
        # pointmap_dict = PerSubsetDataset._prepare_pointmap(raw_pointmap, self.return_pointmap, rgb_image_mask)
        # pointmap = pointmap_dict.pop("pointmap", None)

        # How the images are processed (into crops, padded, etc)
        image_dict = self.preprocessor._process_image_mask_pointmap_mess(
            rgb_image, rgb_image_mask, raw_pointmap
        )

        sampled_pose = self._load_pose(uid, image_fname)
        latent_dict = self._load_latent(uid)
        mesh_dict = self._load_mesh(uid)
        voxel_dict = self._load_voxel(uid)

        item = {}
        item.update(latent_dict)
        item.update(image_dict)
        item.update(sampled_pose)
        item.update(mesh_dict)
        item.update(voxel_dict)
        return img_path, item


# This is deprecated; leaving here for backward compatibility
class TrellisDataset(torch.utils.data.Dataset):
    VALID_SPLITS = {"train", "val"}

    def __init__(
        self,
        path,
        split,
        subsets,
        latent_type="structure",
        max_num_voxels=25000,
        resnet_normalization=False,
        img_size=518,
        pad=True,
        tight_obj_boundary=False,
        random_pad=0.0,
        use_color_aug=False,
        metadata_fname="metadata.csv",
        metadata_filters=None,
        remove_img_bg=True,
        box_size_factor=1.0,
        padding_factor=0.1,
    ):
        assert (
            split in PerSubsetDatasetDeprecated.VALID_SPLITS
        ), f"split should be in {PerSubsetDatasetDeprecated.VALID_SPLITS}"
        if metadata_filters is not None:
            assert len(metadata_filters) == len(subsets), (
                f"Error: The number of metadata filters ({len(metadata_filters)}) "
                f"does not match the number of subsets ({len(subsets)})."
            )

        subset_datasets = []
        for i, subset in enumerate(subsets):
            subset_dir = os.path.join(path, subset)
            assert os.path.exists(subset_dir), f"{subset_dir} does not exist!"
            subset_dataset = PerSubsetDatasetDeprecated(
                subset_dir,
                split,
                latent_type,
                max_num_voxels,
                resnet_normalization=resnet_normalization,
                img_size=img_size,
                pad=pad,
                tight_obj_boundary=tight_obj_boundary,
                random_pad=random_pad,
                use_color_aug=use_color_aug,
                metadata_fname=metadata_fname,
                metadata_filter=(
                    metadata_filters[i]
                    if metadata_filters is not None
                    else custom_metadata_filter(None)
                ),
                remove_img_bg=remove_img_bg,
                box_size_factor=box_size_factor,
                padding_factor=padding_factor,
            )
            subset_datasets.append(subset_dataset)
        self.dataset = torch.utils.data.dataset.ConcatDataset(subset_datasets)

    def __len__(self) -> int:
        return len(self.dataset)

    # defer garbage collection to individual dataset
    def __getitem__(self, index):
        return self.dataset[index]


# TODO Hao: make this a generic overfit dataset
class TrellisDatasetOverfit(TrellisDataset):
    def __init__(
        self,
        path,
        split,
        subsets,
        latent_type="structure",
        max_num_voxels=25000,
        resnet_normalization=False,
        img_size=518,
        pad=True,
    ):
        super().__init__(
            path,
            split,
            subsets,
            latent_type,
            max_num_voxels,
            resnet_normalization,
            img_size,
            pad,
        )

    def __len__(self) -> int:
        return 1000

    # defer garbage collection to individual dataset
    def __getitem__(self, index):
        return self.dataset[0]
