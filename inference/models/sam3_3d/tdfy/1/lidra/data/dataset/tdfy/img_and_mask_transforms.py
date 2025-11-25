from collections import namedtuple
import random
from typing import Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional
from lidra.data.dataset.tdfy.img_processing import pad_to_square_centered
from lidra.model.backbone.dit.embedder.point_remapper import PointRemapper
from typing import Optional, Dict
from loguru import logger
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tv_transforms
import torchvision.transforms.functional
import torchvision.transforms.functional as TF

from lidra.data.dataset.tdfy.img_processing import pad_to_square_centered


def UNNORMALIZE(mean, std):
    mean = torch.tensor(mean).reshape((3, 1, 1))
    std = torch.tensor(std).reshape((3, 1, 1))

    def unnormalize_img(img):
        assert img.ndim == 3 and img.shape[0] == 3

        return img * std.to(img.device) + mean.to(img.device)

    return unnormalize_img


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


IMAGENET_NORMALIZATION = tv_transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
IMAGENET_UNNORMALIZATION = UNNORMALIZE(IMAGENET_MEAN, IMAGENET_STD)


class BoundingBoxError(Exception):
    pass


def check_bounding_box(bbox_w, bbox_h):
    if bbox_w < 2 or bbox_h < 2:
        raise BoundingBoxError("Bounding box dimensions must be at least 2x2.")


class RGBAImageProcessor:
    def __init__(
        self,
        resize_and_make_square_kwargs: Optional[Dict] = None,
        object_crop_kwargs: Optional[Dict] = None,
        remove_background: bool = False,
        imagenet_normalization: bool = False,
    ):
        self.remove_background = remove_background
        self.resize_and_pad_kwargs = resize_and_make_square_kwargs
        self.object_crop_kwargs = object_crop_kwargs
        self.imagenet_normalization = imagenet_normalization
        if resize_and_make_square_kwargs is not None:
            self.transforms = resize_and_make_square(**resize_and_make_square_kwargs)

    def __call__(
        self, image: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            assert (
                image.shape[0] == 4
            ), f"Requires 4 channels (RGB + alpha), got {image.shape[0]=}"
            image, mask = split_rgba(image)
        else:
            assert (
                image.shape[0] == 3
            ), f"Requires 3 channels (RGB), got {image.shape[0]=}"
            assert mask.dim() == 2, f"Requires 2D mask, got {mask.dim()=}"

        if not self.object_crop_kwargs in [None, False]:
            image, mask = crop_around_mask_with_padding(
                image, mask, **self.object_crop_kwargs
            )

        if self.remove_background:
            image, mask = rembg(image, mask)

        image = self.transforms["img_transform"](image)
        mask = self.transforms["mask_transform"](mask.unsqueeze(0))

        if self.imagenet_normalization:
            image = IMAGENET_NORMALIZATION(image)
        return image, mask


def load_rgb(fpath: str) -> torch.Tensor:
    """
    Load a RGB(A) image from a file path.
    """
    image = plt.imread(fpath)  # Why use matplotlib?
    if image.dtype == "uint8":
        image = image / 255
        image = image.astype(np.float32)
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1).contiguous()
    return image


def concat_rgba(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Create a 4-channel RGBA image from a 3-channel RGB image and a mask.
    """
    assert rgb_image.dim() == 3, f"{rgb_image.shape=}"
    assert mask.dim() == 2, f"{mask.shape=}"
    assert rgb_image.shape[0] == 3, f"{rgb_image.shape[0]=}"
    assert rgb_image.shape[1:] == mask.shape, f"{rgb_image.shape[1:]=} != {mask.shape=}"
    return torch.cat((rgb_image, mask[None, ...]), dim=0)


def split_rgba(rgba_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a 4-channel RGBA image into a 3-channel RGB image and a 1-channel mask.

    Args:
        rgba_image: A 4-channel RGBA image.

    Returns:
        A tuple of (rgb_image, mask).
    """
    assert rgba_image.dim() == 3, f"{rgba_image.shape=}"
    assert rgba_image.shape[0] == 4, f"{rgba_image.shape[0]=}"
    return rgba_image[:3], rgba_image[3]


def get_mask(
    rgb_image: torch.Tensor,
    depth_image: torch.Tensor,
    mask_source: str,
) -> torch.Tensor:
    """
    Extract a mask from either the alpha channel of an RGB image or a depth image.

    Args:
        rgb_image: Tensor of shape (B, C, H, W) or (C, H, W) where C >= 4 if using alpha channel
        depth_image: Tensor of shape (B, 1, H, W) or (1, H, W) containing depth information
        mask_source: Source of the mask, either "ALPHA_CHANNEL" or "DEPTH"

    Returns:
        mask: Tensor of shape (B, 1, H, W) or (1, H, W) containing the extracted mask
    """
    # Handle unbatched inputs (add batch dimension if needed)
    is_batched = len(rgb_image.shape) == 4

    if not is_batched:
        rgb_image = rgb_image.unsqueeze(0)
        if depth_image is not None:
            depth_image = depth_image.unsqueeze(0)

    if mask_source == "ALPHA_CHANNEL":
        if rgb_image.shape[1] != 4:
            logger.warning(f"No ALPHA CHANNEL for the image, cannot read mask.")
            mask = None
        else:
            mask = rgb_image[:, 3:4, :, :]
    elif mask_source == "DEPTH":
        mask = depth_image
    else:
        raise ValueError(f"Invalid mask source: {mask_source}")

    # Remove batch dimension if input was unbatched
    if not is_batched:
        mask = mask.squeeze(0)

    return mask


def rembg(image, mask, pointmap=None):
    """
    Remove the background from an image using a mask.
    For pointmaps, sets background regions to NaN.

    This function follows the standard transform pattern:
    - If called with (image, mask), returns (image, mask)
    - If called with (image, mask, pointmap), returns (image, mask, pointmap)
    """
    masked_image = image * mask

    if pointmap is not None:
        masked_pointmap = torch.where(mask > 0, pointmap, torch.nan)
        return masked_image, mask, masked_pointmap

    return masked_image, mask


def resize_and_make_square(
    img_size: int,
    make_square: bool | str = False,
):
    """
    Create image and mask transforms based on configuration.

    Returns:
        dict: {"img_transform": img_transform, "mask_transform": mask_transform}
    """
    if isinstance(make_square, str):
        make_square = make_square.lower()
    assert make_square in ["pad", "crop", False]
    pre_resize_transform = tv_transforms.Lambda(lambda x: x)
    post_resize_transform = tv_transforms.Lambda(lambda x: x)
    if make_square == "pad":
        pre_resize_transform = pad_to_square_centered
    elif make_square == "crop":
        post_resize_transform = tv_transforms.CenterCrop(img_size)

    img_resize = tv_transforms.Resize(img_size)
    mask_resize = tv_transforms.Resize(
        img_size,
        interpolation=tv_transforms.InterpolationMode.BILINEAR,
    )

    img_transform = tv_transforms.Compose(
        [
            pre_resize_transform,
            img_resize,
            post_resize_transform,
        ]
    )

    mask_transform = tv_transforms.Compose(
        [
            pre_resize_transform,
            mask_resize,
            post_resize_transform,
        ]
    )

    return {
        "img_transform": img_transform,
        "mask_transform": mask_transform,
    }


def crop_around_mask_with_random_box_size_factor(
    loaded_image: torch.Tensor,
    mask: torch.Tensor,
    random_box_size_factor: float = 1.0,
    pointmap: Optional[torch.Tensor] = None,
) -> np.ndarray:
    return crop_around_mask_with_padding(
        loaded_image,
        mask,
        box_size_factor=1.0 + random.uniform(0, 1) * random_box_size_factor,
        padding_factor=0.0,
        pointmap=pointmap,
    )


def crop_around_mask_with_padding(
    loaded_image: torch.Tensor,
    mask: torch.Tensor,
    box_size_factor: float = 1.6,
    padding_factor: float = 0.1,
    pointmap: Optional[torch.Tensor] = None,
) -> np.ndarray:
    # cast to ensure the function can be called normally
    cast_mask = False
    if mask.dim() == 3:
        assert mask.shape[0] == 1, "cannot take mask with channel dimension not 1"
        mask = mask[0]
        cast_mask = True
    loaded_image = concat_rgba(loaded_image, mask)

    bbox = compute_mask_bbox(mask, box_size_factor)
    loaded_image = torchvision.transforms.functional.crop(
        loaded_image, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]
    )

    # Crop pointmap if provided
    if pointmap is not None:
        pointmap = torchvision.transforms.functional.crop(
            pointmap, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]
        )

    C, H, W = loaded_image.shape
    max_dim = max(H, W)  # Get the larger dimension

    # Step 1: Pad to square shape
    pad_h = (max_dim - H) // 2
    pad_w = (max_dim - W) // 2
    pad_h_extra = (max_dim - H) - pad_h  # To ensure even padding
    pad_w_extra = (max_dim - W) - pad_w

    loaded_image = torch.nn.functional.pad(
        loaded_image, (pad_w, pad_w_extra, pad_h, pad_h_extra), mode="constant", value=0
    )
    if pointmap is not None:
        pointmap = torch.nn.functional.pad(
            pointmap,
            (pad_w, pad_w_extra, pad_h, pad_h_extra),
            mode="constant",
            value=float("nan"),
        )

    # Step 2: Extend by 10% on each side; idk but this seems to have better results overall
    if padding_factor > 0:
        extend_size = int(max_dim * padding_factor)  # 10% extension on each side
        loaded_image = torch.nn.functional.pad(
            loaded_image,
            (extend_size, extend_size, extend_size, extend_size),
            mode="constant",
            value=0,
        )

        if pointmap is not None:
            pointmap = torch.nn.functional.pad(
                pointmap,
                (extend_size, extend_size, extend_size, extend_size),
                mode="constant",
                value=float("nan"),
            )

    rgb_image, mask = split_rgba(loaded_image)
    if cast_mask:
        mask = mask[None]

    if pointmap is not None:
        return rgb_image, mask, pointmap
    return rgb_image, mask


def compute_mask_bbox(
    mask: torch.Tensor, box_size_factor: float = 1.0
) -> tuple[float, float, float, float]:
    """
    Compute a bounding box around a binary mask with optional size adjustment.

    Args:
        mask: A 2D binary tensor where non-zero values represent the object of interest.
        box_size_factor: Factor to scale the bounding box size. Values > 1.0 create a larger box.
            Default is 1.0 (tight bounding box).

    Returns:
        A tuple of (x1, y1, x2, y2) coordinates representing the bounding box,
        where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Raises:
        ValueError: If mask is not a torch.Tensor or not a 2D tensor.
    """
    if not isinstance(mask, torch.Tensor):
        raise ValueError("Mask must be a torch.Tensor")
    if not mask.dim() == 2:
        raise ValueError("Mask must be a 2D tensor")
    bbox_indices = torch.nonzero(mask)
    if bbox_indices.numel() == 0:
        # Handle empty mask case
        return (0, 0, 0, 0)

    y_indices = bbox_indices[:, 0]
    x_indices = bbox_indices[:, 1]

    min_x = torch.min(x_indices).item()
    min_y = torch.min(y_indices).item()
    max_x = torch.max(x_indices).item()
    max_y = torch.max(y_indices).item()

    bbox = (min_x, min_y, max_x, max_y)

    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2

    bbox_w, bbox_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    check_bounding_box(bbox_w, bbox_h)

    size = max(bbox_w, bbox_h, 2)
    size = int(size * box_size_factor)

    bbox = (
        int(center_x - size // 2),
        int(center_y - size // 2),
        int(center_x + size // 2),
        int(center_y + size // 2),
    )
    # bbox = tuple(map(int, bbox))
    return bbox


def crop_and_pad(image, bbox):
    """
    Crop an image using a bounding box and pad with zeros if out of bounds.

    Args:
        image (torch.Tensor): CxHxW image.
        bbox (tuple): (x1, y1, x2, y2) bounding box.

    Returns:
        torch.Tensor: Cropped and zero-padded image.
    """
    C, H, W = image.shape
    x1, y1, x2, y2 = bbox

    # Ensure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Compute cropping coordinates
    x1_pad, y1_pad = max(0, -x1), max(0, -y1)
    x2_pad, y2_pad = max(0, x2 - W), max(0, y2 - H)

    # Compute valid region in the original image
    x1_crop, y1_crop = max(0, x1), max(0, y1)
    x2_crop, y2_crop = min(W, x2), min(H, y2)

    # Extract the valid part
    cropped = image[:, y1_crop:y2_crop, x1_crop:x2_crop]

    # Create a zero-padded output
    padded = torch.zeros((C, y2 - y1, x2 - x1), dtype=image.dtype)

    # Place the cropped image into the zero-padded array
    padded[
        :, y1_pad : y1_pad + cropped.shape[1], x1_pad : x1_pad + cropped.shape[2]
    ] = cropped

    return padded


def resize_all_to_same_size(
    rgb_image: torch.Tensor,
    mask: torch.Tensor,
    pointmap: Optional[torch.Tensor] = None,
    target_size: Optional[tuple[int, int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Resize RGB image, mask, and pointmap to the same size.

    This is crucial when pointmaps have different resolution than RGB images,
    which must be done BEFORE any cropping operations.

    Args:
        rgb_image: RGB image tensor of shape (C, H, W)
        mask: Mask tensor of shape (H, W) or (1, H, W)
        pointmap: Optional pointmap tensor of shape (C_p, H_p, W_p)
        target_size: Target size as (H, W). If None, uses RGB image size.

    Returns:
        Tuple of (resized_rgb, resized_mask, resized_pointmap)
    """
    squeeze_mask = mask.dim() == 2
    if squeeze_mask:
        mask = mask.unsqueeze(0)

    if target_size is None:
        target_size = (rgb_image.shape[1], rgb_image.shape[2])  # (H, W)

    rgb_needs_resize = (rgb_image.shape[1], rgb_image.shape[2]) != target_size
    if rgb_needs_resize:
        rgb_image = torchvision.transforms.functional.resize(
            rgb_image,
            target_size,
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        mask = torchvision.transforms.functional.resize(
            mask,
            target_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )

    if pointmap is not None:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        if pointmap_size != target_size:
            # Handle NaN values in pointmap during resizing
            # Direct resize would propagate NaN values, so we need special handling
            nan_mask = torch.isnan(pointmap).any(dim=0)
            pointmap_clean = torch.where(
                torch.isnan(pointmap), torch.zeros_like(pointmap), pointmap
            )
            pointmap_resized = torchvision.transforms.functional.resize(
                pointmap_clean,
                target_size,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )

            # Resize the nan mask to identify which regions should remain invalid
            nan_mask_resized = (
                torchvision.transforms.functional.resize(
                    nan_mask.unsqueeze(0).float(),
                    target_size,
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ).squeeze(0)
                > 0.5
            )

            # Restore NaN values in regions that were originally invalid
            pointmap = torch.where(
                nan_mask_resized.unsqueeze(0).expand_as(pointmap_resized),
                torch.full_like(pointmap_resized, float("nan")),
                pointmap_resized,
            )

    if squeeze_mask:
        mask = mask.squeeze(0)

    if pointmap is not None:
        return rgb_image, mask, pointmap
    return rgb_image, mask


SSINormalizedPointmap = namedtuple(
    "SSINormalizedPointmap", ["pointmap", "scale", "shift"]
)


class SSIPointmapNormalizer:

    def normalize(
        self,
        pointmap: torch.Tensor,
        mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> SSINormalizedPointmap:
        if scale is None or shift is None:
            normalized_pointmap, scale, shift = normalize_pointmap_ssi(pointmap)
        else:
            assert scale.shape == (3,) and shift.shape == (
                3,
            ), "scale and shift must be in (3,) format"
            normalized_pointmap = _apply_metric_to_ssi(pointmap, scale, shift)
        return SSINormalizedPointmap(normalized_pointmap, scale, shift)

    def denormalize(
        self, pointmap: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        pointmap = _apply_metric_to_ssi(pointmap, scale, shift, apply_inverse=True)
        return pointmap


class ObjectCentricSSI(SSIPointmapNormalizer):
    def __init__(
        self,
        use_scene_scale: bool = True,
        quantile_drop_threshold: float = 0.1,
        clip_beyond_scale: Optional[float] = None,
        # scale_factor: float = 3.8076, # e^(1.337); empirical mean of R3+Artist train
        scale_factor: float = 1.0,  # e^(1.337); empirical mean of R3+Artist train
        allow_scale_and_shift_override: bool = False,
        raise_on_no_valid_points: bool = False,
    ):
        self.use_scene_scale = use_scene_scale
        self.quantile_drop_threshold = quantile_drop_threshold
        self.clip_beyond_scale = clip_beyond_scale
        self.scale_factor = scale_factor
        self.allow_scale_and_shift_override = allow_scale_and_shift_override
        self.raise_on_no_valid_points = raise_on_no_valid_points

    def _compute_scale_and_shift(
        self, pointmap: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])

        mask_resized = torchvision.transforms.functional.resize(
            mask,
            pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        # Get valid points from the mask
        mask_bool = mask_resized.reshape(-1) > 0.5
        mask_points = pointmap_flat[:, mask_bool]

        if mask_points.isfinite().max() == 0:
            if self.raise_on_no_valid_points:
                raise ValueError(f"No valid points found in mask")
            logger.warning(
                f"No valid points found in mask; setting scale to {self.scale_factor} and shift to 0"
            )
            return torch.ones_like(
                pointmap_flat[:, 0]
            ) * self.scale_factor, torch.zeros_like(pointmap_flat[:, 0])

        # Compute median for shift
        shift = mask_points.nanmedian(dim=-1).values
        # logger.info(f"{pointmap.shape=} {mask_resized.shape=} {shift.shape=}")

        if self.use_scene_scale == True:
            # Normalize by the scene scale
            points_centered = pointmap_flat - shift.unsqueeze(-1)
            max_dims = points_centered.abs().max(dim=0).values
            scale = max_dims.nanmedian(dim=-1).values
        elif self.use_scene_scale == False:
            # Normalize by the object scale
            shifted_mask_points = mask_points - shift.unsqueeze(-1)
            norm = shifted_mask_points.norm(dim=0)
            quantiles = torch.nanquantile(
                norm,
                torch.tensor(
                    [self.quantile_drop_threshold, 1.0 - self.quantile_drop_threshold],
                    device=shifted_mask_points.device,
                ),
                dim=-1,
            )
            scale = (quantiles[1] - quantiles[0]).max(dim=-1).values * 2.0
        elif self.use_scene_scale.upper() == "OBJECT_NORM_MEDIAN":
            # Normalize by the object scale
            shifted_mask_points = mask_points - shift.unsqueeze(-1)
            norm = shifted_mask_points.norm(dim=0)
            scale = norm.nanmedian(dim=-1).values
        else:
            raise ValueError(f"Invalid use_scene_scale: {self.use_scene_scale}")
        scale = scale.expand_as(shift)  # per-dim scaling
        scale = scale * self.scale_factor
        return scale, shift

    def normalize(
        self,
        pointmap: torch.Tensor,
        mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. resize mask to size of pointmap using nearest interpolation
        # 2. get mask points: pointmap[mask > 0.5]
        # 3. shift = mask_points.median() # xyz
        # 4. scale = # filter. If no points, then
        # logger.info(f"{pointmap.shape=} {mask.shape=}")
        assert pointmap.shape[0] == 3, "pointmap must be in (3, H, W) format"
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])

        _scale, _shift = self._compute_scale_and_shift(pointmap, mask)
        if scale is not None and self.allow_scale_and_shift_override:
            _scale = scale
        if shift is not None and self.allow_scale_and_shift_override:
            _shift = shift
        return_scale, return_shift = _scale, _shift

        # Apply normalization
        pointmap_normalized = _apply_metric_to_ssi(pointmap, return_scale, return_shift)

        if self.clip_beyond_scale is not None and self.clip_beyond_scale > 0:
            new_norm = pointmap_normalized.norm(dim=0)
            pointmap_normalized = torch.where(
                new_norm > self.clip_beyond_scale,
                torch.full_like(pointmap_normalized, float("nan")),
                pointmap_normalized,
            )

        return SSINormalizedPointmap(pointmap_normalized, return_scale, return_shift)


class ObjectApparentSizeSSI(SSIPointmapNormalizer):
    def __init__(
        self,
        clip_beyond_scale: Optional[float] = None,
        use_scene_scale: bool = True,
        scale_factor: float = 1.0,  # e^(1.337); empirical mean of R3+Artist train
    ):
        self.clip_beyond_scale = clip_beyond_scale
        self.use_scene_scale = use_scene_scale
        self.scale_factor = scale_factor

    def _get_scale_and_shift(
        self, pointmap: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        pointmap_flat = pointmap.reshape(3, -1)

        if not self.use_scene_scale:
            # Get valid points from the mask
            mask_resized = torchvision.transforms.functional.resize(
                mask,
                pointmap_size,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ).squeeze(0)
            mask_bool = mask_resized.reshape(-1) > 0.5
            pointmap_flat = pointmap_flat[:, mask_bool]

        # Median z-distance
        median_z = pointmap_flat[-1, ...].nanmedian().unsqueeze(0)
        scale = median_z.expand(3) * self.scale_factor
        shift = torch.zeros_like(scale)
        # logger.info(f'median z = {median_z}')
        return scale, shift

    def normalize(
        self,
        pointmap: torch.Tensor,
        mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert pointmap.shape[0] == 3, "pointmap must be in (3, H, W) format"
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])

        if scale is None or shift is None:
            scale, shift = self._get_scale_and_shift(pointmap, mask)
        else:
            assert scale.shape == (3,) and shift.shape == (
                3,
            ), "scale and shift must be in (3,) format"

        # Apply normalization and clip
        pointmap_normalized = _apply_metric_to_ssi(pointmap, scale, shift)
        # logger.info(f"{pointmap_normalized.shape=}")

        if self.clip_beyond_scale is not None and self.clip_beyond_scale > 0:
            pointmap_normalized = torch.where(
                pointmap_normalized[-1, ...] > self.clip_beyond_scale,
                torch.full_like(pointmap_normalized, float("nan")),
                pointmap_normalized,
            )

        # return pointmap_normalized, scale, shift
        return SSINormalizedPointmap(pointmap_normalized, scale, shift)


class NormalizedDisparitySpaceSSI(SSIPointmapNormalizer):
    def __init__(
        self,
        clip_beyond_scale: Optional[float] = None,
        use_scene_scale: bool = True,
        log_disparity_shift: float = 0.0,
    ):
        self.clip_beyond_scale = clip_beyond_scale
        self.use_scene_scale = use_scene_scale
        self.point_remapper = PointRemapper(remap_type="exp_disparity")
        self.log_disparity_shift = log_disparity_shift

    def normalize(
        self,
        pointmap: torch.Tensor,
        mask: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert pointmap.shape[0] == 3, "pointmap must be in (3, H, W) format"

        disparity_space_pointmap = self.point_remapper.forward(
            pointmap.permute(1, 2, 0)
        ).permute(2, 0, 1)
        if scale is None or shift is None:
            scale, shift = self._get_scale_and_shift(disparity_space_pointmap, mask)
        else:
            assert scale.shape == (3,) and shift.shape == (
                3,
            ), "scale and shift must be in (3,) format"

        # pointmap_normalized = pointmap.clone().detach()
        pointmap_normalized = _apply_metric_to_ssi(
            disparity_space_pointmap, scale, shift
        )
        # logger.info(f"{pointmap_normalized.shape=}")

        if self.clip_beyond_scale is not None and self.clip_beyond_scale > 0:
            pointmap_normalized = torch.where(
                pointmap_normalized[2, ...].abs() > self.clip_beyond_scale,
                torch.full_like(pointmap_normalized, float("nan")),
                pointmap_normalized,
            )

        # return pointmap_normalized, scale, shift
        return SSINormalizedPointmap(pointmap_normalized, scale, shift)

    def denormalize(
        self, pointmap: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        pointmap = _apply_metric_to_ssi(pointmap, scale, shift, apply_inverse=True)
        pointmap = self.point_remapper.inverse(pointmap.permute(1, 2, 0)).permute(
            2, 0, 1
        )
        return pointmap

    def _get_scale_and_shift(
        self, pointmap: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        mask_resized = torchvision.transforms.functional.resize(
            mask,
            pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        ).squeeze(0)

        pointmap_flat = pointmap.reshape(3, -1)
        if self.use_scene_scale:
            median_z = pointmap_flat[-1, ...].nanmedian().unsqueeze(0)
            shift = torch.zeros_like(median_z.expand(3))
            shift[-1, ...] = median_z[0] + self.log_disparity_shift
        else:
            # Get valid points from the mask (shift, x/z, y/z, log(z))
            mask_bool = mask_resized.reshape(-1) > 0.5
            pointmap_flat = pointmap_flat[:, mask_bool]
            shift = pointmap_flat.nanmedian(dim=-1).values

        scale = torch.ones_like(shift)
        # logger.info(f'median z = {median_z}')
        return scale, shift


def normalize_pointmap_ssi(
    pointmap: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize pointmap using Scale-Shift Invariant (SSI) normalization.

    Args:
        pointmap: Pointmap tensor of shape (H, W, 3) or (3, H, W)

    Returns:
        Tuple of (normalized_pointmap, scale, shift)
    """
    from lidra.data.dataset.tdfy.pose_target import ScaleShiftInvariant

    # Convert to (H, W, 3) if needed for get_scale_and_shift
    if pointmap.shape[0] == 3:
        pointmap_hw3 = pointmap.permute(1, 2, 0)
        original_format = "chw"
    else:
        pointmap_hw3 = pointmap
        original_format = "hwc"

    # Get scale and shift using existing method
    scale, shift = ScaleShiftInvariant.get_scale_and_shift(pointmap_hw3)

    pointmap_normalized = _apply_metric_to_ssi(pointmap, scale, shift)
    return pointmap_normalized, scale, shift


def _apply_metric_to_ssi(
    pointmap: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    apply_inverse: bool = False,
) -> torch.Tensor:
    """
    Normalize pointmap using Scale-Shift Invariant (SSI) normalization.

    Args:
        pointmap: Pointmap tensor of shape (H, W, 3) or (3, H, W)

    Returns:
        Tuple of (normalized_pointmap, scale, shift)
    """
    from lidra.data.dataset.tdfy.pose_target import ScaleShiftInvariant

    # Convert to (H, W, 3) if needed for get_scale_and_shift
    if pointmap.shape[0] == 3:
        pointmap_hw3 = pointmap.permute(1, 2, 0)
        original_format = "chw"
    else:
        pointmap_hw3 = pointmap
        original_format = "hwc"

    # Apply normalization
    ssi_to_metric = ScaleShiftInvariant.ssi_to_metric(scale, shift)
    metric_to_ssi = ssi_to_metric.inverse()
    transform_to_apply = metric_to_ssi

    if apply_inverse:
        transform_to_apply = ssi_to_metric

    pointmap_flat = pointmap_hw3.reshape(-1, 3)
    pointmap_normalized = transform_to_apply.transform_points(pointmap_flat)

    # Reshape back to original format
    if original_format == "chw":
        pointmap_normalized = pointmap_normalized.reshape(
            pointmap.shape[1], pointmap.shape[2], 3
        ).permute(2, 0, 1)
    else:
        pointmap_normalized = pointmap_normalized.reshape(pointmap_hw3.shape)

    return pointmap_normalized


def perturb_mask_translation(
    image: torch.Tensor,
    mask: torch.Tensor,
    max_px_delta: int = 5,
):
    """
    Applies data augmentation to the mask by randomly translating the mask.

    Args:
        image: (C, H, W) float32 [0, 1] tensor.
        mask: (1, H, W) float32 [0, 1] tensor.
        max_px_delta: The maximum number of pixels we will randomly shift by in each 2D direction.
    """
    dx = random.randint(-max_px_delta, max_px_delta)
    dy = random.randint(-max_px_delta, max_px_delta)

    mask = mask.squeeze(0)
    mask = torch.roll(mask, shifts=(dy, dx), dims=(0, 1))

    # Zero out wrapped regions
    if dy > 0:
        mask[:dy, :] = 0
    elif dy < 0:
        mask[dy:, :] = 0
    if dx > 0:
        mask[:, :dx] = 0
    elif dx < 0:
        mask[:, dx:] = 0

    mask = mask.unsqueeze(0)
    return image, mask


def perturb_mask_boundary(
    image: torch.Tensor,
    mask: torch.Tensor,
    kernel_range: tuple[int, int] = (2, 5),
    p_erode: float = 0.1,
    p_dilate: float = 0.8,
    **kwargs,
):
    """
    Applies data augmentation to the mask by randomly eroding or dilating the mask.

    Args:
        image: (C, H, W) float32 [0, 1] tensor.
        mask: (1, H, W) float32 [0, 1] tensor.
        kernel_range: Range of kernel sizes to sample from.
        p_erode: Probability of erosion.
        p_dilate: Probability of dilation.
        kwargs: Kwargs for the cv2 erode/dilate function.
    """
    import cv2

    C, H, W = image.shape
    assert mask.shape == (1, H, W)
    assert mask.dtype == torch.float32
    assert torch.all((mask == 0) | (mask == 1)), "Mask must be binary (0 or 1)"

    p_none = 1.0 - p_erode - p_dilate
    assert 0 <= p_none <= 1, "Probabilities must sum to 1 and be valid."

    # Sample operation.
    op = random.choices(
        ["erode", "dilate", "none"], weights=[p_erode, p_dilate, p_none], k=1
    )[0]

    if op == "none":
        pass
    else:
        # Sample kernel size
        ksize = random.randint(*kernel_range)
        kernel = np.ones((ksize, ksize), np.uint8)

        mask = mask.squeeze().cpu().numpy().astype(np.uint8)  # (H, W)

        if op == "erode":
            mask = cv2.erode(mask, kernel, **kwargs)
        elif op == "dilate":
            mask = cv2.dilate(mask, kernel, **kwargs)
        else:
            raise NotImplementedError

        mask = torch.from_numpy(mask).float()[None]  # (1, H, W)

    return image, mask


def resolution_blur(
    image: torch.Tensor,
    mask: torch.Tensor,
    scale_range=(0.05, 0.95),
    interpolation_down=tv_transforms.InterpolationMode.BICUBIC,
    interpolation_up=tv_transforms.InterpolationMode.BICUBIC,
):
    """
    Blur the input image by applying upsample(downsample(x)).

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W), float32, with values in [0, 1].
        mask (torch.Tensor): Mask tensor of shape (1, H, W), float32, with values in [0, 1]. The mask is returned unchanged.
        scale_range: Tuple of (min_scale, max_scale) for downsampling.
        interpolation_down: Interpolation mode for downsampling.
        interpolation_up: Interpolation mode for upsampling.
    """
    C, H, W = image.shape
    scale = random.uniform(*scale_range)
    new_H, new_W = max(1, int(H * scale)), max(1, int(W * scale))

    # Downsample
    image = TF.resize(image, size=[new_H, new_W], interpolation=interpolation_down)

    # Upsample back to original size
    image = TF.resize(image, size=[H, W], interpolation=interpolation_up)

    return image, mask


def gaussian_blur(
    image: torch.Tensor,
    mask: torch.Tensor,
    kernel_range: tuple[int, int] = (3, 15),
    sigma_range: tuple[int, int] = (0.1, 4.0),
):
    """
    Apply gaussian blur to the input image.

    Args:
        image (torch.Tensor): Image tensor of shape (C, H, W), float32, with values in [0, 1].
        mask (torch.Tensor): Mask tensor of shape (1, H, W), float32, with values in [0, 1]. The mask is returned unchanged.
        kernel_range (tuple): Range of odd kernel sizes to sample from for the Gaussian blur (min, max).
        sigma_range (tuple): Range of sigma values (standard deviation) to sample from for the Gaussian kernel (min, max).
    """
    kernel_size = random.choice(
        [k for k in range(kernel_range[0], kernel_range[1] + 1) if k % 2 == 1]
    )
    sigma = random.uniform(*sigma_range)
    pad = kernel_size // 2

    # Step 1: Pad the image
    image = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="replicate")

    # Step 2: Apply gaussian blur
    image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=sigma)

    # Step 3: Unpad to get back to original size
    image = image[:, :, pad:-pad, pad:-pad]

    return image.squeeze(0), mask


def apply_blur_augmentation(
    image: torch.Tensor,
    mask: torch.Tensor,
    p_resolution: float = 0.33,
    p_gaussian: float = 0.33,
    gaussian_kwargs: dict = None,
    resolution_kwargs: dict = None,
):
    """Apply blur augmentation with configurable parameters"""

    # Handle None defaults BEFORE unpacking
    if gaussian_kwargs is None:
        gaussian_kwargs = {}
    if resolution_kwargs is None:
        resolution_kwargs = {}

    p_none = 1.0 - p_gaussian - p_resolution
    assert 0 <= p_none <= 1, "Probabilities must sum to 1 and be valid."

    operation = random.choices(
        ["gaussian", "resolution", "none"],
        weights=[p_gaussian, p_resolution, p_none],
        k=1,
    )[0]

    if operation == "gaussian":
        return gaussian_blur(image, mask, **gaussian_kwargs)
    elif operation == "resolution":
        return resolution_blur(image, mask, **resolution_kwargs)
    elif operation == "none":
        return image, mask
    else:
        raise NotImplementedError
