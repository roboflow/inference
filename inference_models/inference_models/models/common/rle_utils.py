from typing import Optional

import numpy as np
import torch
from pycocotools import mask as mask_utils

from inference_models.models.base.types import InstancesRLEMasks


def torch_mask_to_coco_rle(mask: torch.Tensor) -> dict:
    # Convert to uncompressed run length encoding in GPU
    # coco tools expect fortran order (column-wise)
    mask_flat = mask.permute(1, 0).reshape(-1)
    values, lengths = torch.unique_consecutive(mask_flat, return_counts=True)
    counts = lengths.cpu().tolist()

    if values[0] == 1:
        counts.insert(0, 0)

    h, w = mask.shape
    # compress
    rle = mask_utils.frPyObjects({"counts": counts, "size": [h, w]}, h, w)
    return rle


def coco_rle_masks_to_numpy_mask(instances_masks: InstancesRLEMasks) -> np.ndarray:
    if len(instances_masks.masks) == 0:
        return np.empty(
            (0, instances_masks.image_size[0], instances_masks.image_size[1]),
            dtype=bool,
        )
    return np.ascontiguousarray(
        mask_utils.decode(instances_masks.to_coco_rle_masks())
        .transpose(2, 0, 1)
        .astype(bool)
    )


def coco_rle_masks_to_torch_mask(
    instances_masks: InstancesRLEMasks, device: Optional[torch.device] = None
) -> torch.Tensor:
    if len(instances_masks.masks) == 0:
        return torch.empty(
            size=(0, instances_masks.image_size[0], instances_masks.image_size[1]),
            dtype=torch.bool,
            device=device,
        )
    return torch.from_numpy(
        np.ascontiguousarray(
            mask_utils.decode(instances_masks.to_coco_rle_masks())
            .transpose(2, 0, 1)
            .astype(bool)
        )
    ).to(device=device, dtype=torch.bool)
