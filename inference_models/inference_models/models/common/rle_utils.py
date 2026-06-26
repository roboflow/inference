from typing import List, Optional

import numpy as np
import torch
from pycocotools import mask as mask_utils

from inference_models.models.base.types import InstancesRLEMasks


def torch_mask_to_coco_rle(mask: torch.Tensor) -> dict:
    # Convert to uncompressed run length encoding in GPU
    # coco tools expect fortran order (column-wise)
    mask_flat = mask.permute(1, 0).reshape(-1)
    values, lengths = torch.unique_consecutive(mask_flat, return_counts=True)
    counts = lengths.cpu().tolist()  # <-- device->host sync, once per detection

    if values[0] == 1:  # <-- second per-detection sync (reads a GPU scalar)
        counts.insert(0, 0)

    h, w = mask.shape
    # compress
    rle = mask_utils.frPyObjects({"counts": counts, "size": [h, w]}, h, w)
    return rle


def torch_masks_to_coco_rle_batch(masks: torch.Tensor) -> List[dict]:
    """Batch equivalent of ``torch_mask_to_coco_rle`` for an ``[N, H, W]`` stack.

    Encodes every mask to compressed COCO RLE with a SINGLE device->host
    transfer followed by one vectorized ``pycocotools.mask.encode`` call,
    instead of one ``torch_mask_to_coco_rle`` call per detection. Each per-mask
    call does its own ``.cpu()`` sync; on Jetson those per-detection syncs
    serialize the GPU N times per frame and dominate instance-segmentation
    post-processing. The encoded output is identical to encoding each mask
    individually (pycocotools encodes in Fortran/column-major order, matching
    ``torch_mask_to_coco_rle``).
    """
    n = masks.shape[0]
    if n == 0:
        return []
    # pycocotools expects a Fortran-ordered [H, W, N] uint8 array. The single
    # .cpu() here replaces the 2*N per-detection syncs in torch_mask_to_coco_rle.
    masks_hwn = masks.to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
    return mask_utils.encode(np.asfortranarray(masks_hwn))


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
