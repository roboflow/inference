from typing import List, Optional, Tuple

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


def _column_to_runs(column: np.ndarray) -> List[Tuple[int, int]]:
    """Split a single 1-D column into (value, run_length) pairs of consecutive equal values."""
    if column.shape[0] == 0:
        return []
    change_points = np.flatnonzero(column[1:] != column[:-1]) + 1
    boundaries = np.concatenate(([0], change_points, [column.shape[0]]))
    return [
        (int(column[start]), int(end - start))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def _embed_single_mask_counts(
    column_major_slice: np.ndarray,
    offset_xy: Tuple[int, int],
    target_size_hw: Tuple[int, int],
) -> List[int]:
    """Build the column-major uncompressed COCO counts for one slice placed onto the canvas.

    ``column_major_slice`` is the dense (h, w) slice already laid out as a list of its w
    columns (each of height h). The big (H, W) canvas is never densified: we emit run
    lengths directly and merge adjacent equal-value runs.
    """
    h, w = column_major_slice.shape
    x0, y0 = offset_xy
    target_h, target_w = target_size_hw

    bottom_zeros = target_h - h - y0
    leading_zero_pixels = x0 * target_h
    trailing_zero_pixels = (target_w - w - x0) * target_h

    # (value, run_length) pairs in column-major order across the whole canvas.
    runs: List[Tuple[int, int]] = []
    if leading_zero_pixels > 0:
        runs.append((0, leading_zero_pixels))
    for column_index in range(w):
        column = column_major_slice[:, column_index]
        if y0 > 0:
            runs.append((0, y0))
        runs.extend(_column_to_runs(column))
        if bottom_zeros > 0:
            runs.append((0, bottom_zeros))
    if trailing_zero_pixels > 0:
        runs.append((0, trailing_zero_pixels))

    # COCO uncompressed counts are alternating run lengths starting with a zero run.
    merged_counts: List[int] = [0]
    current_value = 0
    for value, length in runs:
        if length == 0:
            continue
        if value == current_value:
            merged_counts[-1] += length
        else:
            merged_counts.append(length)
            current_value = value
    return merged_counts


def embed_rle_masks_in_larger_canvas(
    masks: InstancesRLEMasks,
    offset_xy: Tuple[int, int],
    target_size_hw: Tuple[int, int],
) -> InstancesRLEMasks:
    """Place N slice-resolution RLE masks onto a larger all-zeros canvas, in RLE.

    Each input mask sits at slice resolution ``masks.image_size == (h, w)``. The returned
    masks live on a ``(H, W)`` canvas with the slice's top-left at ``offset_xy == (x0, y0)``;
    everything outside the slice is zero. COCO RLE here is column-major (fortran), matching
    ``torch_mask_to_coco_rle``. The big canvas is never densified -- only the small slice is
    decoded to dense and run lengths are emitted directly onto the canvas.
    """
    h, w = masks.image_size
    x0, y0 = offset_xy
    target_h, target_w = target_size_hw

    if x0 < 0 or y0 < 0:
        raise ValueError(
            f"embed_rle_masks_in_larger_canvas got negative offset_xy={offset_xy}; "
            f"offsets must be non-negative."
        )
    if x0 + w > target_w or y0 + h > target_h:
        raise ValueError(
            f"Slice of size (h={h}, w={w}) at offset_xy=(x0={x0}, y0={y0}) does not fit "
            f"into target canvas of size (H={target_h}, W={target_w}): requires "
            f"x0+w={x0 + w} <= W={target_w} and y0+h={y0 + h} <= H={target_h}."
        )

    if len(masks.masks) == 0:
        return InstancesRLEMasks(image_size=(target_h, target_w), masks=[])

    # Decode only the small (h, w) slices to dense; the big canvas stays in run-list form.
    dense_slices = coco_rle_masks_to_numpy_mask(masks).astype(np.uint8)

    embedded: List[bytes] = []
    for dense_slice in dense_slices:
        # Lay the (h, w) slice out as columns (column j of height h) -> shape (h, w).
        counts = _embed_single_mask_counts(
            column_major_slice=dense_slice,
            offset_xy=offset_xy,
            target_size_hw=target_size_hw,
        )
        rle = mask_utils.frPyObjects(
            {"counts": counts, "size": [target_h, target_w]}, target_h, target_w
        )
        embedded.append(rle["counts"])

    return InstancesRLEMasks(image_size=(target_h, target_w), masks=embedded)
